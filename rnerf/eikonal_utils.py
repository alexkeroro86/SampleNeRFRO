import gin
import numpy as np
import functools
from typing import Callable, Any

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

from rnerf import math_utils
from rnerf import ior_utils


class OneEikonalStep(nn.Module):
  grid: jnp.ndarray
  step_size: float
  ndim: list
  nmin: list
  nmax: list
  stage: str

  def setup(self):
    self.idx_model = ior_utils.VoxMLP(ndim=self.ndim, nmin=self.nmin, nmax=self.nmax, grid=self.grid)

  def wrapper_idx_model_grad_mlp(self, ray_pos, condition=None, annealed_alpha=1.0):
    return self.idx_model.wrapper_grad_mlp(ray_pos, condition=condition, annealed_alpha=annealed_alpha)

  @nn.compact
  def __call__(self, carry, iterable):
    rp, rd, rt, i, annealed_alpha = carry
    idx_data, idx_grad, pred_grad = self.idx_model(rp, annealed_alpha)

    if self.stage.startswith("all"):
      grad = jnp.where(jnp.linalg.norm(idx_grad, axis=-1, keepdims=True) > 1e-3, pred_grad, idx_grad)
      # # TODO: take all or where silhouette, and so does dataset
      # grad = pred_grad
    else:
      grad = idx_grad

    next_rp = rp + self.step_size / idx_data * rd
    next_rd = rd + self.step_size * grad
    # next_rp = rp + self.step_size * rd
    # next_rd = rd
    next_rt = rt + jnp.linalg.norm(rp - next_rp, axis=-1, keepdims=True)

    carry = (next_rp, next_rd, next_rt, i + 1, annealed_alpha)
    out = jnp.concatenate([next_rp, next_rd, next_rt, idx_data, idx_grad], axis=-1)
    return carry, out


@gin.configurable
class PathSampler(nn.Module):
  num_samples: int
  near: float
  far: float
  stage: str

  grid: jnp.ndarray
  step_size: float
  ndim: list
  nmin: list
  nmax: list

  normal_radius_scale: float = 0.1

  def setup(self):
    self.ndelta = [
      (self.nmax[0] - self.nmin[0]) / (self.ndim[0] - 1.),
      (self.nmax[1] - self.nmin[1]) / (self.ndim[1] - 1.),
      (self.nmax[2] - self.nmin[2]) / (self.ndim[2] - 1.),
    ]

    # https://github.com/google/flax/discussions/1395
    self.scan = nn.scan(functools.partial(OneEikonalStep,
                                          step_size=self.step_size,
                                          ndim=self.ndim, nmin=self.nmin, nmax=self.nmax,
                                          grid=self.grid, stage=self.stage),
                        variable_broadcast=('params'),
                        split_rngs={'params': False},
                        in_axes=0,
                        out_axes=0)()

  def compute_normal_loss_and_smooth(self, ray_pos, idx_grad, annealed_alpha):
    # factor = 1.0 #1.0 / 20.0 if self.stage == "ior" else 1.0 / 100.0
    # ray_pos = stop_gradient(ray_pos)
    # idx_grad = stop_gradient(idx_grad)

    pred_grad = self.scan.apply(self.scan.variables, ray_pos, condition=idx_grad, annealed_alpha=annealed_alpha, method=self.scan.wrapper_idx_model_grad_mlp)
    # loss = jnp.sum(((pred_grad - idx_grad) * factor)**2, axis=-1, keepdims=True).mean()
    
    factor = math_utils.safe_l2_norm(idx_grad)
    pred_grad_rand = self.scan.apply(
      self.scan.variables, ray_pos + jnp.array(np.random.normal(scale=self.normal_radius_scale, size=ray_pos.shape) * np.array(self.ndelta)[None, None]),
      condition=idx_grad, annealed_alpha=annealed_alpha, method=self.scan.wrapper_idx_model_grad_mlp)
    smoothness = jnp.sum(jnp.abs((pred_grad - pred_grad_rand) / factor), axis=-1, keepdims=True).mean()

    return 0.0, smoothness

  @nn.compact
  def __call__(self, origin, direction, annealed_alpha):
    bs = origin.shape[0]

    init_ray_pos = origin + self.near * direction
    init_ray_dir = direction
    init_ray_dist = self.near * jnp.ones((bs, 1))

    carry = init_ray_pos, init_ray_dir, init_ray_dist, 0, annealed_alpha
    carry, out = self.scan(carry, jnp.arange(self.num_samples))  # one more step
    out = jnp.transpose(out, (1, 0, 2))  # [batch, sample, feature]

    ray_pos = jnp.concatenate([init_ray_pos[:, None], out[:, :-1, :3]], axis=1)
    ray_dir = math_utils.safe_l2_normalize(jnp.concatenate([direction[:, None], out[:, :-1, 3:6]], axis=1))
    ray_dist = jnp.concatenate([init_ray_dist[:, None], out[:, :-1, 6:7]], axis=1)
    idx_data = out[..., 7:8]
    idx_grad = out[..., 8:11]

    return (
      ray_pos,  # ray position
      ray_dir,  # ray direction
      stop_gradient(ray_dist[..., 0]),  # ray distance
      idx_data,  # refractive index
      idx_grad,  # gradient of refractive index
    )
