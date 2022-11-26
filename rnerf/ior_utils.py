from typing import Callable, Any
import os
import queue
import threading
import numpy as np
import trimesh
from tqdm import tqdm
from pysdf import SDF  # NOTE: remove 'experimental/propagate_const' for gcc 5.4.0
import gin
import functools

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from rnerf import utils
from rnerf import model_utils
from rnerf import rl_utils
from rnerf import math_utils


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

# # NOTE: trimesh is slow
# class Dataset:
#   def __init__(self, args):
#     super().__init__()
#     self.mesh = trimesh.load(os.path.join(args.data_dir, 'mesh.obj'))
#     self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
#     self.batch_size = args.batch_size // jax.process_count()

#   def __iter__(self):
#     return self

#   def __next__(self):
#     return utils.shard(self._generate_samples())

#   def peek(self):
#     return utils.shard(self._generate_samples())

#   def _generate_samples(self):
#     num_samples = self.batch_size // 4
#     rand_sample = np.random.rand(self.batch_size, 3) * 8. - 4.
#     near_sample, _ = trimesh.sample.sample_surface(self.mesh, num_samples)
#     near_sample += np.random.rand(num_samples, 3) * 0.01
#     surf_sample = trimesh.sample.volume_mesh(self.mesh, num_samples)
#     ns = surf_sample.shape[0]
#     samples = np.concatenate([
#         rand_sample[:(self.batch_size - num_samples - ns)],
#         near_sample, surf_sample], axis=0)
#     labels = self.intersector.contains_points(samples)[..., None]
#     return {'samples': samples, 'labels': labels}

class Dataset(threading.Thread):
  def __init__(self, args):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True

    # TODO: trimesh not work with python threading?
    mesh = trimesh.load(os.path.join(args.data_dir, 'mesh.obj'))
    self.extents = mesh.extents
    self.bounds = mesh.bounds
    self.sdf = SDF(mesh.vertices, mesh.faces)
    self.batch_size = args.batch_size // jax.process_count()
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    x = self.queue.get()
    return utils.shard(x)

  def peek(self):
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    return utils.shard(x)

  def run(self):
    while True:
      num_samples = self.batch_size // 4
      extent = 3
      rand_sample = np.random.rand(self.batch_size // 2, 3) * extent * 2. - extent
      near_sample = self.sdf.sample_surface(num_samples * 2)
      near_sample += np.random.normal(scale=0.01, size=(num_samples * 2, 3))
      # trimesh.sample.volume_mesh
      points = (np.random.random((num_samples, 3)) * self.extents) + self.bounds[0]
      contained = self.sdf.contains(points)
      surf_sample = points[contained][:num_samples]
      ns = surf_sample.shape[0]
      samples = np.concatenate([
          rand_sample[:(self.batch_size // 2 - ns)],
          near_sample], axis=0)
      labels = self.sdf.contains(samples)[..., None]
      labels = np.concatenate([labels.astype(np.float32), np.ones((ns, 1))], axis=0)
      self.queue.put({
          'samples': np.concatenate([samples, surf_sample], axis=0),
          'labels': np.where(labels > 0.5, 1.33, 1.0)})
      
      # # equal-space sampling
      # extent = np.random.rand() * 3. + 0.5
      # X, Y, Z = np.meshgrid(np.linspace(0, 1, 10),
      #                       np.linspace(0, 1, 10),
      #                       np.linspace(0, 1, 10))
      # x_max = y_max = z_max = extent
      # x_min = y_min = z_min = -extent
      # X = X * (x_max - x_min) + x_min
      # Y = Y * (y_max - y_min) + y_min
      # Z = Z * (z_max - z_min) + z_min
      # samples = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
      # labels = self.sdf.contains(samples)[..., None]
      # self.queue.put({
      #     'samples': samples,
      #     'labels': np.where(labels > 0.5, 1.33, 1.0)})

# -----------------------------------------------------------------------------
# Network layers
# -----------------------------------------------------------------------------

@gin.configurable
class VoxMLP(nn.Module):
  ndim: list
  nmin: list
  nmax: list
  grid: jnp.ndarray = struct.field(pytree_node=False)  # https://github.com/google/jax/issues/2588
  interp_method: str = "linear3"
  use_direct_output: bool = True  # True: 3-element vector; False: coefs of spherical domain
  use_residual: bool = True  # True: add to initial value [Animatable NeRF]; False, from scratch [NeRFactor]
  normalized: bool = False  # True: output normal vector; False, output raw vector
  annealed: bool = True
  num_actions: int = 4
  min_deg_point: int = 0
  max_deg_point: int = 10
  # deg_view: int = 4

  def setup(self):
    self.ndelta = [
      (self.nmax[0] - self.nmin[0]) / (self.ndim[0] - 1.),
      (self.nmax[1] - self.nmin[1]) / (self.ndim[1] - 1.),
      (self.nmax[2] - self.nmin[2]) / (self.ndim[2] - 1.),
    ]

    # NOTE(nerfactor)
    num_out_channels = 3 if self.use_direct_output else self.num_actions * self.num_actions * 2
    self.so3_mlp = model_utils.MLP(
      net_width=128, net_depth=4, skip_layer=2,
      num_out_channels=num_out_channels,
      output_init=jax.nn.initializers.normal(stddev=1e-5) if self.use_residual else jax.nn.initializers.xavier_uniform()
    )
    if self.annealed:
      self.embed = functools.partial(model_utils.annealed_pos_enc, min_deg=self.min_deg_point, max_deg=self.max_deg_point)
    else:
      self.embed = functools.partial(model_utils.pos_enc, min_deg=self.min_deg_point, max_deg=self.max_deg_point, legacy_posenc_order=True)
    if not self.use_direct_output:
      action_basis = rl_utils.compute_action_space(self.num_actions, shrink=0.0)
      self.action_basis = jnp.concatenate([action_basis, action_basis * jnp.array([[1, 1, -1]])], axis=0)  # sphere

    self.data = jnp.concatenate([self.grid, self._compute_grad()], axis=-1)
    # self.sigma_activation = lambda x: jax.nn.sigmoid(x) * 2.0
    # self.rgb_activation = lambda x: jax.nn.tanh(x) * 120.0
  
  def _compute_grad(self):
    paaded = jnp.pad(self.grid.reshape(self.ndim[0], self.ndim[1], self.ndim[2], 1)[..., 0],
                     ((1, 1), (1, 1), (1, 1)), "edge")
   # TODO: boundary case is only one step size by central difference
    dx = (paaded[2:, 1:-1, 1:-1] - paaded[:-2, 1:-1, 1:-1]) / (2 * self.ndelta[0])
    dy = (paaded[1:-1, 2:, 1:-1] - paaded[1:-1, :-2, 1:-1]) / (2 * self.ndelta[1])
    dz = (paaded[1:-1, 1:-1, 2:] - paaded[1:-1, 1:-1, :-2]) / (2 * self.ndelta[2])
    return jnp.stack([dx, dy, dz], axis=-1).reshape(-1, 3)

  def _nn3(self, pts):
    x = (pts[..., 0] - self.nmin[0]) / self.ndelta[0]
    y = (pts[..., 1] - self.nmin[1]) / self.ndelta[1]
    z = (pts[..., 2] - self.nmin[2]) / self.ndelta[2]

    # nearest neighbor
    x0 = jnp.round(x).astype(int)
    y0 = jnp.round(y).astype(int)
    z0 = jnp.round(z).astype(int)
    x0 = jnp.clip(x0, 0, self.ndim[0]-1)
    y0 = jnp.clip(y0, 0, self.ndim[1]-1)
    z0 = jnp.clip(z0, 0, self.ndim[2]-1)
    return self.data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y0+z0]
  
  def _linear3(self, pts):
    x = (pts[..., 0] - self.nmin[0]) / self.ndelta[0]
    y = (pts[..., 1] - self.nmin[1]) / self.ndelta[1]
    z = (pts[..., 2] - self.nmin[2]) / self.ndelta[2]

    # 8 neighbor vertex
    x0 = jnp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(int)
    y1 = y0 + 1
    z0 = jnp.floor(z).astype(int)
    z1 = z0 + 1

    xd = ((x - x0) / (x1 - x0))[..., None]
    yd = ((y - y0) / (y1 - y0))[..., None]
    zd = ((z - z0) / (z1 - z0))[..., None]

    # clamp to edge
    x0 = jnp.clip(x0, 0, self.ndim[0]-1)
    x1 = jnp.clip(x1, 0, self.ndim[0]-1)
    y0 = jnp.clip(y0, 0, self.ndim[1]-1)
    y1 = jnp.clip(y1, 0, self.ndim[1]-1)
    z0 = jnp.clip(z0, 0, self.ndim[2]-1)
    z1 = jnp.clip(z1, 0, self.ndim[2]-1)

    # 7 linear interpolation
    c00 = self.data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y0+z0]*(1-xd) + self.data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y0+z0]*xd
    c01 = self.data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y0+z1]*(1-xd) + self.data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y0+z1]*xd
    c10 = self.data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y1+z0]*(1-xd) + self.data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y1+z0]*xd
    c11 = self.data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y1+z1]*(1-xd) + self.data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y1+z1]*xd

    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    c = c0*(1-zd) + c1*zd
    return c

  def wrapper_grad_mlp(self, x, condition=None, annealed_alpha=1.0):
    if self.annealed:
      raw_out = self.so3_mlp(self.embed(x=x, alpha=annealed_alpha * self.max_deg_point))
    else:
      raw_out = self.so3_mlp(self.embed(x=x))
    
    # return raw_out
    # return jnp.sum(raw_out[..., None] * rl_utils.local_axis(self.action_basis, condition), axis=-2)
    # return jnp.sum(20.0 * jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2)
    # return jnp.linalg.norm(condition, axis=-1, keepdims=True) * jnp.sum(jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2)
    
    # # Spherical domain
    # raw_out = jnp.exp(raw_out)
    # return jnp.sum((raw_out - jnp.mean(raw_out, axis=-1, keepdims=True))[..., None] * self.action_basis[None, None], axis=-2)
    # # Normalized spherical domain
    # return jnp.linalg.norm(condition, axis=-1, keepdims=True) * math_utils.safe_l2_normalize(jnp.sum(jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2))
    # # Hemispherical domain
    # raw_out = jnp.exp(raw_out)
    # return jnp.sum((raw_out - jnp.mean(raw_out, axis=-1, keepdims=True))[..., None] * rl_utils.local_axis(self.action_basis, condition), axis=-2)

    if self.use_residual:
      if self.normalized:
        raise NotImplementedError()
      else:
        if self.use_direct_output:
          # NOTE(nan)
          theta = math_utils.safe_l2_norm(raw_out)
          e = raw_out / theta
          a = math_utils.safe_l2_norm(condition)
          v = condition / a
          # r = r / theta
          return a * (jnp.cos(theta) * v + jnp.sin(theta) * jnp.cross(e, v) + (1 - jnp.cos(theta)) * jnp.sum(e * v, axis=-1, keepdims=True) * e)
        else:
          theta, phi, r = jax.nn.tanh(raw_out[..., 0:1]) * jnp.pi, jax.nn.tanh(raw_out[..., 1:2]) * jnp.pi, jax.nn.softplus(raw_out[..., 2:3] - 1.0)
          return jnp.concatenate([jnp.sin(phi) * jnp.cos(theta), jnp.sin(phi) * jnp.sin(theta), jnp.cos(phi)], axis=-1) * r + condition
    else:
      if self.normalized:
        if self.use_direct_output:
          return jnp.linalg.norm(condition, axis=-1, keepdims=True) * math_utils.safe_l2_normalize(raw_out)
        else:
          return jnp.linalg.norm(condition, axis=-1, keepdims=True) * math_utils.safe_l2_normalize(jnp.sum(jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2))
      else:
        raise NotImplementedError()
    
  @nn.compact
  def __call__(self, x, annealed_alpha=1.0):
    """
    Args:
      x: jnp.ndarray, [batch, 3]
    
    Returns:
    """
    if self.interp_method == "linear3":
      ret = self._linear3(x)
    else:
      raise NotImplementedError()

    if self.annealed:
      raw_out = self.so3_mlp(self.embed(x=x[:, None], alpha=annealed_alpha * self.max_deg_point))
    else:
      raw_out = self.so3_mlp(self.embed(x=x[:, None]))

    # return ret[..., :1], ret[..., 1:], raw_out[:, 0]
    # return ret[..., :1], ret[..., 1:], jnp.sum(raw_out[..., None] * rl_utils.local_axis(self.action_basis, ret[:, None, 1:]), axis=-2)[:, 0]
    # return ret[..., :1], ret[..., 1:], jnp.sum(20.0 * jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2)[:, 0]
    # return ret[..., :1], ret[..., 1:], (jnp.linalg.norm(ret[:, None, 1:], axis=-1, keepdims=True) * jnp.sum(jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2))[:, 0]
    
    # # Spherical domain
    # raw_out = jnp.exp(raw_out)
    # return ret[..., :1], ret[..., 1:], (jnp.sum((raw_out - jnp.mean(raw_out, axis=-1, keepdims=True))[..., None] * self.action_basis[None, None], axis=-2))[:, 0]
    # # Normalized spherical domain
    # return ret[..., :1], ret[..., 1:], (jnp.linalg.norm(ret[:, None, 1:] + 1e-6, axis=-1, keepdims=True) * math_utils.safe_l2_normalize(jnp.sum(jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2)))[:, 0]
    # # Hemispherical domain
    # raw_out = jnp.exp(raw_out)
    # return ret[..., :1], ret[..., 1:], (jnp.sum((raw_out - jnp.mean(raw_out, axis=-1, keepdims=True))[..., None] * rl_utils.local_axis(self.action_basis, ret[:, None, 1:]), axis=-2))[:, 0]

    if self.use_residual:
      if self.normalized:
        raise NotImplementedError()
      else:
        if self.use_direct_output:
          # NOTE(nan)
          theta = math_utils.safe_l2_norm(raw_out[:, 0])
          e = raw_out[:, 0] / theta
          a = math_utils.safe_l2_norm(ret[..., 1:])
          v = ret[..., 1:] / a
          # r = r[:, 0] / theta
          return ret[..., :1], ret[..., 1:], a * (jnp.cos(theta) * v + jnp.sin(theta) * jnp.cross(e, v) + (1 - jnp.cos(theta)) * jnp.sum(e * v, axis=-1, keepdims=True) * e)
          # return ret[..., :1], ret[..., 1:], raw_out[:, 0] + ret[..., 1:]
        else:
          theta, phi, r = jax.nn.tanh(raw_out[..., 0:1]) * jnp.pi, jax.nn.tanh(raw_out[..., 1:2]) * jnp.pi, jax.nn.softplus(raw_out[..., 2:3] - 1.0)
          return ret[..., :1], ret[..., 1:], (jnp.concatenate([jnp.sin(phi) * jnp.cos(theta), jnp.sin(phi) * jnp.sin(theta), jnp.cos(phi)], axis=-1) * r)[:, 0] + ret[..., 1:]
    else:
      if self.normalized:
        if self.use_direct_output:
          return ret[..., :1], ret[..., 1:], (jnp.linalg.norm(ret[:, None, 1:] + 1e-6, axis=-1, keepdims=True) * math_utils.safe_l2_normalize(raw_out))[:, 0]
        else:
          return ret[..., :1], ret[..., 1:], (jnp.linalg.norm(ret[:, None, 1:] + 1e-6, axis=-1, keepdims=True) * math_utils.safe_l2_normalize(jnp.sum(jax.nn.softmax(raw_out, axis=-1)[..., None] * self.action_basis[None, None], axis=-2)))[:, 0]
      else:
        raise NotImplementedError()

# NOTE(prefilter)
def conv3d_normal(grid, ndim, ws, s):
  """Convolution by 3D Gaussian kernel.
  Args:
    - grid: np.ndarray, refractive index, [N_x * N_y * N_z, 1]
    - ndim: list, the resolution of voxel grid
    - ws: int, kernel size
    - s: float, std of Gaussian
  
  Returns:
    - out: jnp.ndarray, blur, [N_x * N_y * N_z, 1]
  """
  hws = ws//2
  data = jnp.asarray(grid)
  data = data.reshape(ndim[0], ndim[1], ndim[2], 1)
  data = jnp.pad(data[..., 0], ((hws, hws), (hws, hws), (hws, hws)), 'edge')
  data = data[..., None]
  data = data[None]

  a = jnp.linspace(-hws, hws, ws)
  xx, yy, zz = jnp.meshgrid(a, a, a)
  kernel = jnp.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * s**2))
  kernel = kernel[..., None, None] / jnp.sum(kernel)

  dn = jax.lax.conv_dimension_numbers(
    data.shape, kernel.shape,
    ('NHWDC', 'HWDIO', 'NHWDC'))

  out = jax.lax.conv_general_dilated(
    data,    # lhs = image tensor
    kernel,  # rhs = conv kernel tensor
    (1,1,1), # window strides
    'VALID', # padding mode
    (1,1,1), # lhs/image dilation
    (1,1,1), # rhs/kernel dilation
    dn)      # dimension_numbers
  
  return out[0].reshape(-1, 1)