# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different model implementation plus a general port for all the models."""
from typing import Any, Callable

from flax import linen as nn
from jax import random
import jax
import jax.numpy as jnp
import gin

from rnerf import model_utils
from rnerf import math_utils
from rnerf import utils
from rnerf import eikonal_utils
from rnerf import sh
from rnerf import mip


def get_model(key, example_batch, args):
  """A helper function that wraps around a 'model zoo'."""
  model_dict = {
      "nerf": construct_nerf,
  }
  return model_dict[args.model](key, example_batch, args)


@gin.configurable()
class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""
  ndim: list
  nmin: list
  nmax: list
  grid: jnp.ndarray
  stage: str
  # NOTE(regularization)
  use_fine_sparsity: bool
  use_online_sparsity: bool

  num_coarse_samples: int  # The number of samples for the coarse nerf.
  num_fine_samples: int  # The number of samples for the fine nerf.
  use_viewdirs: bool  # If True, use viewdirs as an input.
  # NOTE(sh)
  sh_deg: int  # If != -1, use spherical harmonics output up to given degree
  near: float  # The distance to the near plane
  far: float  # The distance to the far plane
  noise_std: float  # The std dev of noise added to raw sigma.
  net_depth: int  # The depth of the first part of MLP.
  net_width: int  # The width of the first part of MLP.
  net_depth_condition: int  # The depth of the second part of MLP.
  net_width_condition: int  # The width of the second part of MLP.
  net_activation: Callable[..., Any]  # MLP activation
  skip_layer: int  # How often to add skip connections.
  num_rgb_channels: int  # The number of RGB channels.
  num_sigma_channels: int  # The number of density channels.
  white_bkgd: bool  # If True, use a white background.
  min_deg_point: int  # The minimum degree of positional encoding for positions.
  max_deg_point: int  # The maximum degree of positional encoding for positions.
  deg_view: int  # The degree of positional encoding for viewdirs.
  lindisp: bool  # If True, sample linearly in disparity rather than in depth.
  rgb_activation: Callable[..., Any]  # Output RGB activation.
  sigma_activation: Callable[..., Any]  # Output sigma activation.
  legacy_posenc_order: bool  # Keep the same ordering as the original tf code.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  sigma_bias: float = -1.  # The shift added to raw sigma pre-activation.

  num_path_samples: int = 8
  sh_direnc_deg: int = -1

  # NOTE(voxelize)
  use_mask_bbox: bool = False
  # NOTE(ef)
  bd_cut_dist: float = None
  cfg_name: str = None
  use_random_choice: bool = True

  def setup(self):
    # Construct the "coarse" MLP.
    self.coarse_mlp = model_utils.NerfMLP(
      net_depth=self.net_depth,
      net_width=self.net_width,
      net_depth_condition=self.net_depth_condition,
      net_width_condition=self.net_width_condition,
      net_activation=self.net_activation,
      skip_layer=self.skip_layer,
      num_rgb_channels=self.num_rgb_channels,
      num_sigma_channels=self.num_sigma_channels)
    
    # Construct the "fine" MLP.
    if self.num_fine_samples > 0:
      self.fine_mlp = model_utils.NerfMLP(
        net_depth=self.net_depth,
        net_width=self.net_width,
        net_depth_condition=self.net_depth_condition,
        net_width_condition=self.net_width_condition,
        net_activation=self.net_activation,
        skip_layer=self.skip_layer,
        num_rgb_channels=self.num_rgb_channels,
        num_sigma_channels=self.num_sigma_channels)

    # Construct the "bkgd-term" MLP.
    self.bkgd_mlp = model_utils.MLP(
      net_width=128, net_depth=4, skip_layer=2,
      num_out_channels=self.num_rgb_channels)

    # NOTE(eikonal)
    num_samples = self.num_coarse_samples * self.num_path_samples
    step_size = (self.far - self.near) / (num_samples - 1)
    self.path_sampler = eikonal_utils.PathSampler(
      near=self.near,
      far=self.far,
      stage=self.stage,
      num_samples=num_samples,
      step_size=step_size,
      ndim=self.ndim, nmin=self.nmin, nmax=self.nmax,
      grid=self.grid,
    )

    self.coarse_step_size = (self.far - self.near) / self.num_coarse_samples
    self.fine_step_size = (self.far - self.near) / (self.num_coarse_samples + self.num_fine_samples)

    self.beta0 = 1. / 9.
    self.beta1 = 1.

  def wrapper_compute_normal_loss_and_smooth(self, ray_pos, idx_grad, annealed_alpha=1.0):
    return self.path_sampler.compute_normal_loss_and_smooth(ray_pos, idx_grad, annealed_alpha=annealed_alpha)

  def compute_sparsity_loss(self, ray_pos, coarse_alpha_target, fine_alpha_target):
    ray_dir = jnp.zeros_like(ray_pos)
    samples_enc = model_utils.pos_enc(ray_pos, self.min_deg_point, self.max_deg_point, legacy_posenc_order=self.legacy_posenc_order)
    #NOTE(sh)
    if self.sh_direnc_deg > 0:
      viewdirs_enc = sh.dir_enc(ray_dir, self.sh_direnc_deg)
    else:
      viewdirs_enc = model_utils.pos_enc(
          ray_dir,
          0,
          self.deg_view,
          self.legacy_posenc_order,
        )
    
    # Feed forward
    if self.use_viewdirs:
      raw_rgb, raw_sigma = self.coarse_mlp(samples_enc, viewdirs_enc)
    else:
      raw_rgb, raw_sigma = self.coarse_mlp(samples_enc)
    del raw_rgb
    sigma = self.sigma_activation(raw_sigma + self.sigma_bias)
    alpha = 1 - jnp.exp(-self.coarse_step_size * sigma)
    #rgb_alpha = rgb * alpha
    loss_sp = (jnp.abs(alpha - coarse_alpha_target)).mean()
    next_coarse_alpha_target = alpha.mean()

    if self.num_fine_samples > 0 and self.use_fine_sparsity:
      if self.use_viewdirs:
        raw_rgb, raw_sigma = self.fine_mlp(samples_enc, viewdirs_enc)
      else:
        raw_rgb, raw_sigma = self.fine_mlp(samples_enc)
      del raw_rgb
      sigma = self.sigma_activation(raw_sigma + self.sigma_bias)
      alpha = 1 - jnp.exp(-self.fine_step_size * sigma)
      #rgb_alpha = rgb * alpha
      loss_sp += (jnp.abs(alpha - fine_alpha_target)).mean()
      next_fine_alpha_target = alpha.mean()
    return loss_sp, next_coarse_alpha_target, next_fine_alpha_target

  def forward_envmap(self, viewdirs):
    viewdirs_enc = model_utils.pos_enc(
        viewdirs,
        0,
        self.deg_view,
        self.legacy_posenc_order,
      )
    raw_bkgd = self.bkgd_mlp(viewdirs_enc[:, None])[:, 0]
    bkgd = self.rgb_activation(raw_bkgd)
    bkgd = bkgd * (1 + 2 * self.rgb_padding) - self.rgb_padding
    return bkgd

  def sample_points(self, pts, viewdirs):
    samples_enc = model_utils.pos_enc(pts, self.min_deg_point, self.max_deg_point, legacy_posenc_order=self.legacy_posenc_order)
    #NOTE(sh)
    if self.sh_direnc_deg > 0:
      viewdirs_enc = sh.dir_enc(viewdirs, self.sh_direnc_deg)
    else:
      viewdirs_enc = model_utils.pos_enc(
          viewdirs,
          0,
          self.deg_view,
          self.legacy_posenc_order,
        )
    
    # Feed forward
    mlp = self.fine_mlp if self.num_fine_samples > 0 else self.coarse_mlp
    step_size = self.fine_step_size if self.num_fine_samples > 0 else self.coarse_step_size
    if self.use_viewdirs:
      raw_rgb, raw_sigma = mlp(samples_enc, viewdirs_enc)
    else:
      raw_rgb, raw_sigma = mlp(samples_enc)
    rgb = self.rgb_activation(raw_rgb)
    rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
    sigma = self.sigma_activation(raw_sigma + self.sigma_bias)
    alpha = (1 - jnp.exp(-step_size * sigma))
    return rgb, alpha
  
  @nn.compact
  def __call__(self, rng_0, rng_1, rays, randomized, annealed_alpha=1.0):
    """Nerf Model.

    Args:
      rng_0: jnp.ndarray, random number generator for coarse model sampling.
      rng_1: jnp.ndarray, random number generator for fine model sampling.
      rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
      randomized: bool, use randomized stratified sampling.

    Returns:
      ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
    """
    # Stratified sampling along rays by Eikonal equation
    key, rng_0 = random.split(rng_0)
    ray_pos, ray_dir, ray_dist, idx_data, idx_grad = self.path_sampler(
      rays.origins,
      rays.viewdirs,
      annealed_alpha,
    )
    # Loss backward through these
    jitter = jnp.arange(0, self.num_coarse_samples * self.num_path_samples, self.num_path_samples)
    if self.use_random_choice:
      jitter += random.randint(key, [self.num_coarse_samples,], minval=0, maxval=self.num_path_samples)
    ray_pos_c = ray_pos[:, jitter]
    ray_dir_c = ray_dir[:, jitter]
    ray_dist_c = ray_dist[:, jitter]
    idx_data_c = idx_data[:, jitter]
    idx_grad_c = idx_grad[:, jitter]

    # t_vals = jnp.concatenate([
    #     ray_dist_c,
    #     ray_dist_c[..., -1:] + jnp.broadcast_to([1e-3], ray_dist_c[..., -1:].shape)
    # ], -1)
    # samples = mip.cast_rays(t_vals, ray_pos_c, ray_dir_c, rays.radii, "cone", self.near)
    # samples_enc = mip.integrated_pos_enc(samples, self.min_deg_point, self.max_deg_point)

    # Point attribute predictions
    samples_enc = model_utils.pos_enc(ray_pos_c, self.min_deg_point, self.max_deg_point, legacy_posenc_order=self.legacy_posenc_order)
    # samples_enc = model_utils.annealed_pos_enc(ray_pos_c, self.min_deg_point, self.max_deg_point, alpha=annealed_alpha * self.max_deg_point)

    # NOTE(voxelize)
    if self.use_mask_bbox:
      # small mask bbox
      nmin = self.nmin
      nmax = self.nmax
      # # large mask bbox
      # nmin = [0.0 - 0.05, 0.0 - 0.05, 0.0 - 0.05]
      # nmax = [0.81 + 0.05, 0.81 + 0.05, 0.81 + 0.05]
      mask_bbox = (
        (ray_pos_c[..., 0] >= nmin[0]) * (ray_pos_c[..., 0] <= nmax[0]) *
        (ray_pos_c[..., 1] >= nmin[1]) * (ray_pos_c[..., 1] <= nmax[1]) *
        (ray_pos_c[..., 2] >= nmin[2]) * (ray_pos_c[..., 2] <= nmax[2])
      )

      # # ball
      # nmin = [-1, 0.03597, -1]
      # nmax = [1, 2.03597, 1]
      # mask_bbox = 1. - (
      #   (ray_pos_c[..., 0] >= nmin[0]) * (ray_pos_c[..., 0] <= nmax[0]) *
      #   (ray_pos_c[..., 1] >= nmin[1]) * (ray_pos_c[..., 1] <= nmax[1]) *
      #   (ray_pos_c[..., 2] >= nmin[2]) * (ray_pos_c[..., 2] <= nmax[2])
      # )
    else:
      mask_bbox = None

    #NOTE(sh)
    if self.sh_direnc_deg > 0:
      viewdirs_enc = sh.dir_enc(ray_dir_c, self.sh_direnc_deg)
    else:
      viewdirs_enc = model_utils.pos_enc(
          ray_dir_c, # * jnp.pi
          0,
          self.deg_view,
          self.legacy_posenc_order,
        )
      # viewdirs_enc = model_utils.annealed_pos_enc(
      #     ray_dir_c, # * jnp.pi
      #     0,
      #     self.deg_view,
      #     annealed_alpha * self.deg_view,
      #   )

    # Feed forward
    raw_bkgd = self.bkgd_mlp(viewdirs_enc[:, -1:])[:, 0]
    # raw_bkgd = self.bkgd_mlp(sh.dir_enc(ray_dir_c[:, -1:], 5))[:, 0]
    if self.use_viewdirs:
      raw_rgb, raw_sigma = self.coarse_mlp(samples_enc, viewdirs_enc)
    else:
      raw_rgb, raw_sigma = self.coarse_mlp(samples_enc)
    
    # Add noises to regularize the density predictions if needed
    key, rng_0 = random.split(rng_0)
    raw_sigma = model_utils.add_gaussian_noise(
        key,
        raw_sigma,
        self.noise_std,
        randomized,
    )

    # NOTE(sh)
    if self.sh_deg >= 0:
      # (256, 64, 48) (256, [N], 3)
      raw_rgb = sh.eval_sh(self.sh_deg, raw_rgb.reshape(
        *raw_rgb.shape[:-1],
        -1,
        (self.sh_deg + 1) ** 2), ray_dir_c)
      
      raw_bkgd = raw_bkgd[:, None]
      raw_bkgd = sh.eval_sh(self.sh_deg, raw_bkgd.reshape(
        *raw_bkgd.shape[:-1],
        -1,
        (self.sh_deg + 1) ** 2), ray_dir_c[:, -1:])[:, 0]
    
    # NOTE: widened sigmoid, shifted softplus
    rgb = self.rgb_activation(raw_rgb)
    rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
    bkgd = self.rgb_activation(raw_bkgd)
    bkgd = bkgd * (1 + 2 * self.rgb_padding) - self.rgb_padding
    sigma = self.sigma_activation(raw_sigma + self.sigma_bias)

    # Volumetric rendering
    comp_rgb, disp, acc, weights, alpha, trans, trans_rgb_bkgd = model_utils.volumetric_rendering(
        rgb,
        sigma,
        ray_dist_c,
        ray_dir_c,
        white_bkgd=self.white_bkgd,
        rgb_bkgd=bkgd,
        mask_bbox=mask_bbox,
    )

    if self.use_online_sparsity:
      mask = jnp.linalg.norm(idx_grad_c, axis=-1) > 1e-6  # boundary
      # mask = idx_data_c[..., 0] > 1.0  # inside
      loss_sp = (mask * math_utils.safe_log(alpha)).sum() / (jnp.sum(mask) + 1)
      # loss_sp = (self.beta0 * math_utils.safe_log(alpha) + self.beta1 * math_utils.safe_log(1 - alpha)).mean()
    else:
      loss_sp = 0.

    ret = [
        (comp_rgb, disp, acc, trans, trans_rgb_bkgd),
    ]
    # # NOTE(debug): visualize coarse and fine samples
    # ret = [
    #     (comp_rgb, disp, acc, ray_pos_c, ray_dir_c, idx_grad_c, trans, trans_rgb_bkgd),
    # ]

    # Hierarchical sampling based on coarse predictions
    if self.num_fine_samples > 0:
      key, rng_1 = random.split(rng_1)
      # NOTE(eikonal)
      ray_dist_c_mid = .5 * (ray_dist_c[..., 1:] + ray_dist_c[..., :-1])
      ray_dist_c, ray_pos_c, ray_dir_c, idx_grad_c = model_utils.sample_pdf(
          key,
          ray_dist_c_mid,
          weights[..., 1:-1],
          ray_pos,
          ray_dir,
          ray_dist,
          idx_grad,
          self.num_fine_samples,
          randomized,
          jitter,
          self.near,
      )

      # t_vals = jnp.concatenate([
      #     ray_dist_c,
      #     ray_dist_c[..., -1:] + jnp.broadcast_to([1e-3], ray_dist_c[..., -1:].shape)
      # ], -1)
      # samples = mip.cast_rays(t_vals, ray_pos_c, ray_dir_c, rays.radii, "cone", self.near)
      # samples_enc = mip.integrated_pos_enc(samples, self.min_deg_point, self.max_deg_point)

      # Point attribute predictions
      samples_enc = model_utils.pos_enc(ray_pos_c, self.min_deg_point, self.max_deg_point, legacy_posenc_order=self.legacy_posenc_order)
      # samples_enc = model_utils.annealed_pos_enc(ray_pos_c, self.min_deg_point, self.max_deg_point, alpha=annealed_alpha * self.max_deg_point)

      # NOTE(voxelize)
      if self.use_mask_bbox:
        # small mask bbox
        nmin = self.nmin
        nmax = self.nmax
        # # large mask bbox
        # nmin = [0.0 - 0.05, 0.0 - 0.05, 0.0 - 0.05]
        # nmax = [0.81 + 0.05, 0.81 + 0.05, 0.81 + 0.05]
        mask_bbox = (
          (ray_pos_c[..., 0] >= nmin[0]) * (ray_pos_c[..., 0] <= nmax[0]) *
          (ray_pos_c[..., 1] >= nmin[1]) * (ray_pos_c[..., 1] <= nmax[1]) *
          (ray_pos_c[..., 2] >= nmin[2]) * (ray_pos_c[..., 2] <= nmax[2])
        )

        # # ball
        # nmin = [-1, 0.03597, -1]
        # nmax = [1, 2.03597, 1]
        # mask_bbox = 1. - (
        #   (ray_pos_c[..., 0] >= nmin[0]) * (ray_pos_c[..., 0] <= nmax[0]) *
        #   (ray_pos_c[..., 1] >= nmin[1]) * (ray_pos_c[..., 1] <= nmax[1]) *
        #   (ray_pos_c[..., 2] >= nmin[2]) * (ray_pos_c[..., 2] <= nmax[2])
        # )
      else:
        mask_bbox = None

      # NOTE(sh)
      if self.sh_direnc_deg > 0:
        viewdirs_enc = sh.dir_enc(ray_dir_c, self.sh_direnc_deg)
      else:
        viewdirs_enc = model_utils.pos_enc(
          ray_dir_c, # * jnp.pi
          0,
          self.deg_view,
          self.legacy_posenc_order,
        )
        # viewdirs_enc = model_utils.annealed_pos_enc(
        #   ray_dir_c, # * jnp.pi
        #   0,
        #   self.deg_view,
        #   annealed_alpha * self.deg_view,
        # )

      # Feed forward
      if self.use_viewdirs:
        raw_rgb, raw_sigma = self.fine_mlp(samples_enc, viewdirs_enc)
      else:
        raw_rgb, raw_sigma = self.fine_mlp(samples_enc)
      
      # Add noises to regularize the density predictions if needed
      key, rng_1 = random.split(rng_1)
      raw_sigma = model_utils.add_gaussian_noise(
          key,
          raw_sigma,
          self.noise_std,
          randomized,
      )

      # NOTE(sh)
      if self.sh_deg >= 0:
        # (256, 64, 48) (256, [N], 3)
        raw_rgb = sh.eval_sh(self.sh_deg, raw_rgb.reshape(
          *raw_rgb.shape[:-1],
          -1,
          (self.sh_deg + 1) ** 2), ray_dir_c)
      
      # NOTE: widened sigmoid, shifted softplus
      rgb = self.rgb_activation(raw_rgb)
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
      sigma = self.sigma_activation(raw_sigma + self.sigma_bias)

      # Volumetric rendering
      comp_rgb, disp, acc, unused_weights, alpha, trans, trans_rgb_bkgd = model_utils.volumetric_rendering(
          rgb,
          sigma,
          ray_dist_c,  # NOTE(eikonal)
          ray_dir_c,  # NOTE(eikonal)
          white_bkgd=self.white_bkgd,
          rgb_bkgd=bkgd,
          mask_bbox=mask_bbox,
      )

      # NOTE(ef)
      if self.bd_cut_dist is not None:
        assert not self.use_mask_bbox, "'use_mask_bbox' is true"
        # v1
        # mask_bbox = ray_dist_c < self.bd_cut_dist

        # v2
        if "pen" in self.cfg_name:
          nmin = [self.nmin[0], self.nmin[1], self.nmin[2]]
          nmax = [self.nmax[0], self.nmax[1], self.nmax[2]]
          nmax[1] -= 0.6
        elif "ball" in self.cfg_name:
          nmin = [-1, 0.03597, -1]
          nmax = [1, 2.03597, 1]
        elif "glass" in self.cfg_name:
          nmin = [self.nmin[0], self.nmin[1], self.nmin[2]]
          nmax = [self.nmax[0], self.nmax[1], self.nmax[2]]
          nmax[1] -= 0.7
        else:
          raise NotImplementedError()
        mask_bbox = (
          (ray_pos_c[..., 0] >= nmin[0]) * (ray_pos_c[..., 0] <= nmax[0]) *
          (ray_pos_c[..., 1] >= nmin[1]) * (ray_pos_c[..., 1] <= nmax[1]) *
          (ray_pos_c[..., 2] >= nmin[2]) * (ray_pos_c[..., 2] <= nmax[2])
        )
        mask_bbox = (jnp.cumsum(mask_bbox[:, ::-1], axis=-1) > 0.0)[:, ::-1]

        _, _, _, _, _, trans, _ = model_utils.volumetric_rendering(
            rgb,
            sigma,
            ray_dist_c,
            ray_dir_c,
            white_bkgd=self.white_bkgd,
            rgb_bkgd=None,
            mask_bbox=mask_bbox,
        )

        trans_rgb_bkgd, _, _, _, _, _, _ = model_utils.volumetric_rendering(
            rgb,
            sigma,
            ray_dist_c,
            ray_dir_c,
            white_bkgd=self.white_bkgd,
            rgb_bkgd=bkgd,
            mask_bbox=(1.0 - mask_bbox),
        )
        trans_rgb_bkgd = trans * trans_rgb_bkgd

      if self.use_online_sparsity and self.use_fine_sparsity:
        mask = jnp.linalg.norm(idx_grad_c, axis=-1) > 1e-6  # boundary
        # mask = idx_data_c[..., 0] > 1.0  # inside
        loss_sp += (mask * math_utils.safe_log(alpha)).sum() / (jnp.sum(mask) + 1)
        # loss_sp += (self.beta0 * math_utils.safe_log(alpha) + self.beta1 * math_utils.safe_log(1 - alpha)).mean()
      
      ret.append((comp_rgb, disp, acc, trans, trans_rgb_bkgd))
      # # NOTE(debug): visualize coarse and fine samples
      # ret.append((comp_rgb, disp, acc, ray_pos_c, ray_dir_c, idx_grad_c, trans, trans_rgb_bkgd))
    return ret, loss_sp


def construct_nerf(key, example_batch, args, ndim, nmin, nmax, grid):
  """Construct a Neural Radiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.
    args: FLAGS class. Hyperparameters of nerf.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  net_activation = getattr(nn, str(args.net_activation))
  rgb_activation = getattr(nn, str(args.rgb_activation))
  sigma_activation = getattr(nn, str(args.sigma_activation))

  # Assert that rgb_activation always produces outputs in [0, 1], and
  # sigma_activation always produce non-negative outputs.
  x = jnp.exp(jnp.linspace(-90, 90, 1024))
  x = jnp.concatenate([-x[::-1], x], 0)

  rgb = rgb_activation(x)
  if jnp.any(rgb < 0) or jnp.any(rgb > 1):
    raise NotImplementedError(
        "Choice of rgb_activation `{}` produces colors outside of [0, 1]"
        .format(args.rgb_activation))

  sigma = sigma_activation(x)
  if jnp.any(sigma < 0):
    raise NotImplementedError(
        "Choice of sigma_activation `{}` produces negative densities".format(
            args.sigma_activation))
  # NOTE(sh)
  if args.sh_deg >= 0:
      assert not args.use_viewdirs, (
              "You can only use up to one of: SH or use_viewdirs.")
      args.num_rgb_channels *= (args.sh_deg + 1) ** 2

  model = NerfModel(
      min_deg_point=args.min_deg_point,
      max_deg_point=args.max_deg_point,
      deg_view=args.deg_view,
      num_coarse_samples=args.num_coarse_samples,
      num_fine_samples=args.num_fine_samples,
      use_viewdirs=args.use_viewdirs,
      sh_deg=args.sh_deg,
      near=args.near,
      far=args.far,
      noise_std=args.noise_std,
      white_bkgd=args.white_bkgd,
      net_depth=args.net_depth,
      net_width=args.net_width,
      net_depth_condition=args.net_depth_condition,
      net_width_condition=args.net_width_condition,
      skip_layer=args.skip_layer,
      num_rgb_channels=args.num_rgb_channels,
      num_sigma_channels=args.num_sigma_channels,
      lindisp=args.lindisp,
      net_activation=net_activation,
      rgb_activation=rgb_activation,
      sigma_activation=sigma_activation,
      legacy_posenc_order=args.legacy_posenc_order,
      ndim=ndim, nmin=nmin, nmax=nmax,
      grid=grid,
      stage=args.stage,
      num_path_samples=args.num_path_samples,
      use_fine_sparsity=args.use_fine_sparsity,
      use_online_sparsity=args.use_online_sparsity,
      sh_direnc_deg=args.sh_direnc_deg,
      cfg_name=args.config)
  rays = example_batch["rays"]
  key1, key2, key3 = random.split(key, num=3)

  init_variables = model.init(
      key1,
      rng_0=key2,
      rng_1=key3,
      rays=utils.namedtuple_map(lambda x: x[0], rays),
      randomized=args.randomized)

  return model, init_variables
