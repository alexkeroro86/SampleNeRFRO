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
"""Helper functions/classes for model definition."""

import functools
from typing import Any, Callable
import numpy as np

from flax import linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp


class NerfMLP(nn.Module):
  """A Nerf MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  net_activation: Callable[..., Any] = nn.relu  # The activation function.
  skip_layer: int = 4  # The layer to add skip layers to.
  num_rgb_channels: int = 3  # The number of RGB channels.
  num_sigma_channels: int = 1  # The number of sigma channels.

  @nn.compact
  def __call__(self, x, condition=None):
    """Evaluate the MLP.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.
      condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.

    Returns:
      raw_rgb: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_rgb_channels].
      raw_sigma: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_sigma_channels].
    """
    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
    inputs = x
    for i in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    raw_sigma = dense_layer(self.num_sigma_channels)(x).reshape(
        [-1, num_samples, self.num_sigma_channels])

    if condition is not None:
      # Output of the first part of MLP.
      bottleneck = dense_layer(self.net_width)(x)
      # Broadcast condition from [batch, feature] to
      # [batch, num_samples, feature] since all the samples along the same ray
      # have the same viewdir.
      
      # Collapse the [batch, num_samples, feature] tensor to
      # [batch * num_samples, feature] so that it can be fed into nn.Dense.
      condition = condition.reshape([-1, condition.shape[-1]])
      x = jnp.concatenate([bottleneck, condition], axis=-1)
      # Here use 1 extra layer to align with the original nerf model.
      for i in range(self.net_depth_condition):
        x = dense_layer(self.net_width_condition)(x)
        x = self.net_activation(x)
    raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape(
        [-1, num_samples, self.num_rgb_channels])
    return raw_rgb, raw_sigma


class MLP(nn.Module):
  """A simple MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  net_activation: Callable[..., Any] = nn.relu  # The activation function.
  skip_layer: int = 4  # The layer to add skip layers to.
  num_out_channels: int = 1  # The number of sigma channels.
  output_init: Callable = jax.nn.initializers.xavier_uniform()

  @nn.compact
  def __call__(self, x, condition=None):
    """Evaluate the MLP.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.
      condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.

    Returns:
      raw_out: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_out_channels].
    """
    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.xavier_uniform())
    inputs = x
    for i in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    if condition is not None:
      condition = condition.reshape([-1, condition.shape[-1]])
      x = jnp.concatenate([x, condition], axis=-1)
      for i in range(self.net_depth_condition):
        x = dense_layer(self.net_width_condition)(x)
        x = self.net_activation(x)
    raw_out = nn.Dense(self.num_out_channels, kernel_init=self.output_init)(x).reshape(
        [-1, num_samples, self.num_out_channels])

    return raw_out


def cast_rays(z_vals, origins, directions):
  return origins[..., None, :] + z_vals[..., None] * directions[..., None, :]


# def sample_along_rays(key, origins, directions, num_samples, near, far,
#                       randomized, lindisp):
#   """Stratified sampling along the rays.

#   Args:
#     key: jnp.ndarray, random generator key.
#     origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
#     directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
#     num_samples: int.
#     near: float, near clip.
#     far: float, far clip.
#     randomized: bool, use randomized stratified sampling.
#     lindisp: bool, sampling linearly in disparity rather than depth.

#   Returns:
#     z_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
#     points: jnp.ndarray, [batch_size, num_samples, 3], sampled points.
#   """
#   batch_size = origins.shape[0]

#   t_vals = jnp.linspace(0., 1., num_samples)
#   if lindisp:
#     t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
#   else:
#     t_vals = near * (1. - t_vals) + far * t_vals

#   if randomized:
#     mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
#     upper = jnp.concatenate([mids, t_vals[..., -1:]], -1)
#     lower = jnp.concatenate([t_vals[..., :1], mids], -1)
#     t_rand = random.uniform(key, [batch_size, num_samples])
#     t_vals = lower + (upper - lower) * t_rand
#   else:
#     # Broadcast t_vals to make the returned shape consistent.
#     t_vals = jnp.broadcast_to(t_vals[None, ...], [batch_size, num_samples])

#   coords = cast_rays(t_vals, origins, directions)
#   return t_vals, coords


def pos_enc(x, min_deg, max_deg, legacy_posenc_order=False, amp=1.0):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.
    legacy_posenc_order: bool, keep the same ordering as the original tf code.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  if legacy_posenc_order:
    xb = x[..., None, :] * scales[:, None]
    four_feat = jnp.reshape(
        jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
        list(x.shape[:-1]) + [-1])
  else:
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [amp * four_feat], axis=-1)


# Reference: https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/modules.py#L231
def cosine_easing_window(min_freq_log2, max_freq_log2, num_bands, alpha):
  """Eases in each frequency one by one with a cosine.
  This is equivalent to taking a Tukey window and sliding it to the right
  along the frequency spectrum.
  Args:
    min_freq_log2: the lower frequency band.
    max_freq_log2: the upper frequency band.
    num_bands: the number of frequencies.
    alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.
  Returns:
    A 1-d numpy array with num_sample elements containing the window.
  """
  if max_freq_log2 is None:
    max_freq_log2 = num_bands - 1.0
  bands = jnp.linspace(min_freq_log2, max_freq_log2, num_bands)
  x = jnp.clip(alpha - bands, 0.0, 1.0)
  return 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))

def annealed_pos_enc(x, min_deg, max_deg, alpha, amp=1.0):
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])

  xb = x[..., None, :] * scales[:, None]  # [batch, sample, num_deg, channel]
  window = cosine_easing_window(min_deg, max_deg - 1, len(scales), alpha)[:, None]  # [num_deg, 1]
  four_feat = jnp.reshape(jnp.concatenate([jnp.sin(xb) * window, jnp.sin(xb + 0.5 * jnp.pi) * window], axis=-1),
                          list(x.shape[:-1]) + [-1])  # [batch, sample, channel]
  return amp * four_feat

def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd, rgb_bkgd, mask_bbox=None):
  """Volumetric Rendering Function.

  Args:
    rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
    density: jnp.ndarray(float32), sigma, [batch_size, num_samples, 1].
    t_vals: jnp.ndarray(float32), [batch_size, num_samples].
    dirs: jnp.ndarray(float32), [batch_size, 3].
    white_bkgd: bool.

  Returns:
    comp_rgb: jnp.ndarray(float32), [batch_size, 3].
    disp: jnp.ndarray(float32), [batch_size].
    acc: jnp.ndarray(float32), [batch_size].
    weights: jnp.ndarray(float32), [batch_size, num_samples]
  """
  eps = 1e-10
  # NOTE(bkgd): To account for boundary term, narrow down the last delta
  t_dists = jnp.concatenate([
      t_vals[..., 1:] - t_vals[..., :-1],
      jnp.broadcast_to([1e-3], t_vals[..., :1].shape)
  ], -1)
  # NOTE(eikonal)
  delta = t_dists * jnp.linalg.norm(dirs, axis=-1)
  # Note that we're quietly turning density from [..., 0] to [...].
  density_delta = density[..., 0] * delta

  # NOTE(voxelize)
  if mask_bbox is not None:
    density_delta *= mask_bbox

  # Check beta/sparsity
  # if mask is not None:
  #   mask = mask * ((1 - jnp.exp(-density_delta)) > 0.5)
  # mask = (1 - jnp.exp(-density_delta)) > 0.1
  # density_delta = density_delta * mask

  # Exponential transmittance model
  alpha = 1 - jnp.exp(-density_delta)
  trans = jnp.exp(-jnp.concatenate([
      jnp.zeros_like(density_delta[..., :1]),
      jnp.cumsum(density_delta[..., :], axis=-1)
  ], axis=-1))
  # # Linear transmittance model
  # alpha = 1 - nn.relu(1 - density_delta)
  # trans = jnp.concatenate([
  #     jnp.ones_like(alpha[..., :1], alpha.dtype),
  #     jnp.cumprod(nn.relu(1 - density_delta[..., :]) + eps, axis=-1)
  # ], axis=-1)
  weights = alpha * trans[..., :-1]

  if rgb_bkgd is not None:
    comp_rgb = (weights[..., None] * rgb).sum(axis=-2) + trans[..., -1:] * rgb_bkgd
  else:
    comp_rgb = (weights[..., None] * rgb).sum(axis=-2)
    rgb_bkgd = jnp.ones(list(trans[..., -1:].shape[:-1]) + [3])
  acc = weights.sum(axis=-1)
  distance = (weights * t_vals).sum(axis=-1) / acc
  distance = jnp.clip(jnp.nan_to_num(distance, jnp.inf), t_vals[:, 0], t_vals[:, -1])

  if white_bkgd:
    comp_rgb = comp_rgb + (1. - acc[..., None])
  return comp_rgb, distance, acc, weights, alpha, trans[..., -1:], trans[..., -1:] * lax.stop_gradient(rgb_bkgd)


def sorted_piecewise_constant_pdf(key, bins, weights, num_samples, randomized):
  """Piecewise-Constant PDF sampling from sorted bins.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, num_bins].
    num_samples: int, the number of samples.
    randomized: bool, use randomized samples.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
  # avoids NaNs when the input is zeros or small, but has no effect otherwise.
  eps = 1e-5
  weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
  padding = jnp.maximum(0, eps - weight_sum)
  weights += padding / weights.shape[-1]
  weight_sum += padding

  # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
  # starts with exactly 0 and ends with exactly 1.
  pdf = weights / weight_sum
  cdf = jnp.minimum(1, jnp.cumsum(pdf[..., :-1], axis=-1))
  cdf = jnp.concatenate([
      jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf,
      jnp.ones(list(cdf.shape[:-1]) + [1])
  ], axis=-1)

  # Draw uniform samples.
  if randomized:
    # mipnerf
    s = 1 / num_samples
    u = jnp.arange(num_samples) * s
    u += jax.random.uniform(
        key,
        list(cdf.shape[:-1]) + [num_samples],
        maxval=s - jnp.finfo('float32').eps)
    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u = jnp.minimum(u, 1. - jnp.finfo('float32').eps)
  else:
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    u = jnp.linspace(0., 1. - jnp.finfo('float32').eps, num_samples)
    u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

  # Identify the location in `cdf` that corresponds to a random sample.
  # The final `True` index in `mask` will be the start of the sampled interval.
  mask = u[..., None, :] >= cdf[..., :, None]

  def find_interval(x):
    # Grab the value where `mask` switches from True to False, and vice versa.
    # This approach takes advantage of the fact that `x` is sorted.
    x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
    return x0, x1

  bins_g0, bins_g1 = find_interval(bins)
  cdf_g0, cdf_g1 = find_interval(cdf)

  t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
  samples = bins_g0 + t * (bins_g1 - bins_g0)
  return samples


def sample_pdf(key, bins, weights, origins, directions, z_vals, idx_grads, num_samples,
               randomized, jitter, near, stop_grad=True, resample_padding=0.01):
  """Hierarchical sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, num_bins].
    origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
    directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
    z_vals: jnp.ndarray(float32), [batch_size, num_coarse_samples].
    num_samples: int, the number of samples.
    randomized: bool, use randomized samples.
    stop_grad: bool, whether or not to backprop through sampling.
    resample_padding: float, added to the weights before normalizing.

  Returns:
    z_vals: jnp.ndarray(float32),
      [batch_size, num_coarse_samples + num_fine_samples].
    points: jnp.ndarray(float32),
      [batch_size, num_coarse_samples + num_fine_samples, 3].
  """
  # Do a blurpool.

  z_samples = sorted_piecewise_constant_pdf(key, bins, weights, num_samples, randomized)

  # Compute united z_vals and sample points
  # NOTE(debug): visualize coarse and fine samples
  z_samples = jnp.sort(jnp.concatenate([z_vals[:, jitter], z_samples], axis=-1), axis=-1)  # coarse + fine samples
  if stop_grad:
    origins = lax.stop_gradient(origins)
    directions = lax.stop_gradient(directions)
    z_samples = lax.stop_gradient(z_samples)
    z_vals = lax.stop_gradient(z_vals)
    idx_grads = lax.stop_gradient(idx_grads)

  ret = jnp.ones(list(z_samples.shape) + [3 + 3 + 3])
  # Assume that z_vals is monotonically increasing / sorted
  def sorted_find_nearest(xi, x, y):
    # # Shift x points to centers for rounding
    # spacing = jnp.diff(x) / 2
    # x = x + jnp.hstack([spacing, spacing[-1]])
    # Append head, tail for indexing
    y = jnp.hstack([y[0], y, y[-1]])
    return y[jnp.searchsorted(x, xi, side="left")]
  def loop_body(i, val):
    idx = sorted_find_nearest(z_samples[i], z_vals[i], jnp.arange(z_vals.shape[1]))
    rd = directions[i, idx]
    val = val.at[i, :, :3].set(origins[i, idx] + rd * (z_samples[i] - z_vals[i, idx])[..., None])
    val = val.at[i, :, 3:6].set(rd)
    val = val.at[i, :, 6:9].set(idx_grads[i, idx])
    return val
  ret = jax.lax.fori_loop(0, ret.shape[0], loop_body, ret)
  # for i in range(z_vals.shape[0]):
  #   ret = ret.at[i].set(directions[i, sorted_find_nearest(z_samples[i], z_vals[i], jnp.arange(z_vals.shape[1]))])
  
  # t = jnp.concatenate([z_samples[:, 0:1] - near, z_samples[:, 1:] - z_samples[:, :-1]], axis=-1)[..., None]
  # coords = jnp.cumsum(ret[..., :3] * t, axis=1) + origins[:, 0:1]
  return z_samples, ret[..., :3], ret[..., 3:6], ret[..., 6:9]


def add_gaussian_noise(key, raw, noise_std, randomized):
  """Adds gaussian noise to `raw`, which can used to regularize it.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    raw: jnp.ndarray(float32), arbitrary shape.
    noise_std: float, The standard deviation of the noise to be added.
    randomized: bool, add noise if randomized is True.

  Returns:
    raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
  """
  if (noise_std is not None) and randomized:
    return raw + random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
  else:
    return raw
