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
"""Training script for Nerf."""

import functools
import gc
import time
from os import path
from absl import app
from absl import flags
import pickle
import numpy as np

import flax
import optax
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze, unfreeze
import jax
from jax import config
from jax import random
import jax.numpy as jnp

from rnerf import datasets
from rnerf import math_utils
from rnerf import models
from rnerf import utils
from rnerf import ior_utils

# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

FLAGS = flags.FLAGS

utils.define_flags()
config.parse_flags_with_absl()


def train_step(model, rng, state, batch):
  """One optimization step.

  Args:
    model: The linen model.
    rng: jnp.ndarray, random number generator.
    state: utils.TrainState, state of the model/optimizer.
    batch: dict, a mini-batch of data for training.
    lr: float, real-time learning rate.

  Returns:
    new_state: utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
    rng: jnp.ndarray, updated random number generator.
  """
  rng, key_0, key_1 = random.split(rng, 3)

  def loss_fn(variables):
    annealed_alpha = batch["annealed_alpha"][0]
    beta0 = 1.
    beta1 = 1.
    if FLAGS.stage.startswith("radiance") or FLAGS.stage.startswith("all"):
      rays = batch["rays"]
      ret, loss_sp = model.apply(variables, key_0, key_1, rays, FLAGS.randomized, annealed_alpha)
      if len(ret) not in (1, 2):
        raise ValueError(
            "ret should contain either 1 set of output (coarse only), or 2 sets"
            "of output (coarse as ret[0] and fine as ret[1]).")
      
      # The main prediction is always at the end of the ret list.
      rgb, unused_distance, unused_acc, trans, trans_rgb_bkgd = ret[-1]
      loss = ((rgb - batch["pixels"][..., :3])**2).mean()
      if FLAGS.bg_weight > 0:
        mask_bg = trans > 0.5
        loss_bg = (annealed_alpha > 0) * (mask_bg * jnp.abs(trans_rgb_bkgd - batch["pixels"][..., :3])).sum() / (jnp.sum(mask_bg) + 1)
      else:
        loss_bg = 0.
      if FLAGS.beta_weight > 0:
        # NOTE(nan)
        loss_b = (beta0 * math_utils.safe_log(trans) + beta1 * math_utils.safe_log(1 - trans)).mean()
      else:
        loss_b = 0.
      psnr = utils.compute_psnr(loss)
      if len(ret) > 1:
        # If there are both coarse and fine predictions, we compute the loss for
        # the coarse prediction (ret[0]) as well.
        rgb_c, unused_distance_c, unused_acc_c, trans_c, trans_rgb_bkgd_c = ret[0]
        loss_c = ((rgb_c - batch["pixels"][..., :3])**2).mean()
        loss_b_c = 0.
        loss_bg_c = 0.
        psnr_c = utils.compute_psnr(loss_c)
      else:
        loss_c = 0.
        psnr_c = 0.
        loss_bg_c = 0.
        loss_b_c = 0.
      
      if not FLAGS.use_online_sparsity and FLAGS.sparsity_weight > 0:
        loss_sp, next_coarse_alpha_target, next_fine_alpha_target = model.apply(variables, batch["pts"], batch["coarse_alpha_target"][0], batch["fine_alpha_target"][0], method=model.compute_sparsity_loss)
      else:
        next_coarse_alpha_target, next_fine_alpha_target = 0., 0.
      
      if FLAGS.stage.startswith("all") and (FLAGS.normal_loss_weight + FLAGS.normal_smooth_weight) > 0:
        normal_loss, normal_smooth = model.apply(variables, batch["pts"], batch["grads"], annealed_alpha, method=model.wrapper_compute_normal_loss_and_smooth)
        loss_nrm = FLAGS.normal_loss_weight * normal_loss + FLAGS.normal_smooth_weight * normal_smooth
      else:
        loss_nrm = 0.

      # NOTE(envmap)
      if FLAGS.bg_smooth_weight > 0:
        ps = batch["env_rays"].viewdirs.shape[0]
        rgb_env = model.apply(variables, batch["env_rays"].viewdirs.reshape(-1, 3), method=model.forward_envmap).reshape(ps, ps, -1)
        loss_bg_smooth = (annealed_alpha > 0) * jnp.mean(0.5 * ((rgb_env[1:, :] - rgb_env[:-1, :])**2).reshape(-1) + 0.5 * ((rgb_env[:, 1:] - rgb_env[:, :-1])**2).reshape(-1))
      else:
        loss_bg_smooth = 0.
    elif FLAGS.stage.startswith("ior"):
      # The main prediction is always at the end of the ret list.
      normal_loss, normal_smooth = model.apply(variables, batch["pts"], batch["grads"], annealed_alpha, method=model.wrapper_compute_normal_loss_and_smooth)
      loss_nrm = normal_loss

      loss = 0.
      psnr = 0.
      loss_c = 0.
      psnr_c = 0.
      loss_sp = 0.
      loss_bg_smooth = 0.
      next_coarse_alpha_target = 0.
      next_fine_alpha_target = 0.

    def tree_sum_fn(fn):
      return jax.tree_util.tree_reduce(
          lambda x, y: x + fn(y), variables, initializer=0)

    weight_l2 = (
        tree_sum_fn(lambda z: jnp.sum(z**2)) /
        tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))

    t = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(annealed_alpha, 1.0) + jnp.pi))
    annealing_rate = 0.0#jnp.where(annealed_alpha > 0, 1.0 * (1 - t) + 0.25 * t, 0)
    stats = utils.Stats(
        loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c, weight_l2=weight_l2,
        loss_sp=FLAGS.sparsity_weight * annealing_rate * loss_sp + FLAGS.beta_weight * annealing_rate * (loss_b + loss_b_c), loss_nrm=annealing_rate * loss_nrm,
        annealing_rate=annealed_alpha, coarse_alpha_target=next_coarse_alpha_target, fine_alpha_target=next_fine_alpha_target,
        loss_bg=FLAGS.bg_weight * loss_bg, loss_bg_c=FLAGS.bg_weight * loss_bg_c, loss_bg_smooth=loss_bg_smooth)
    return loss + loss_c + FLAGS.bg_weight * (loss_bg + loss_bg_c) + FLAGS.sparsity_weight * annealing_rate * loss_sp + FLAGS.beta_weight * annealing_rate * (loss_b + loss_b_c) + annealing_rate * loss_nrm + FLAGS.bg_smooth_weight * loss_bg_smooth + FLAGS.weight_decay_mult * weight_l2, stats

  (_, stats), grads = (
      jax.value_and_grad(loss_fn, has_aux=True)(state.params))
  grads = jax.lax.pmean(grads, axis_name="batch")
  stats = jax.lax.pmean(stats, axis_name="batch")

  # Clip the gradient by value.
  if FLAGS.grad_max_val > 0:
    clip_fn = lambda z: jnp.clip(z, -FLAGS.grad_max_val, FLAGS.grad_max_val)
    grads = jax.tree_util.tree_map(clip_fn, grads)

  # Clip the (possibly value-clipped) gradient by norm.
  if FLAGS.grad_max_norm > 0:
    grad_norm = jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), grads, initializer=0))
    mult = jnp.minimum(1, FLAGS.grad_max_norm / (1e-7 + grad_norm))
    grads = jax.tree_util.tree_map(lambda z: mult * z, grads)

  new_state = state.apply_gradients(grads=grads)
  return new_state, stats, rng


def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by process_index() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())

  cfg = utils.load_config()

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")
  
  # Create dataset
  dataset = datasets.get_dataset("train", FLAGS)
  val_dataset = datasets.get_dataset("val", FLAGS)

  rng, key = random.split(rng)
  # NOTE(voxelize): Load voxel grid
  with open(path.join(FLAGS.data_dir, cfg.voxel_grid, "mesh.pkl"), "rb") as f:
    mesh_dict = pickle.load(f)
  if mesh_dict["extent"] > 0:
    nmin = [-mesh_dict["extent"], -mesh_dict["extent"], -mesh_dict["extent"]]
    nmax = [mesh_dict["extent"], mesh_dict["extent"], mesh_dict["extent"]]
  else:
    nmin = mesh_dict["min_point"]
    nmax = mesh_dict["max_point"]
  ndim = [mesh_dict["num_voxels"], mesh_dict["num_voxels"], mesh_dict["num_voxels"]]

  # NOTE(prefilter)
  refractive_index = 0.33 if ("glass" in FLAGS.config) or ("wineglass" in FLAGS.config) or ("pen" in FLAGS.config) or ("torus_skydome-bkgd_cycles" in FLAGS.config) or ("dolphin" in FLAGS.config) or ("lighthouse" in FLAGS.config) or ("yellow" in FLAGS.config) else 0.5
  print(refractive_index)
  if cfg.kernel_size > 0:
    grid = ior_utils.conv3d_normal((mesh_dict["data"] - 1.0) * refractive_index / 0.33 + 1.0, ndim, cfg.kernel_size, cfg.kernel_sigma)
  else:
    grid = jnp.asarray((mesh_dict["data"] - 1.0) * refractive_index / 0.33 + 1.0)

  # Create model
  model, variables = models.construct_nerf(key, dataset.peek(), FLAGS,
                                           ndim=ndim, nmin=nmin, nmax=nmax,
                                           grid=grid)
  print(utils.pretty_repr(variables))

  if FLAGS.stage.startswith("ior"):
    dataset = datasets.Grid("train", FLAGS, grid, ndim, nmax, nmin)
  if FLAGS.stage.startswith("radiance") or FLAGS.stage.startswith("all"):
    # ior_dataset = datasets.Grid("train", FLAGS, grid, ndim, nmax, nmin, target=(mesh_dict["data"] - 1.0) * refractive_index / 0.33 + 1.0)
    extra_dataset = datasets.Grid("train", FLAGS, grid, ndim, nmax, nmin)

  train_pstep = jax.pmap(
      functools.partial(train_step, model),
      axis_name="batch",
      in_axes=(0, 0, 0),
      donate_argnums=(2,))

  def render_fn(variables, key_0, key_1, rays):
    return jax.lax.all_gather(
        model.apply(variables, key_0, key_1, rays, FLAGS.randomized),
        axis_name="batch")

  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0),  # Only distribute the data input.
      donate_argnums=(3,),
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  FLAGS.stage_dir = path.join(FLAGS.train_dir, FLAGS.stage)
  if not utils.isdir(FLAGS.stage_dir):
    utils.makedirs(FLAGS.stage_dir)
  utils.save_flags(FLAGS)
  utils.save_config(FLAGS)

  print(utils.pretty_repr(variables))

  # Create optimizer
  learning_rate_fn = functools.partial(
      utils.learning_rate_decay,
      lr_init=FLAGS.lr_init,
      lr_final=FLAGS.lr_final,
      max_steps=FLAGS.max_steps,
      lr_delay_steps=FLAGS.lr_delay_steps,
      lr_delay_mult=FLAGS.lr_delay_mult)
  learning_rate_fn1 = functools.partial(
      utils.learning_rate_decay,
      lr_init=FLAGS.lr_init,
      lr_final=FLAGS.lr_final,
      max_steps=FLAGS.max_steps,
      lr_start_steps=FLAGS.anneal_delay_steps,
      lr_delay_steps=0,
      lr_delay_mult=FLAGS.lr_delay_mult)
    
  if FLAGS.stage.startswith("radiance"):
    param_labels = {"params": {
      "path_sampler": "zero",
      "bkgd_mlp": "adam_lr_scheduler",
      "coarse_mlp": "adam_lr_scheduler",
    }}
    if FLAGS.num_fine_samples > 0:
      param_labels["params"]["fine_mlp"] = "adam_lr_scheduler"
    param_labels = flax.core.frozen_dict.freeze(param_labels)
  elif FLAGS.stage.startswith("ior"):
    param_labels = flax.core.frozen_dict.freeze({"params": {
      "path_sampler": "adam_lr_scheduler",
      "bkgd_mlp": "zero",
      "coarse_mlp": "zero",
      "fine_mlp": "zero",
    }})
  elif FLAGS.stage.startswith("all"):
    param_labels = {"params": {
      "path_sampler": "adam_lr_scheduler",
      "bkgd_mlp": "adam_lr_scheduler",
      "coarse_mlp": "adam_lr_scheduler",
    }}
    if FLAGS.num_fine_samples > 0:
      param_labels["params"]["fine_mlp"] = "adam_lr_scheduler"
    param_labels = flax.core.frozen_dict.freeze(param_labels)

  tx = optax.multi_transform({"adam": optax.adam(learning_rate=FLAGS.lr_init),
                              "adam_lr_scheduler": optax.adam(learning_rate=learning_rate_fn),
                              "adam_lr_scheduler1": optax.adam(learning_rate=learning_rate_fn1),
                              "zero": optax.set_to_zero()},
                              param_labels)
  state = TrainState.create(apply_fn=model.apply, params=variables, tx=tx)
  print(utils.pretty_repr(state.params))
  del tx, variables

  # Resume training a the step of the last checkpoint.
  state = checkpoints.restore_checkpoint(FLAGS.stage_dir, state)

  init_step = state.step + 1
  dataset.train_it = init_step - 1
  if FLAGS.render_every > 0:
    val_dataset.test_it = init_step // FLAGS.render_every
  if FLAGS.stage.startswith("radiance") or FLAGS.stage.startswith("all"):
    extra_dataset.train_it = init_step - 1
  state = flax.jax_utils.replicate(state)  # to multiple deivce

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.stage_dir)

  # Prefetch_buffer_size = 3 x batch_size
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  n_local_devices = jax.local_device_count()
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_devices)  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  gc.collect()
  stats_trace = []
  reset_timer = True
  # one_batch = None
  coarse_alpha_target_trace, fine_alpha_target_trace, running_window = [], [], 100  # TODO: no resuming, delay one iteration
  for step, batch in zip(range(init_step, FLAGS.max_steps + 1), pdataset):
    if FLAGS.stage.startswith("radiance") or FLAGS.stage.startswith("all"):
      extra_case = next(extra_dataset)
      batch = {**batch, **extra_case}
    batch["annealed_alpha"] = utils.shard(jnp.tile(
      jnp.array([np.maximum(step - FLAGS.anneal_delay_steps, 0) / (FLAGS.anneal_max_steps - FLAGS.anneal_delay_steps)]), [n_local_devices]))
    batch["coarse_alpha_target"] = utils.shard(jnp.tile(
      jnp.array([0. if len(coarse_alpha_target_trace) == 0 else np.mean(coarse_alpha_target_trace)]), [n_local_devices]))
    batch["fine_alpha_target"] = utils.shard(jnp.tile(
      jnp.array([0. if len(fine_alpha_target_trace) == 0 else np.mean(fine_alpha_target_trace)]), [n_local_devices]))
    # if one_batch is None: one_batch = batch
    if reset_timer:
      t_loop_start = time.time()
      reset_timer = False
    lr = learning_rate_fn(step)
    state, stats, keys = train_pstep(keys, state, batch)
    if jax.process_index() == 0:
      stats_trace.append(stats)
    if step % FLAGS.gc_every == 0:
      gc.collect()
    
    # if jax.process_index() == 0:
    #   if len(coarse_alpha_target_trace) > running_window:
    #     coarse_alpha_target_trace.pop(0)
    #   if len(fine_alpha_target_trace) > running_window:
    #     fine_alpha_target_trace.pop(0)
    #   coarse_alpha_target_trace.append(stats.coarse_alpha_target[0])
    #   fine_alpha_target_trace.append(stats.fine_alpha_target[0])
      
    # Log training summaries. This is put behind a process_index check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.process_index() == 0:
      if step % FLAGS.print_every == 0:
        summary_writer.scalar("train_loss", stats.loss[0], step)
        summary_writer.scalar("train_psnr", stats.psnr[0], step)
        summary_writer.scalar("train_loss_coarse", stats.loss_c[0], step)
        summary_writer.scalar("train_psnr_coarse", stats.psnr_c[0], step)
        summary_writer.scalar("weight_l2", stats.weight_l2[0], step)
        summary_writer.scalar("train_loss_sp", stats.loss_sp[0], step)
        summary_writer.scalar("train_loss_nrm", stats.loss_nrm[0], step)
        summary_writer.scalar("train_loss_bg", stats.loss_bg[0], step)
        summary_writer.scalar("train_loss_bg_c", stats.loss_bg_c[0], step)
        summary_writer.scalar("train_loss_bg_smooth", stats.loss_bg_smooth[0], step)
        avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
        avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
        avg_loss_sp = np.mean(np.concatenate([s.loss_sp for s in stats_trace]))
        avg_loss_nrm = np.mean(np.concatenate([s.loss_nrm for s in stats_trace]))
        avg_loss_c = np.mean(np.concatenate([s.loss_c for s in stats_trace]))
        avg_psnr_c = np.mean(np.concatenate([s.psnr_c for s in stats_trace]))
        avg_loss_bg = np.mean(np.concatenate([s.loss_bg for s in stats_trace]))
        avg_loss_bg_c = np.mean(np.concatenate([s.loss_bg_c for s in stats_trace]))
        avg_loss_bg_smooth = np.mean(np.concatenate([s.loss_bg_smooth for s in stats_trace]))
        stats_trace = []
        summary_writer.scalar("train_avg_loss", avg_loss, step)
        summary_writer.scalar("train_avg_psnr", avg_psnr, step)
        summary_writer.scalar("train_avg_loss_sp", avg_loss_sp, step)
        summary_writer.scalar("train_avg_loss_nrm", avg_loss_nrm, step)
        summary_writer.scalar("train_avg_loss_coarse", avg_loss_c, step)
        summary_writer.scalar("train_avg_psnr_coarse", avg_psnr_c, step)
        summary_writer.scalar("learning_rate", lr, step)
        summary_writer.scalar("learning_rate1", learning_rate_fn1(step), step)
        summary_writer.scalar("annealing_rate", stats.annealing_rate[0], step)
        summary_writer.scalar("train_avg_loss_bg", avg_loss_bg, step)
        summary_writer.scalar("train_avg_loss_bg_c", avg_loss_bg_c, step)
        summary_writer.scalar("train_avg_loss_bg_smooth", avg_loss_bg_smooth, step)
        steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
        reset_timer = True
        rays_per_sec = FLAGS.batch_size * steps_per_sec
        summary_writer.scalar("train_steps_per_sec", steps_per_sec, step)
        summary_writer.scalar("train_rays_per_sec", rays_per_sec, step)
        precision = int(np.ceil(np.log10(FLAGS.max_steps))) + 1
        print(("{:" + "{:d}".format(precision) + "d}").format(step) +
              f"/{FLAGS.max_steps:d}: " + f"i_loss={stats.loss[0]:0.4f}, " +
              f"avg_loss={avg_loss:0.4f}, " + f"avg_loss_c={avg_loss_c:0.4f}, " +
              f"avg_loss_bg={avg_loss_bg:0.4f}, " +
              f"weight_l2={stats.weight_l2[0]:0.2e}, " + f"lr={lr:0.2e}, " +
              f"{rays_per_sec:0.0f} rays/sec")
      if step % FLAGS.save_every == 0:
        state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.stage_dir, state_to_save, int(step), keep=100)

    # Val-set evaluation.
    if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      t_eval_start = time.time()
      eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                   state)).params
      val_case = next(val_dataset)
      pred_color, pred_distance, pred_acc = utils.render_image(
          functools.partial(render_pfn, eval_variables),
          val_case["rays"],
          keys[0],
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk)

      # Log eval summaries on host 0.
      if jax.process_index() == 0:
        psnr = utils.compute_psnr(
            ((pred_color - val_case["pixels"])**2).mean())
        ssim = ssim_fn(pred_color, val_case["pixels"])
        eval_time = time.time() - t_eval_start
        num_rays = jnp.prod(jnp.array(val_case["rays"].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar("test_rays_per_sec", rays_per_sec, step)
        print(f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec")
        summary_writer.scalar("test_psnr", psnr, step)
        summary_writer.scalar("test_ssim", ssim, step)
        summary_writer.image("test_pred_color", pred_color, step)
        summary_writer.image("test_pred_disp", pred_distance, step)
        summary_writer.image("test_pred_acc", pred_acc, step)
        summary_writer.image("test_target", val_case["pixels"], step)

  if FLAGS.max_steps % FLAGS.save_every != 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        FLAGS.stage_dir, state, int(FLAGS.max_steps), keep=100)


if __name__ == "__main__":
  app.run(main)
