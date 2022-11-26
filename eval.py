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
"""Evaluation script for Nerf."""
import functools
from os import path
from absl import app
from absl import flags
import pickle
import numpy as np

import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.core.frozen_dict import freeze, unfreeze
import jax
from jax import random
import jax.numpy as jnp


from rnerf import datasets
from rnerf import models
from rnerf import utils
from rnerf import ior_utils
from rnerf import vis

# # https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# # https://github.com/google/jax/issues/1222
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

FLAGS = flags.FLAGS

utils.define_flags()


def main(unused_argv):

  rng = random.PRNGKey(20200823)

  cfg = utils.load_config()

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  dataset = datasets.get_dataset("test", FLAGS)
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

  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates "speckle" artifacts.
  def render_fn(variables, key_0, key_1, rays):
    return jax.lax.all_gather(
        model.apply(variables, key_0, key_1, rays, False), axis_name="batch")

  # pmap over only the data input.
  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0),
      donate_argnums=3,
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  FLAGS.stage_dir = path.join(FLAGS.train_dir, FLAGS.stage)

  last_step = 0
  if not FLAGS.eval_train:
    out_dir = path.join(FLAGS.stage_dir,
                        "path_renders" if FLAGS.render_path else "test_preds")
  else:
    out_dir = path.join(FLAGS.stage_dir, "train_preds")
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.stage_dir, "eval"))
  while True:
    # Load pretrained weight
    if FLAGS.stage.startswith("radiance") or FLAGS.stage.startswith("ior"):
      pretrain = checkpoints.restore_checkpoint(path.join(FLAGS.train_dir, cfg.radiance_weight_name), None)
      step = int(pretrain["step"])

      variables = unfreeze(variables)
      variables["params"]["bkgd_mlp"].update(pretrain["params"]["params"]["bkgd_mlp"])
      variables["params"]["coarse_mlp"].update(pretrain["params"]["params"]["coarse_mlp"])
      if FLAGS.num_fine_samples > 0:
        variables["params"]["fine_mlp"].update(pretrain["params"]["params"]["fine_mlp"])
      variables = freeze(variables)

      if FLAGS.stage.startswith("ior"):
        pretrain = checkpoints.restore_checkpoint(path.join(FLAGS.train_dir, cfg.ior_weight_name), None)
        step = int(pretrain["step"])

        variables = unfreeze(variables)
        variables["params"]["path_sampler"].update(pretrain["params"]["params"]["path_sampler"])
        variables = freeze(variables)
    elif FLAGS.stage.startswith("all"):
      pretrain = checkpoints.restore_checkpoint(path.join(FLAGS.train_dir, cfg.all_weight_name), None)
      step = int(pretrain["step"])

      variables = unfreeze(variables)
      variables["params"]["bkgd_mlp"].update(pretrain["params"]["params"]["bkgd_mlp"])
      variables["params"]["coarse_mlp"].update(pretrain["params"]["params"]["coarse_mlp"])
      if FLAGS.num_fine_samples > 0:
        variables["params"]["fine_mlp"].update(pretrain["params"]["params"]["fine_mlp"])
      variables["params"]["path_sampler"].update(pretrain["params"]["params"]["path_sampler"])
      variables = freeze(variables)

    print(utils.pretty_repr(variables))

    if step <= last_step:
      continue
    if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
    psnr_values = []
    ssim_values = []
    if not FLAGS.eval_once:
      showcase_index = np.random.randint(0, dataset.size)
    for idx in range(dataset.size):
      print(f"Evaluating {idx+1}/{dataset.size}")
      batch = next(dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, variables),
          batch["rays"],
          rng,
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk)
      # pred_mask = jnp.sum(jnp.linalg.norm(idx_grad, axis=-1) > 1e-3, axis=-1) > 0
      
      vis_suite = vis.visualize_suite(pred_disp[..., 0], pred_acc[..., 0])
      
      if jax.process_index() != 0:  # Only record via host 0.
        continue
      if not FLAGS.eval_once and idx == showcase_index:
        showcase_color = pred_color
        showcase_disp = pred_disp
        showcase_acc = pred_acc
        if not FLAGS.render_path:
          showcase_gt = batch["pixels"]
      if not FLAGS.render_path:
        psnr = utils.compute_psnr(((pred_color - batch["pixels"])**2).mean())
        ssim = ssim_fn(pred_color, batch["pixels"])
        print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
        psnr_values.append(float(psnr))
        ssim_values.append(float(ssim))
      if FLAGS.save_output:
        utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
        utils.save_img(pred_disp[..., 0],
                       path.join(out_dir, "disp_{:03d}.png".format(idx)))
        # utils.save_img(pred_mask, path.join(out_dir, "mask_{:03d}.png".format(idx)))
        for k, v in vis_suite.items():
            utils.save_img(
                v, path.join(out_dir, k + '_{:03d}.png'.format(idx)))
    if (not FLAGS.eval_once) and (jax.process_index() == 0):
      summary_writer.image("pred_color", showcase_color, step)
      summary_writer.image("pred_disp", showcase_disp, step)
      summary_writer.image("pred_acc", showcase_acc, step)
      if not FLAGS.render_path:
        summary_writer.scalar("psnr", np.mean(np.array(psnr_values)), step)
        summary_writer.scalar("ssim", np.mean(np.array(ssim_values)), step)
        summary_writer.image("target", showcase_gt, step)
    if FLAGS.save_output and (not FLAGS.render_path) and (jax.process_index() == 0):
      with utils.open_file(path.join(out_dir, f"psnrs_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in psnr_values]))
      with utils.open_file(path.join(out_dir, f"ssims_{step}.txt"), "w") as f:
        f.write(" ".join([str(v) for v in ssim_values]))
      with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(psnr_values))))
      with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as f:
        f.write("{}".format(np.mean(np.array(ssim_values))))
    if FLAGS.eval_once:
      break
    if int(step) >= FLAGS.max_steps:
      break
    last_step = step


if __name__ == "__main__":
  app.run(main)
