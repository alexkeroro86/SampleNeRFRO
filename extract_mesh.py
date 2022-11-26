from os import path
import functools
import gc
import time
from absl import app
from absl import flags
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mcubes
import trimesh
import imageio
from tqdm import tqdm

import flax
from flax.training import checkpoints
from flax.core.frozen_dict import freeze, unfreeze
import jax
from jax import config
from jax import random
import jax.numpy as jnp

from rnerf import datasets
from rnerf import models
from rnerf import utils
from rnerf import math_utils
from rnerf import plt_utils
from rnerf import ior_utils

# https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# https://github.com/google/jax/issues/1222
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

FLAGS = flags.FLAGS

utils.define_flags()
flags.DEFINE_integer('resolution', 256, 'voxle grid resolution for marching cube')
flags.DEFINE_float('range', 1.2, 'bounding box range for marching cube')
flags.DEFINE_float('threshold', 0.1, 'threshold of isosurface')
config.parse_flags_with_absl()


def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by process_index() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())

  cfg = utils.load_config()
  
  # -----------------------------------------------------------------------------
  # Load trained network weights
  # -----------------------------------------------------------------------------
  
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
  
  # Check magnitude of gradient
  # a = np.array([
  #   [[1, 1, 1, 1, 1, 1.33, 1.33, 1.33, 1.33, 1.33]],
  # ])
  # a = np.tile(a, [10, 10, 1])
  # a = a[..., None]
  # print(a[0])
  # a = a.reshape(-1, 1)
  # b = ior_utils.conv3d_normal(a, [10, 10, 10], cfg.kernel_size, cfg.kernel_sigma)
  # b = b.reshape(10, 10, 10, 1)
  # print(b[0])
  # exit()
  
  # Create model
  model, variables = models.construct_nerf(key, dataset.peek(), FLAGS,
                                           ndim=ndim, nmin=nmin, nmax=nmax,
                                           grid=grid)
  print(utils.pretty_repr(variables))

  # # Set new virtual camera
  # # c2w = math_utils.pose_spherical(30., -20., 4.)
  # # c2w = math_utils.pose_spherical(40., -40., 4.)
  # c2w = math_utils.pose_spherical(0., -90., 4.)
  # # c2w = math_utils.pose_spherical(0., 89., 4.)
  # # c2w = math_utils.pose_spherical(0., 45., 4.)
  # dataset.camtoworlds = np.tile(c2w[None, ...], [dataset.camtoworlds.shape[0], 1, 1])
  # dataset._generate_rays()
  # batch = {
  #   'rays': utils.namedtuple_map(lambda r: r[0], dataset.rays)
  # }

  # Peek from dataset
  c2w = None
  batch = dataset.peek()
  img_idx = 35
  for i in range(img_idx):
    batch = next(dataset)

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

  FLAGS.stage_dir = path.join(FLAGS.train_dir, FLAGS.stage)

  out_dir = path.join(FLAGS.stage_dir, "debug")
  if FLAGS.save_output and (not utils.isdir(out_dir)):
    utils.makedirs(out_dir)

  # Load pre-trained weight
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
      variables["params"]["path_sampler"]["scan"]["idx_model"]["grad_mlp"].update(pretrain["params"]["params"]["path_sampler"]["scan"]["idx_model"]["grad_mlp"])
      variables = freeze(variables)
  elif FLAGS.stage.startswith("all"):
    pretrain = checkpoints.restore_checkpoint(path.join(FLAGS.train_dir, cfg.all_weight_name), None)
    step = int(pretrain["step"])

    variables = unfreeze(variables)
    variables["params"]["bkgd_mlp"].update(pretrain["params"]["params"]["bkgd_mlp"])
    variables["params"]["coarse_mlp"].update(pretrain["params"]["params"]["coarse_mlp"])
    variables["params"]["fine_mlp"].update(pretrain["params"]["params"]["fine_mlp"])
    variables["params"]["path_sampler"]["scan"]["idx_model"]["grad_mlp"].update(pretrain["params"]["params"]["path_sampler"]["scan"]["idx_model"]["grad_mlp"])
    variables = freeze(variables)

  print(utils.pretty_repr(variables))

  # NOTE(eikonal)
  pred_color, pred_distance, pred_acc, ray_pos, ray_dir, idx_grad, pred_trans, ray_pos_c = utils.render_image(
          functools.partial(render_pfn, variables),
          batch["rays"],
          rng,
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk)

  fig = plt.figure()
  plt.imshow(np.clip(np.nan_to_num(pred_color), 0., 1.))
  # plt.show()
  plt.close(fig)
  # pred_color *= jnp.sum(jnp.linalg.norm(idx_grad, axis=-1, keepdims=True) > 1e-3, axis=-2) > 0
  imageio.imwrite(path.join(out_dir, 'color.png'), np.clip(np.nan_to_num(pred_color), 0., 1.))
  imageio.imwrite(path.join(out_dir, 'acc.exr'), np.tile(pred_acc, [1, 1, 3]))
  imageio.imwrite(path.join(out_dir, 'trans.exr'), np.tile(pred_trans, [1, 1, 3]))

  # Render path
  num_steps = ray_pos.shape[-2]
  print(ray_pos.shape)
  print(ray_dir.shape)
  print(idx_grad.shape)
  print(ray_pos_c.shape)

  # ray_pos = ray_pos[::80, ::100]
  # ray_dir = ray_dir[::80, ::100]
  # idx_grad = idx_grad[::80, ::100]
  
  upper_left = (210, 244)
  ray_pos = ray_pos[upper_left[0]:(upper_left[0]+1), upper_left[1]:(upper_left[1]+1)]
  ray_dir = ray_dir[upper_left[0]:(upper_left[0]+1), upper_left[1]:(upper_left[1]+1)]
  idx_grad = idx_grad[upper_left[0]:(upper_left[0]+1), upper_left[1]:(upper_left[1]+1)]
  ray_pos_c = ray_pos_c[upper_left[0]:(upper_left[0]+1), upper_left[1]:(upper_left[1]+1)]

  ray_pos = ray_pos.reshape(-1, num_steps, 3)
  ray_dir = ray_dir.reshape(-1, num_steps, 3)
  idx_grad = idx_grad.reshape(-1, num_steps, 3)
  ray_pos_c = ray_pos_c.reshape(-1, FLAGS.num_coarse_samples, 3)

  with open(path.join(out_dir, f'ray_{(img_idx-1):03d}_{upper_left[0]:03d}_{upper_left[1]:03d}.pkl'), 'wb') as f:
    pickle.dump({
      'ray_pos': ray_pos,
      'ray_dir': ray_dir,
      'idx_grad': idx_grad,
      'transform': c2w,
      'ray_pos_c': ray_pos_c,
    }, f)

  plt_utils.plot_path(ray_pos, out_dir=out_dir)
  exit()

  # -----------------------------------------------------------------------------
  # Query network on dense 3d grid of points
  # -----------------------------------------------------------------------------

  N = FLAGS.resolution
  t = np.linspace(-FLAGS.range, FLAGS.range, N+1)

  query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
  print(query_pts.shape)  # [N+1, N+1, N+1, 3]
  sh = query_pts.shape
  flat = query_pts.reshape([-1,3])

  sigma = []
  for i in tqdm(range(0, flat.shape[0], FLAGS.chunk)):
    viewdirs = np.zeros_like(flat[i:(i+FLAGS.chunk), None, :])
    chunk_density = model.apply(variables, flat[i:(i+FLAGS.chunk), None, :], viewdirs, method=model.sample_points)[1]
    sigma.append(chunk_density)
  sigma = np.reshape(np.concatenate(sigma, 0), list(sh[:-1]))
  print(sigma.shape)

  fig = plt.figure()
  plt.hist(np.maximum(0,sigma.ravel()), log=True)
  plt.show()
  plt.close(fig)

  # -----------------------------------------------------------------------------
  # Marching cubes with PyMCubes
  # -----------------------------------------------------------------------------

  threshold = FLAGS.threshold
  print('fraction occupied', np.mean(sigma > threshold))
  vertices, triangles = mcubes.marching_cubes(sigma, threshold)
  print('done', vertices.shape, triangles.shape)

  # -----------------------------------------------------------------------------
  # Live preview with trimesh
  # -----------------------------------------------------------------------------

  mesh = trimesh.Trimesh(vertices / N - .5, triangles)
  # mesh.show()
  mesh.export(path.join(out_dir, f'mesh_{FLAGS.resolution}_{FLAGS.range}_{FLAGS.threshold}.obj'))

  # -----------------------------------------------------------------------------
  # Save out video with pyrender
  # -----------------------------------------------------------------------------


if __name__ == '__main__':
  app.run(main)
