import os
from absl import app, flags
import trimesh
import pickle
from tqdm import tqdm
import mcubes
from pysdf import SDF  # NOTE: remove 'experimental/propagate_const' for gcc 5.4.0
import numpy as np
import jax
from jax import config, random
import jax.numpy as jnp

from rnerf import utils

# Diable pre-alloection: https://github.com/google/jax/issues/1222
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

FLAGS = flags.FLAGS

utils.define_flags()
flags.DEFINE_integer("num_samples", 4, "sampling resolution of voxelization")
flags.DEFINE_integer("num_voxels", 128, "resolution of voxel grid")
flags.DEFINE_float("extent", 3, "extent of voxel grid")
flags.DEFINE_multi_float("min_point", [-1, -1, -1], "minimum point of voxel grid")
flags.DEFINE_multi_float("max_point", [ 1,  1,  1], "maximum point of voxel grid")
flags.DEFINE_float("threshold", 1.0, "threshold of isosurface")
config.parse_flags_with_absl()


def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by process_index() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  out_dir = os.path.join(FLAGS.data_dir, "voxelize")
  if not utils.isdir(out_dir):
    utils.makedirs(out_dir)

  # -----------------------------------------------------------------------------
  # Create mesh intersector
  # -----------------------------------------------------------------------------

  mesh = trimesh.load(os.path.join(FLAGS.data_dir, 'mesh.obj'))
  intersector = SDF(mesh.vertices, mesh.faces)

  def contain(x):
    """
    Args:
      - x: np.ndarray, [B, 3]
    Returns:
      np.ndarray, [B, 1]
    """
    inside = intersector.contains(x)[..., None]
    return np.where(inside > 0.5, 1.33, 1.0)
    # return np.where(inside > 0.5, 1., 0.0)

  # -----------------------------------------------------------------------------
  # Generate sample
  # -----------------------------------------------------------------------------

  Y, X, Z = np.meshgrid(np.linspace(-1, 1, FLAGS.num_samples),
                        np.linspace(-1, 1, FLAGS.num_samples),
                        np.linspace(-1, 1, FLAGS.num_samples))
  noise_scale = 2 / (FLAGS.num_samples - 1) * 0.5
  noise = (np.random.rand(FLAGS.num_samples, FLAGS.num_samples, FLAGS.num_samples, 3) * 2 - 1) * noise_scale
  # offset = (np.stack([X, Y, Z], axis=-1) + noise).reshape(-1, 3)
  offset = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

  # -----------------------------------------------------------------------------
  # Create voxel
  # -----------------------------------------------------------------------------

  Y, X, Z = np.meshgrid(np.linspace(0, 1, FLAGS.num_voxels),
                        np.linspace(0, 1, FLAGS.num_voxels),
                        np.linspace(0, 1, FLAGS.num_voxels))
  if FLAGS.extent > 0:
    x_max = y_max = z_max = FLAGS.extent
    x_min = y_min = z_min = -FLAGS.extent
  else:
    x_max, y_max, z_max = FLAGS.max_point
    x_min, y_min, z_min = FLAGS.min_point
  offset_scale = (2 * np.array([x_max - x_min, y_max - y_min, z_max - z_min])[None]) / (FLAGS.num_voxels - 1) * 0.5
  X = X * (x_max - x_min) + x_min
  Y = Y * (y_max - y_min) + y_min
  Z = Z * (z_max - z_min) + z_min
  grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

  out = np.zeros((grid.shape[0], 1))
  # TODO: parallel for-loop
  for i in tqdm(range(grid.shape[0])):
    noise = (np.random.rand(FLAGS.num_samples, FLAGS.num_samples, FLAGS.num_samples, 3) * 2 - 1) * noise_scale
    # sample = grid[i:(i + 1)] + (offset + noise.reshape(-1, 3)) * offset_scale
    sample = grid[i:(i + 1)] + offset * offset_scale
    ior = contain(sample)
    out[i] = np.mean(ior)
    # out[i] = 1.33 if np.mean(ior) > 0.5 else 1

  with open(os.path.join(out_dir, "mesh.pkl"), "wb") as f:
    pickle.dump({
      "data": out,
      "extent": FLAGS.extent,
      "min_point": FLAGS.min_point,
      "max_point": FLAGS.max_point,
      "num_voxels": FLAGS.num_voxels
    }, f)

  # -----------------------------------------------------------------------------
  # Marching cubes with PyMCubes
  # -----------------------------------------------------------------------------

  N = FLAGS.num_voxels
  sigma = out.reshape(N, N, N, 1)[..., 0]
  threshold = FLAGS.threshold
  print('fraction occupied', np.mean(sigma > threshold))
  vertices, triangles = mcubes.marching_cubes(sigma, threshold)
  print('done', vertices.shape, triangles.shape)

  # -----------------------------------------------------------------------------
  # Live preview with trimesh
  # -----------------------------------------------------------------------------

  mesh = trimesh.Trimesh(vertices / N - .5, triangles)
  # mesh.show()
  mesh.export(os.path.join(out_dir, f'mesh_{FLAGS.num_samples}_{FLAGS.num_voxels}_{FLAGS.extent}_{FLAGS.threshold}.obj'))


if __name__ == "__main__":
  app.run(main)
