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
"""Different datasets implementation plus a general port for all the datasets."""
INTERNAL = False  # pylint: disable=g-statement-before-imports
import json
import os
from os import path
import queue
import threading
if not INTERNAL:
  import cv2  # pylint: disable=g-import-not-at-top
import numpy as np
from PIL import Image
from glob import glob

import jax

from rnerf import utils

def get_dataset(split, args):
  return dataset_dict[args.dataset](split, args)


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane
  t = -(near + origins[..., 2]) / directions[..., 2]
  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)
  return origins, directions


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, args):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.use_pixel_centers = args.use_pixel_centers
    self.split = split
    # NOTE(nerf): Central crop
    self.train_it = 0
    self.test_it = 0
    self.precrop_iters = args.precrop_iters
    self.precrop_frac = args.precrop_frac
    # NOTE(envmap)
    self.patch_size = args.bg_patch_size
    if split == "train":
      self._train_init(args)
    elif split in ("val", "test"):
      self._test_init(args)
    else:
      raise ValueError(
          "the split argument should be either \"train\" or \"val\" or \"test\", set"
          "to {} here.".format(split))
    self.batch_size = args.batch_size // jax.process_count()
    self.batching = args.batching
    self.render_path = args.render_path
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.get()
    if self.split == "train":
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == "train":
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    if self.split == "train":
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, args):
    """Initialize training."""
    self._load_renderings(args)
    self._generate_rays()

    if args.batching == "all_images":
      # flatten the ray and image dimension together.
      self.images = self.images.reshape([-1, 3])
      self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                       self.rays)
    elif args.batching == "single_image":
      self.images = self.images.reshape([-1, self.resolution, 3])
      self.rays = utils.namedtuple_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
    else:
      raise NotImplementedError(
          f"{args.batching} batching strategy is not implemented.")

  def _test_init(self, args):
    self._load_renderings(args)
    self._generate_rays()

  def _next_train(self):
    """Sample next training batch."""

    if self.batching == "all_images":
      ray_indices = np.random.choice(self.rays[0].shape[0],
                                     (self.batch_size,), replace=False)
      batch_pixels = self.images[ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
    elif self.batching == "single_image":
      image_index = np.random.randint(0, self.n_examples, ())
      if self.train_it < self.precrop_iters:
        dH = int(self.h//2 * self.precrop_frac)
        dW = int(self.w//2 * self.precrop_frac)
        coords = np.arange(self.rays[0][0].shape[0]).reshape(self.h, self.w)[(self.h//2 - dH):(self.h//2 + dH), (self.w//2 - dW):(self.w//2 + dW)]
        ray_indices = np.random.choice(coords.reshape(-1), (self.batch_size,), replace=False)
        if self.train_it == 0:
          print(f"Center cropping of size {2*dH} x {2*dW} is enable until iter {self.precrop_iters}")     
      else:
        ray_indices = np.random.choice(self.rays[0][0].shape[0],
                                       (self.batch_size,), replace=False)
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                        self.rays)
    else:
      raise NotImplementedError(
          f"{self.batching} batching strategy is not implemented.")
    
    # NOTE(envmap)
    if self.patch_size > 0:
      image_index = np.random.randint(0, self.n_examples, ())
      if self.train_it < self.precrop_iters:
        dH = int(self.h//2 * self.precrop_frac)
        dW = int(self.w//2 * self.precrop_frac)
        coords = np.arange(self.rays[0][0].shape[0]).reshape(self.h, self.w)[(self.h//2 - dH):(self.h//2 + dH), (self.w//2 - dW):(self.w//2 + dW)]
        pH, pW = coords.shape
        x = np.random.randint(low=0, high=pW - self.patch_size)
        y = np.random.randint(low=0, high=pH - self.patch_size)
        ray_indices = coords[y:(y + self.patch_size), x:(x + self.patch_size)]
      else:
        coords = np.arange(self.rays[0][0].shape[0]).reshape(self.h, self.w)
        x = np.random.randint(low=0, high=self.w - self.patch_size)
        y = np.random.randint(low=0, high=self.h - self.patch_size)
        ray_indices = coords[y:(y + self.patch_size), x:(x + self.patch_size)]
      batch_env_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                            self.rays)
    else:
      batch_env_rays = None
    
    self.train_it += 1
    return {"pixels": batch_pixels, "rays": batch_rays, "env_rays": batch_env_rays}

  def _next_test(self):
    """Sample next test example."""
    idx = self.test_it
    self.test_it = (self.test_it + 1) % self.n_examples

    if self.render_path:
      return {"rays": utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
    else:
      return {
          "pixels": self.images[idx],
          "rays": utils.namedtuple_map(lambda r: r[idx], self.rays)
      }

  # TODO(bydeng): Swap this function with a more flexible camera model.
  def _generate_rays(self):
    """Generating rays for all images."""
    pixel_center = 0.5 if self.use_pixel_centers else 0.0
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")
    camera_dirs = np.stack([(x - self.w * 0.5) / self.focal,
                            -(y - self.h * 0.5) / self.focal, -np.ones_like(x)],
                           axis=-1)
    directions = ((camera_dirs[None, ..., None, :] *
                   self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    self.rays = utils.Rays(
        origins=origins, directions=directions, viewdirs=viewdirs, radii=radii)


class Grid(Dataset):
  """Gradient Index Dataset."""

  def __init__(self, split, args, grid, ndim, nmax, nmin, target=None):
    self.ndim = ndim
    self.nmax = nmax
    self.nmin = nmin
    self.ndelta = [
      (self.nmax[0] - self.nmin[0]) / (self.ndim[0] - 1.),
      (self.nmax[1] - self.nmin[1]) / (self.ndim[1] - 1.),
      (self.nmax[2] - self.nmin[2]) / (self.ndim[2] - 1.),
    ]
    grid = self._compute_grad(grid)
    if target is None:
      self.grid = grid
    else:
      self.grid = self._compute_grad(target)
    self.candidate_indices = np.stack(np.where(np.linalg.norm(grid, axis=-1) > 1e-3), axis=-1)

    self.extra_batch_size = args.extra_batch_size // jax.process_count()
    super().__init__(split, args)

  def _train_init(self, args):
    self.grid = self.grid.reshape(-1, 3)

  def _next_train(self):
    batch_indices = np.random.choice(self.candidate_indices.shape[0], self.extra_batch_size)
    batch_pts = self.candidate_indices[batch_indices] / np.array(self.ndim)[None]  # [batch, 3]
    batch_pts = batch_pts * (np.array(self.nmax)[None] - np.array(self.nmin)[None]) + np.array(self.nmin)[None]
    batch_pts += np.random.uniform(low=-1.0, high=1.0, size=batch_pts.shape) * np.array(self.ndelta)[None]
    batch_grads = self._linear3(self.grid, batch_pts)
    return {"pts": batch_pts[:, None], "grads": batch_grads[:, None]}

  def _linear3(self, data, pts):
    x = (pts[..., 0] - self.nmin[0]) / self.ndelta[0]
    y = (pts[..., 1] - self.nmin[1]) / self.ndelta[1]
    z = (pts[..., 2] - self.nmin[2]) / self.ndelta[2]

    # 8 neighbor vertex
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    xd = ((x - x0) / (x1 - x0))[..., None]
    yd = ((y - y0) / (y1 - y0))[..., None]
    zd = ((z - z0) / (z1 - z0))[..., None]

    # clamp to edge
    x0 = np.clip(x0, 0, self.ndim[0]-1)
    x1 = np.clip(x1, 0, self.ndim[0]-1)
    y0 = np.clip(y0, 0, self.ndim[1]-1)
    y1 = np.clip(y1, 0, self.ndim[1]-1)
    z0 = np.clip(z0, 0, self.ndim[2]-1)
    z1 = np.clip(z1, 0, self.ndim[2]-1)

    # 7 linear interpolation
    c00 = data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y0+z0]*(1-xd) + data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y0+z0]*xd
    c01 = data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y0+z1]*(1-xd) + data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y0+z1]*xd
    c10 = data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y1+z0]*(1-xd) + data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y1+z0]*xd
    c11 = data[self.ndim[1]*self.ndim[2]*x0+self.ndim[2]*y1+z1]*(1-xd) + data[self.ndim[1]*self.ndim[2]*x1+self.ndim[2]*y1+z1]*xd

    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    c = c0*(1-zd) + c1*zd
    return c

  def _compute_grad(self, grid):
    paaded = np.pad(grid.reshape(*self.ndim, 1)[..., 0],
                    ((1, 1), (1, 1), (1, 1)), 'edge')
    # TODO: boundary case is only one step size by central difference
    dx = (paaded[2:, 1:-1, 1:-1] - paaded[:-2, 1:-1, 1:-1]) / (2 * self.ndelta[0])
    dy = (paaded[1:-1, 2:, 1:-1] - paaded[1:-1, :-2, 1:-1]) / (2 * self.ndelta[1])
    dz = (paaded[1:-1, 1:-1, 2:] - paaded[1:-1, 1:-1, :-2]) / (2 * self.ndelta[2])
    return np.stack([dx, dy, dz], axis=-1)  # [N_x, N_y, N_z, 3]
  
  def _test_init(self, args):
    raise NotImplementedError()

  def _next_test(self):
    raise NotImplementedError()


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    if args.render_path:
      raise ValueError("render_path cannot be used for the blender dataset.")
    with utils.open_file(
        path.join(args.data_dir, "transforms_{}.json".format("train" if args.eval_train else self.split)),
        "r") as fp:
      meta = json.load(fp)
    images = []
    cams = []
    for i in range(0, len(meta["frames"]), args.skip_frames):
      frame = meta["frames"][i]
      fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
      with utils.open_file(fname, "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if args.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif args.factor > 0:
          raise ValueError("Blender dataset only supports factor=0 or 2, {} "
                           "set.".format(args.factor))
      cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
      images.append(image)
    self.images = np.stack(images, axis=0)
    if args.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    camera_angle_x = float(meta["camera_angle_x"])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.n_examples = self.images.shape[0]


class NSVF(Dataset):
  """NSVF Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    if args.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')
    prefix = {
      'train': 0,
      'val': 1,
      'test': 2,
    }
    with open(os.path.join(args.data_dir, 'intrinsics.txt'), 'r') as fp:
      f, cx, cy, _ = map(float, fp.readline().split())
    images = []
    cams = []
    imgfiles = sorted(glob(os.path.join(args.data_dir, 'rgb', f'{int(prefix[self.split])}_*.png')))
    camfiles = sorted(glob(os.path.join(args.data_dir, 'pose', f'{int(prefix[self.split])}_*.txt')))
    for i in range(len(imgfiles)):
      # Load image
      with utils.open_file(imgfiles[i], 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if args.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif args.factor > 0:
          raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                            'set.'.format(args.factor))
      images.append(image)

      # Load camera
      cam = np.loadtxt(camfiles[i], dtype=np.float32)
      cam[:3, 1:3] *= -1  # flip Y, Z axes
      cams.append(cam)
    self.images = np.stack(images, axis=0)
    if args.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    self.focal = f
    if args.factor == 2:
      self.focal *= 0.5
    elif args.factor > 0:
      raise NotImplementedError()
    self.n_examples = self.images.shape[0]


class OpenCV(Dataset):
  """OpenCV Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    if args.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')
    with utils.open_file(
        path.join(args.data_dir, 'transforms_{}.json'.format("train" if args.eval_train else self.split)),
        'r') as fp:
      meta = json.load(fp)
    images = []
    cams = []
    for i in range(0, len(meta['frames']), args.skip_frames):
      frame = meta['frames'][i]
      fname = os.path.join(args.data_dir, frame['file_path'])
      with utils.open_file(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        # if args.factor == 2:
        #   [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
        #   image = cv2.resize(
        #       image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        if args.factor > 0:
          raise ValueError('Opencv dataset does not support factor, {} '
                           'set.'.format(args.factor))
      cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
      images.append(image)
    self.images = np.stack(images, axis=0)
    if args.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    self.cam_mat = meta['cam_mat']
    self.n_examples = self.images.shape[0]

  def _next_test(self):
    """Sample next test example."""
    idx = self.test_it
    self.test_it = (self.test_it + 1) % self.n_examples

    if self.precrop_iters > 0:
      dH = int(self.h//2 * self.precrop_frac)
      dW = int(self.w//2 * self.precrop_frac)
    else:
      dH = self.h//2
      dW = self.w//2

    if self.render_path:
      return {"rays": utils.namedtuple_map(lambda r: r[idx][(self.h//2 - dH):(self.h//2 + dH), (self.w//2 - dW):(self.w//2 + dW)], self.render_rays)}
    else:
      return {
          "pixels": self.images[idx][(self.h//2 - dH):(self.h//2 + dH), (self.w//2 - dW):(self.w//2 + dW)],
          "rays": utils.namedtuple_map(lambda r: r[idx][(self.h//2 - dH):(self.h//2 + dH), (self.w//2 - dW):(self.w//2 + dW)], self.rays)
      }

  def _generate_rays(self):
    """Generating rays for all images."""
    pixel_center = 0.5 if self.use_pixel_centers else 0.0
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    # camera_dirs = np.stack(
    #     [(x - self.cam_mat[0][2] + 0.5) / self.cam_mat[0][0],
    #      -(y - self.cam_mat[1][2] + 0.5) / self.cam_mat[1][1], -np.ones_like(x)],
    #     axis=-1)
    camera_dirs = np.stack([
        (x - self.cam_mat[0][2] + pixel_center) / self.cam_mat[0][0],  # X
        (y - self.cam_mat[1][2] + pixel_center) / self.cam_mat[1][1],  # Y
        np.ones_like(x)  # Z
      ], axis=-1)
    directions = ((camera_dirs[None, ..., None, :] *
                   self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    self.rays = utils.Rays(
        origins=origins, directions=directions, viewdirs=viewdirs, radii=radii)


class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    # Load images.
    imgdir_suffix = ""
    if args.factor > 0:
      imgdir_suffix = "_{}".format(args.factor)
      factor = args.factor
    else:
      factor = 1
    imgdir = path.join(args.data_dir, "images" + imgdir_suffix)
    if not utils.file_exists(imgdir):
      raise ValueError("Image folder {} doesn't exist.".format(imgdir))
    imgfiles = [
        path.join(imgdir, f)
        for f in sorted(utils.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg")# or f.endswith("png")
    ]
    images = []
    for imgfile in imgfiles:
      with utils.open_file(imgfile, "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        images.append(image)
    images = np.stack(images, axis=-1)

    # Load poses and bds.
    with utils.open_file(path.join(args.data_dir, "poses_bounds.npy"),
                         "rb") as fp:
      poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    if poses.shape[-1] != images.shape[-1]:
      raise RuntimeError("Mismatch between imgs {} and poses {}".format(
          images.shape[-1], poses.shape[-1]))

    # Update poses according to downsampling.
    poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale according to a default bd factor.
    scale = 1. / (bds.min() * .75)
    poses[:, :3, 3] *= scale
    bds *= scale

    # Recenter poses.
    poses = self._recenter_poses(poses)

    # Generate a spiral/spherical ray path for rendering videos.
    if args.spherify:
      poses = self._generate_spherical_poses(poses, bds)
      self.spherify = True
    else:
      self.spherify = False
    if not args.spherify and self.split == "test":
      self._generate_spiral_poses(poses, bds)

    # Select the split.
    i_test = np.arange(images.shape[0])[::args.llffhold]
    i_train = np.array(
        [i for i in np.arange(int(images.shape[0])) if i not in i_test])
    if self.split == "train":
      # indices = i_train
      indices = np.arange(100, 200, 1)
    else:
      # indices = i_test
      indices = np.arange(0, 100, 1)
    images = images[indices]
    poses = poses[indices]

    self.images = images
    self.camtoworlds = poses[:, :3, :4]
    self.focal = poses[0, -1, -1]
    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w
    if args.render_path:
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]

  def _generate_rays(self):
    """Generate normalized device coordinate rays for llff."""
    if self.split == "test":
      n_render_poses = self.render_poses.shape[0]
      self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                        axis=0)

    super()._generate_rays()

    if not self.spherify:
      ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                   self.rays.directions,
                                                   self.focal, self.w, self.h)
      mat = ndc_origins
      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :])**2, -1))
      dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

      dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :])**2, -1))
      dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
      # Cut the distance in half, and then round it out so that it's
      # halfway between inscribed by / circumscribed about the pixel.
      radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

      self.rays = utils.Rays(
          origins=ndc_origins,
          directions=ndc_directions,
          viewdirs=self.rays.directions,
          radii=radii)

    # Split poses from the dataset and generated poses
    if self.split == "test":
      self.camtoworlds = self.camtoworlds[n_render_poses:]
      split = [np.split(r, [n_render_poses], 0) for r in self.rays]
      split0, split1 = zip(*split)
      self.render_rays = utils.Rays(*split0)
      self.rays = utils.Rays(*split1)

  def _recenter_poses(self, poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = self._poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

  def _poses_avg(self, poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = self._normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
    return c2w

  def _viewmatrix(self, z, up, pos):
    """Construct lookat view matrix."""
    vec2 = self._normalize(z)
    vec1_avg = up
    vec0 = self._normalize(np.cross(vec1_avg, vec2))
    vec1 = self._normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

  def _normalize(self, x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

  def _generate_spiral_poses(self, poses, bds):
    """Generate a spiral path for rendering."""
    c2w = self._poses_avg(poses)
    # Get average pose.
    up = self._normalize(poses[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset.
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    # Get radii for spiral path.
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    n_views = 120
    n_rots = 2
    # Generate poses for spiral path.
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w_path[:, 4:5]
    zrate = .5
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
      c = np.dot(c2w[:3, :4], (np.array(
          [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
      z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
      render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
    self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

  def _generate_spherical_poses(self, poses, bds):
    """Generate a 360 degree spherical path for rendering."""
    # pylint: disable=g-long-lambda
    p34_to_44 = lambda p: np.concatenate([
        p,
        np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
    ], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
      a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
      b_i = -a_i @ rays_o
      pt_mindist = np.squeeze(-np.linalg.inv(
          (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
      return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = self._normalize(up)
    vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
    vec2 = self._normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = (
        np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
      camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
      up = np.array([0, 0, -1.])
      vec2 = self._normalize(camorigin)
      vec0 = self._normalize(np.cross(vec2, up))
      vec1 = self._normalize(np.cross(vec2, vec0))
      pos = camorigin
      p = np.stack([vec0, vec1, vec2, pos], 1)
      new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([
        new_poses,
        np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
    ], -1)
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)
    if self.split == "test":
      self.render_poses = new_poses[:, :3, :4]
    return poses_reset


dataset_dict = {
    "blender": Blender,
    "llff": LLFF,
    "nsvf": NSVF,
    'opencv': OpenCV,
}
