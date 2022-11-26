"""
Packages: trimesh, PyMCubes
"""
import cv2
# TODO: jax.numpy performance boost
import numpy as np
# import jax.numpy as jnp
import json
from os import path
from tqdm import tqdm
import trimesh
import mcubes
import pickle

import cfg


def to_view_matrix(mat):
  """
  Args:
    - mat: np.ndarray, [4, 4]
  Returns:
    - ret: np.ndarray, [4, 4]
  """
  ret = np.eye(4)
  ret[:3, :3] = mat[:3, :3].T
  ret[:3, 3] = (-mat[:3, :3].T @ mat[:3, 3:]).reshape(-1)
  return ret

def project_2d(pts, cam_mat, view_mat):
  """
  Args
    - pts: np.ndarray, [H, W, D, 4]
    - cam_mat: np.ndarray, [4, 4]
    - view_mat: np.ndarray, [4, 4]
  Returns:
    - uv: np.ndarray, [H, W, 2]
    - z: np.ndarray, [H, w]
  """
  pv_mat = cam_mat @ view_mat
  uv = np.einsum('ij,nklj->nkli', pv_mat, pts)
  z = uv[..., 2]
  uv[..., :2] /= uv[..., 2:3]
  return uv, z

def unit_test_proejct_origin():
  pts = np.array([0., 0., 0., 1.0]).reshape(1, 1, 1, 4)

  with open(path.join(cfg.root, 'calib.json'), 'r') as f:
    calib = json.load(f)

  cam_mat = np.array(calib['cam_mat'])
  p_mat = np.concatenate([cam_mat, np.zeros((3, 1))], axis=1)

  for frame_dict in calib['frames']:
    img = cv2.imread(frame_dict['file_path'])
    v_mat = to_view_matrix(np.array(frame_dict['transform_matrix']))

    uv, z = project_2d(pts, p_mat, v_mat)

    uv = uv.reshape(-1, 3)
    for i in range(uv.shape[0]):
      img = cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), radius=1, thickness=20, color=(0, 0, 255))
      # img[(int(uv[i, 1]) - 10):(int(uv[i, 1]) + 10), (int(uv[i, 0]) - 10):(int(uv[i, 0]) + 10)] = 0
    dictionary, fname = path.split(frame_dict['file_path'])
    cv2.imwrite(path.join(dictionary, '0_' + fname), img)

def create_init_bounding_box(trans_mats):
  poses = np.array(trans_mats)[:, :3, 3]
  pose_avg = np.mean(poses, axis=0)
  max_point = np.max(poses, axis=0)
  min_point = np.min(poses, axis=0)
  side = np.max(max_point - min_point) * 1.5
  return pose_avg + np.ones_like(pose_avg) * side * 0.5, pose_avg - np.ones_like(pose_avg) * side * 0.5

def main():
  with open(path.join(cfg.root, 'calib.json'), 'r') as f:
    calib = json.load(f)

  # import os
  # train = {}
  # train['cam_mat'] = calib['cam_mat']
  # idx = [i for i in range(len(calib['frames']))][::10]
  # train['frames'] = []
  # for i in range(len(calib['frames'])):
  #   if i not in idx:
  #     train['frames'].append(calib['frames'][i])
  # with open(os.path.join(cfg.root, 'train.json'), 'w') as f:
  #   json.dump(train, f)
  # exit()

  cam_mat = np.array(calib['cam_mat'])
  p_mat = np.concatenate([cam_mat, np.zeros((3, 1))], axis=1)

  # Load files
  mask_fnames = []
  trans_mats = []
  view_mats = []
  for frame_dict in calib['frames']:
    dictionary, fname = path.split(frame_dict['file_path'])
    mask_fnames.append(path.join(dictionary, 'mask_' + fname[:-3] + 'png'))
    trans_mats.append(np.array(frame_dict['transform_matrix']))
    view_mats.append(to_view_matrix(np.array(frame_dict['transform_matrix'])))

  num_imgs = len(mask_fnames)

  # Create voxel grid
  if cfg.max_point is None or cfg.min_point is None:
    max_point, min_point = create_init_bounding_box(trans_mats)
  else:
    max_point, min_point = cfg.max_point, cfg.min_point
  Y, X, Z = np.meshgrid(np.linspace(0, 1, cfg.num_voxels),
                        np.linspace(0, 1, cfg.num_voxels),
                        np.linspace(0, 1, cfg.num_voxels))
  x_max, y_max, z_max = max_point
  x_min, y_min, z_min = min_point

  X = X * (x_max - x_min) + x_min
  Y = Y * (y_max - y_min) + y_min
  Z = Z * (z_max - z_min) + z_min
  pts = np.concatenate([np.stack([X, Y, Z], axis=-1), np.ones((cfg.num_voxels, cfg.num_voxels, cfg.num_voxels, 1))], axis=-1)
  count = np.zeros((cfg.num_voxels, cfg.num_voxels, cfg.num_voxels))

  # Project visual hull
  for view_mat, mask_fname in tqdm(zip(view_mats, mask_fnames), total=num_imgs):
    mask_img = cv2.imread(mask_fname)[..., 0]
    uvs, zs = project_2d(pts, p_mat, view_mat)

    us = np.clip(np.round(uvs[..., 0]), 0, mask_img.shape[1] - 1).astype(int)  # width
    vs = np.clip(np.round(uvs[..., 1]), 0, mask_img.shape[0] - 1).astype(int)  # height

    inside = mask_img[vs.reshape(-1), us.reshape(-1)] > 0
    inside = inside.reshape(cfg.num_voxels, cfg.num_voxels, cfg.num_voxels)
    count += inside
  
  count /= num_imgs

  # Marching cube
  with open(path.join(cfg.root, 'mesh.pkl'), 'wb') as f:
    pickle.dump({
      "data": (count > cfg.threshold).reshape(-1, 1) * 0.33 + 1.0,  # IoR of glass is 1.5
      "extent": 0,
      "min_point": min_point,
      "max_point": max_point,
      "num_voxels": cfg.num_voxels,
    }, f)

  vertices, triangles = mcubes.marching_cubes(count > cfg.threshold, 0.5)
  print(f'Marching cube: {vertices.shape} vertices, {triangles.shape} triangles')
  
  vertices /= cfg.num_voxels
  vertices[..., 0] = vertices[..., 0] * (x_max - x_min) + x_min
  vertices[..., 1] = vertices[..., 1] * (y_max - y_min) + y_min
  vertices[..., 2] = vertices[..., 2] * (z_max - z_min) + z_min

  mesh = trimesh.Trimesh(vertices, triangles)
  mesh.export(path.join(cfg.root, f'mesh_{cfg.num_voxels}_0_{cfg.threshold}.obj'))

if __name__ == '__main__':
  # unit_test_proejct_origin()
  main()
