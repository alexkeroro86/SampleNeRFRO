import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


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

def main():
  # Config
  DATASET = 'opencv'

  data_dir = '/home/cgv839/data/real/dolphin'

  # Load
  with open(os.path.join(data_dir, 'transforms_test.json'), 'r') as f:
    meta = json.load(f)

  img = cv2.imread(os.path.join(data_dir, meta['frames'][0]['file_path'] + ('.png' if DATASET == "blender" else "")))
  h, w = img.shape[:2]

  # Define 3D coordinates of the corners for each of the AprilTags
  tag_pts_dict = {}
  scale = 0.1
  # Face 0~7
  anchor_z = scale * 0
  for k, anchor_x, anchor_y in zip(list(range(8)), [0, 2.7, 5.4, 5.4, 5.4, 2.7, 0, 0], [0, 0, 0, 2.7, 5.4, 5.4, 5.4, 2.7]):
    for i in range(3):
      for j in range(3):
        x = scale * anchor_x + scale * 0.9 * j
        y = scale * anchor_y + scale * 0.9 * i
        tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
          [x + scale * 0.7, y + scale * 0.2, anchor_z, 1.0],
          [x + scale * 0.7, y + scale * 0.7, anchor_z, 1.0],
          [x + scale * 0.2, y + scale * 0.7, anchor_z, 1.0],
          [x + scale * 0.2, y + scale * 0.2, anchor_z, 1.0]], dtype=np.float32)
  # Face 8
  anchor_x = scale * 5.4
  anchor_y, anchor_z = 2.7, 2.7
  k = 8
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      y = scale * anchor_y + scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [anchor_x, y + scale * 0.2, z - scale * 0.7, 1.0],
        [anchor_x, y + scale * 0.7, z - scale * 0.7, 1.0],
        [anchor_x, y + scale * 0.7, z - scale * 0.2, 1.0],
        [anchor_x, y + scale * 0.2, z - scale * 0.2, 1.0]], dtype=np.float32)
  # Face 10
  anchor_x = scale * 2.7
  anchor_y, anchor_z = 5.4, 2.7
  k = 10
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      y = scale * anchor_y - scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [anchor_x, y - scale * 0.2, z - scale * 0.7, 1.0],
        [anchor_x, y - scale * 0.7, z - scale * 0.7, 1.0],
        [anchor_x, y - scale * 0.7, z - scale * 0.2, 1.0],
        [anchor_x, y - scale * 0.2, z - scale * 0.2, 1.0]], dtype=np.float32)
  # Face 9
  anchor_y = scale * 5.4
  anchor_z, anchor_x = 2.7, 5.4
  k = 9
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      x = scale * anchor_x - scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [x - scale * 0.2, anchor_y, z - scale * 0.7, 1.0],
        [x - scale * 0.7, anchor_y, z - scale * 0.7, 1.0],
        [x - scale * 0.7, anchor_y, z - scale * 0.2, 1.0],
        [x - scale * 0.2, anchor_y, z - scale * 0.2, 1.0]], dtype=np.float32)
  # Face 11
  anchor_y = scale * 2.7
  anchor_z, anchor_x = 2.7, 2.7
  k = 11
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      x = scale * anchor_x + scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [x + scale * 0.2, anchor_y, z - scale * 0.7, 1.0],
        [x + scale * 0.7, anchor_y, z - scale * 0.7, 1.0],
        [x + scale * 0.7, anchor_y, z - scale * 0.2, 1.0],
        [x + scale * 0.2, anchor_y, z - scale * 0.2, 1.0]], dtype=np.float32)

  # # DEBUG: calibration pattern
  # fig = plt.figure(figsize=(8, 8))
  # ax = fig.add_subplot(projection='3d')
  # for key, pts in tag_pts_dict.items():
  #   ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
  # ax.set_xlim(0, 1)
  # ax.set_ylim(0, 1)
  # ax.set_zlim(0, 1)
  # ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
  # plt.show()
  # exit()

  cam_mat = np.array(meta['cam_mat'])
  p_mat = np.concatenate([cam_mat, np.zeros((3, 1))], axis=1)

  cmap = cm.rainbow(np.linspace(0, 1, len(tag_pts_dict.keys())))


  for frame_dict in meta['frames']:
    img = cv2.imread(os.path.join(data_dir, frame_dict['file_path']))
    v_mat = to_view_matrix(np.array(frame_dict['transform_matrix']))

    for j, (key, pts) in enumerate(tag_pts_dict.items()):
      uv, z = project_2d(pts.reshape(4, 1, 1, 4), p_mat, v_mat)

      uv = uv.reshape(-1, 3)
      for i in range(uv.shape[0]):
        img = cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), radius=1, thickness=5, color=(int(cmap[j][2] * 255), int(cmap[j][1] * 255), int(cmap[j][0] * 255)))
    
    # # DEBUG: show one
    # fig = plt.figure()
    # plt.imshow(img[..., ::-1])
    # plt.show()
    # exit()
    dictionary, fname = os.path.split(frame_dict['file_path'])
    cv2.imwrite(os.path.join(data_dir, dictionary, 'proj_' + fname), img)

if __name__ == '__main__':
  main()
