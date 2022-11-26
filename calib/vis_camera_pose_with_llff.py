import os
import numpy as np
from glob import glob
import cv2
import scipy.spatial.transform as transform
import open3d as o3d
import json

import cfg


# ------ LLFF Dataset Code ------
IMG_CUT_INDEX = 112

def _recenter_poses(poses):
  """Recenter poses according to the original NeRF code."""
  poses_ = poses.copy()
  bottom = np.reshape([0, 0, 0, 1.], [1, 4])
  # TODO: change index
  c2w = _poses_avg(poses[IMG_CUT_INDEX:])
  # c2w = _poses_avg(poses)
  c2w = np.concatenate([c2w[:3, :4], bottom], -2)
  bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
  poses = np.concatenate([poses[:, :3, :4], bottom], -2)
  poses = np.linalg.inv(c2w) @ poses
  poses_[:, :3, :4] = poses[:, :3, :4]
  poses = poses_
  # TODO: change index
  # return poses[:IMG_CUT_INDEX]  # test
  return poses[IMG_CUT_INDEX:]  # train
  # return poses  # all

def _poses_avg(poses):
  """Average poses according to the original NeRF code."""
  hwf = poses[0, :3, -1:]
  center = poses[:, :3, 3].mean(0)
  vec2 = _normalize(poses[:, :3, 2].sum(0))
  up = poses[:, :3, 1].sum(0)
  c2w = np.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
  return c2w

def _viewmatrix(z, up, pos):
  """Construct lookat view matrix."""
  vec2 = _normalize(z)
  vec1_avg = up
  vec0 = _normalize(np.cross(vec1_avg, vec2))
  vec1 = _normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, pos], 1)
  return m

def _normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)

# ------ vis_camera_pose_with_opencv ------
def to_view_matrix(rot_mat, tvec):
  """
  Args:
    - rvec: np.ndarray, [3, 1]
    - tvec: np.ndarray, [3, 1]
  Returns:
    - mat: np.ndarray, [4, 4]
  """
  mat = np.eye(4)
  mat[:3, :3] = rot_mat.T
  mat[:3, 3] = (-rot_mat.T @ tvec).reshape(-1)
  return mat

def to_transform_matrix(extrinsic):
  # Extrinsic: [px, py, pz, qx, qy, qz, qw]
  C, q = extrinsic[:3], extrinsic[3:]
  T_mat = np.eye(4).astype(np.float32)
  T_mat[:3, :3] = transform.Rotation.from_quat(q).as_matrix()
  T_mat[:3, 3] = C
  return T_mat

def to_trans_quat(mat):
  """
  """
  trans = mat[:3, 3]
  quat = transform.Rotation.from_matrix(mat[:3, :3]).as_quat()
  return np.concatenate([trans, quat])

def to_frustum(trans, quat, scale=10):
  # R = [X, Y, Z], where X \belong R^{3x1}
  rot_mat = transform.Rotation.from_quat(quat).as_matrix()
  forward = rot_mat[:3, 2]  # Z
  right = rot_mat[:3, 0]  # X
  up = rot_mat[:3, 1]  # Y

  # Frustum geometry
  center = trans + forward * 0.5 * scale
  tl = center - right * 0.4 * scale + up * 0.3 * scale
  tr = center + right * 0.4 * scale + up * 0.3 * scale
  bl = center - right * 0.4 * scale - up * 0.3 * scale
  br = center + right * 0.4 * scale - up * 0.3 * scale
  points = [tl, tr, bl, br, trans]
  lines = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
  return points, lines

def to_basis(trans, quat, scale=10):
  rot_mat = transform.Rotation.from_quat(quat).as_matrix()
  forward = rot_mat[:3, 2]
  right = rot_mat[:3, 0]
  up = rot_mat[:3, 1]

  # Basis geometry
  points = [trans, trans + forward * scale, trans + right * scale, trans + up * scale]
  lines = [[0, 1], [0, 2], [0, 3]]
  return points, lines

def blender_to_opencv(extrinsic):
  C, q = extrinsic[0:3], extrinsic[3:7]
  ''' World coordinate system (Blender)
  z
  ^ y
  |/
  .-->x
  '''
  # TODO: wineglass video
  # C = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) @ C
  C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ C
  ''' Camera coordinate system (Blender)
  y
  ^ -z
  |/
  .--> x
  '''
  R = transform.Rotation.from_quat(q).as_matrix()
  # TODO: wineglass video
  # R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) @ R @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ R @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
  q = transform.Rotation.from_matrix(R).as_quat()
  return np.concatenate([C, q])

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def main():
  # Load images.
  imgfiles = list(sorted(glob(os.path.join(cfg.root, '*.JPG'))))
  image = cv2.imread(imgfiles[0])

  # Load poses and bds.
  with open(os.path.join(cfg.root, 'poses_bounds.npy'), 'rb') as f:
    poses_arr = np.load(f)#[:131]
  poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
  bds = poses_arr[:, -2:].transpose([1, 0])

  # assert len(imgfiles) == poses.shape[-1]

  # Update poses according to downsampling.
  factor = 1.0
  poses[:2, 4, :] = np.array(image.shape[:2]).reshape([2, 1])  # image height, width
  poses[2, 4, :] = poses[2, 4, :] * 1. / factor  # focal

  # Correct rotation matrix ordering and move variable dim to axis 0.
  poses = np.concatenate(
      [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
  poses = np.moveaxis(poses, -1, 0).astype(np.float32)
  bds = np.moveaxis(bds, -1, 0).astype(np.float32)

  # Rescale according to a default bd factor.
  scale = 1. / (bds.min() * .75)
  poses[:, :3, 3] *= scale
  bds *= scale

  # Recenter poses.
  poses = _recenter_poses(poses)

  # ignore fy, cx, cy
  # https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py#L11
  camtoworlds = poses[:, :3, :4]
  focal = poses[0, -1, -1]
  h, w = image.shape[:2]

  # Visualization.
  json_dict = {}
  json_dict['cam_mat'] = np.array([[focal, 0, w * 0.5],
                                   [0, focal, h * 0.5],
                                   [0, 0, 1]])
  json_dict['frames'] = []
  trans_quat_list = []
  for i in range(len(camtoworlds)):
    mat = to_transform_matrix(blender_to_opencv(to_trans_quat(camtoworlds[i])))
    trans_quat_list.append(to_trans_quat(mat))

    json_dict['frames'].append({
      'file_path': imgfiles[i],
      'transform_matrix': mat,
    })

  with open(os.path.join(cfg.root, 'calib.json'), 'w') as f:
    json.dump(json_dict, f, cls=NumpyEncoder)

  vis = o3d.visualization.Visualizer()
  vis.create_window(width=1280, height=720)
  ctr = vis.get_view_control()
  opt = vis.get_render_option()
  opt.mesh_show_back_face = True

  scale = 0.1
  for i in range(len(trans_quat_list)):
    pts, lines = to_frustum(trans_quat_list[i][0:3], trans_quat_list[i][3:], scale=scale)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    pts, lines = to_basis(trans_quat_list[i][0:3], trans_quat_list[i][3:], scale=scale)
    colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

  origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
  vis.add_geometry(origin)

  # Generate rays
  cam_idx = 0
  for cam_idx in range(len(trans_quat_list)):
    pixel_center = 0.5
    cam_mat = json_dict['cam_mat']
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w // 2, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(h // 2, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")
    camera_dirs = np.stack([(x - cam_mat[0][2]) / cam_mat[0][0],
                            (y - cam_mat[1][2]) / cam_mat[1][1],
                            np.ones_like(x)],
                            axis=-1)
    camtoworld = np.array(json_dict['frames'][cam_idx]['transform_matrix'])

    direction = (camera_dirs[h // 2 - 1, w // 2 - 1] * camtoworld[:3, :3]).sum(axis=-1)
    direction /= np.linalg.norm(direction)
    origin = camtoworld[:3, -1]
    # print(direction.shape, origin.shape)

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(origin)
    # sphere.compute_vertex_normals()
    # vis.add_geometry(sphere)

    # # Near, far plane
    # start = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(origin + cfg.near * direction)
    # start.compute_vertex_normals()
    # start.paint_uniform_color([0.8, 0.3, 0.3])
    # vis.add_geometry(start)

    # end = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(origin + cfg.far * direction)
    # end.compute_vertex_normals()
    # end.paint_uniform_color([0.3, 0.3, 0.8])
    # vis.add_geometry(end)
  
  # Whether visual hull is exist
  obj_fnames = list(sorted(glob(os.path.join(cfg.root, '*.obj'))))
  if len(obj_fnames) > 0:
    print(obj_fnames[0])
    visual_hull_mesh = o3d.io.read_triangle_mesh(obj_fnames[0])
    visual_hull_mesh.compute_vertex_normals()
    visual_hull_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    print(visual_hull_mesh.get_max_bound())
    print(visual_hull_mesh.get_min_bound())
    vis.add_geometry(visual_hull_mesh)

  vis.run()
  vis.destroy_window()

if __name__ == '__main__':
  main()
