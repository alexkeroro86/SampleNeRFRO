"""
Convert from OpenCV to Blender format dataset, and visualize it.
"""
import numpy as np
import scipy.spatial.transform as transform
import pickle
from glob import glob
import open3d as o3d
import json
import os

import cfg


def to_view_matrix(rvec, tvec):
  """
  Args:
    - rvec: np.ndarray, [3, 1]
    - tvec: np.ndarray, [3, 1]
  Returns:
    - mat: np.ndarray, [4, 4]
  """
  rot_mat = transform.Rotation.from_rotvec(rvec[:, 0]).as_matrix()
  mat = np.eye(4)
  mat[:3, :3] = rot_mat.T
  mat[:3, 3] = (-rot_mat.T @ tvec).reshape(-1)
  return mat

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

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def main():
  with open(os.path.join(cfg.root, 'calib.pkl'), 'rb') as f:
    calib = pickle.load(f)

  print(calib.keys())
  fnames = calib['fnames']
  rvecs = calib['rvecs']
  tvecs = calib['tvecs']

  # Convert cam->world to world->cam matrix, and save Blender format JSON
  trans_quat_list = []
  json_dict = {}
  json_dict['cam_mat'] = calib['cameraMatrix']
  json_dict['frames'] = []
  for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    mat = to_view_matrix(rvec, tvec)
    trans_quat_list.append(to_trans_quat(mat))

    json_dict['frames'].append({
      'file_path': fnames[i],
      'transform_matrix': mat,
    })

  with open(os.path.join(cfg.root, 'calib.json'), 'w') as f:
    json.dump(json_dict, f, cls=NumpyEncoder)

  # Visualize camera pose
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
  w, h = 2560//2, 1920//2
  cam_idx = 0
  for cam_idx in range(len(trans_quat_list)):
    pixel_center = 0.5
    cam_mat = calib['cameraMatrix']
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")
    camera_dirs = np.stack([(x - cam_mat[0][2]) / cam_mat[0][0],
                            (y - cam_mat[1][2]) / cam_mat[1][1],
                            np.ones_like(x)],
                            axis=-1)
    camtoworld = np.array(json_dict['frames'][cam_idx]['transform_matrix'])

    direction = (camera_dirs[h//2-1, w//2-1] * camtoworld[:3, :3]).sum(axis=-1)
    direction /= np.linalg.norm(direction)
    origin = camtoworld[:3, -1]
    # print(direction.shape, origin.shape)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(origin)
    sphere.compute_vertex_normals()
    vis.add_geometry(sphere)

    # Near, far plane
    start = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(origin + cfg.near * direction)
    start.compute_vertex_normals()
    start.paint_uniform_color([0.8, 0.3, 0.3])
    vis.add_geometry(start)

    end = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(origin + cfg.far * direction)
    end.compute_vertex_normals()
    end.paint_uniform_color([0.3, 0.3, 0.8])
    vis.add_geometry(end)

  # Whether visual hull is exist
  obj_fnames = glob(os.path.join(cfg.root, '*.obj'))
  if len(obj_fnames) > 0:
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
