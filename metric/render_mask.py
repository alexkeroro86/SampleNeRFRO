import os
import json
import numpy as np
import pyrender
import open3d as o3d
import cv2

# https://github.com/mmatl/pyrender/issues/86
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"


def main():
  # Config
  DATASET = 'opencv'

  data_dir = '/home/cgv839/data/real/glass'

  # Load
  with open(os.path.join(data_dir, 'transforms_test.json'), 'r') as f:
    meta = json.load(f)

  img = cv2.imread(os.path.join(data_dir, meta['frames'][0]['file_path'] + ('.png' if DATASET == "blender" else "")))
  h, w = img.shape[:2]

  mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, 'mesh.obj'))
  mesh.compute_vertex_normals()
  mesh.paint_uniform_color([0.8, 0.8, 0.8])

  # Loop
  for frame in meta['frames']:
    scene = pyrender.Scene()

    # Mesh
    m = pyrender.Mesh(
      primitives=[
        pyrender.Primitive(
          positions=np.asarray(mesh.vertices).astype(np.float32),
          normals=np.asarray(mesh.vertex_normals).astype(np.float32),
          color_0=np.asarray(mesh.vertex_colors).astype(np.float32),
          indices=np.asarray(mesh.triangles).astype(np.int32),
          mode=pyrender.GLTF.TRIANGLES,
        )
      ],
      is_visible=True,
    )
    mesh_node = pyrender.Node(mesh=m, matrix=np.eye(4))
    scene.add_node(mesh_node)

    # Camera
    if DATASET == "blender":
      focal = .5 * w / np.tan(.5 * meta["camera_angle_x"])

      cam = pyrender.IntrinsicsCamera(
        fx=focal,
        fy=focal,
        cx=w//2,
        cy=h//2,
        znear=0.1,
        zfar=10.0,
      )
      T = np.array(frame['transform_matrix'])
    elif DATASET == "opencv":
      cam = pyrender.IntrinsicsCamera(
        fx=meta["cam_mat"][0][0],
        fy=meta["cam_mat"][1][1],
        cx=meta["cam_mat"][0][2],
        cy=meta["cam_mat"][1][2],
        znear=0.1,
        # zfar=1.0,
        zfar=100.0,
      )
      T = np.array(frame['transform_matrix'])
      cv2gl = np.array(
          [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
      )
      T = T @ cv2gl
    cam_node = pyrender.Node(camera=cam, matrix=T)
    scene.add_node(cam_node)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node, parent_node=cam_node)

    render = pyrender.OffscreenRenderer(w, h)
    color, depth = render.render(scene)
    render.delete()

    dictionary, fname = os.path.split(frame['file_path'])
    mask_fname = os.path.join(data_dir, dictionary, 'mask_' + (fname if DATASET == "blender" else fname[:-4]) + '.png')

    mask = np.where(depth != 0, 1, 0).astype(np.uint8) * 255
    kernel = np.ones((35, 35)).astype(np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite(mask_fname, mask)

if __name__ == '__main__':
  main()
