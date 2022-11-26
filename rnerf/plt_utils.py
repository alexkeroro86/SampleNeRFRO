from itertools import combinations, product
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

from rnerf import math_utils


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def plot_cube(ax, r):
  """
  Args:
    r: list[int, int]
  """
  for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
      ax.plot(*zip(s, e), color='r')

def plot_path(ray_pos, idx_grad=None, out_dir=None):
  nmax = np.max(ray_pos.reshape(-1, 3), axis=0)
  nmin = np.min(ray_pos.reshape(-1, 3), axis=0)
  center = np.mean(ray_pos.reshape(-1, 3), axis=0)
  side = np.max(nmax - nmin)
  scale = side / 100 * 10

  sz = int(np.math.ceil(np.sqrt(ray_pos.shape[0])))
  u, v = np.meshgrid(np.arange(sz + 1)[1:], np.arange(sz + 1)[1:], indexing='xy')
  color = np.stack([u, v, np.zeros_like(u)], axis=-1).reshape(-1, 3) / sz

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(projection='3d', computed_zorder=False)

  # text
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  # position
  ax.scatter(ray_pos[0, :, 0:1], ray_pos[0, :, 1:2], ray_pos[0, :, 2:3],
             facecolors=np.tile(np.array([[255, 255, 255]]) / 255.0, [ray_pos.shape[1], 1]),
             edgecolors=np.tile(np.array([[139, 206, 151]]) / 255.0, [ray_pos.shape[1], 1]), s=50, depthshade=True, zorder=4.4)
  ax.plot(ray_pos[0, :, 0], ray_pos[0, :, 1], np.ones_like(ray_pos[0, :, 2]) * (center[2] - side * 0.5),
          color="#8bce97")
  for i in list(range(0, ray_pos.shape[1], 16)) + [-1]:
    ax.plot([ray_pos[0, i, 0], ray_pos[0, i, 0]],
            [ray_pos[0, i, 1], ray_pos[0, i, 1]],
            [ray_pos[0, i, 2], center[2] - side * 0.5], 'k:')
  
  # direction
  if idx_grad is not None:
    idx_grad = math_utils.safe_l2_normalize(idx_grad) * scale
    ax.quiver(ray_pos[0, :, 0:1], ray_pos[0, :, 1:2], ray_pos[0, :, 2:3],
              idx_grad[0, :, 0:1] * scale, idx_grad[0, :, 1:2] * scale, idx_grad[0, :, 2:3] * scale,
              color='r')

  # equal aspect ratio
  ax.set_xlim(center[0] - side * 0.5, center[0] + side * 0.5)
  ax.set_ylim(center[1] - side * 0.5, center[1] + side * 0.5)
  ax.set_zlim(center[2] - side * 0.5, center[2] + side * 0.5)
  ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

  # style
  ax.grid(False)
  ax.xaxis.pane.set_edgecolor('black')
  ax.yaxis.pane.set_edgecolor('black')
  ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.5))
  ax.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.9))
  ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.5))
  ax.view_init(elev=20, azim=145)

  plt.tight_layout()

  if out_dir is not None:
    for name, elev, azim in zip(['top', 'right', 'front', 'free'], [90., 0., 0., 30.], [0., 0., 90., -60.]):
      ax.view_init(elev=elev, azim=azim)
      plt.draw()
      imageio.imwrite(f'{out_dir}/{name}.png', get_img_from_fig(fig, dpi=180))

  plt.show()
  plt.close(fig)
