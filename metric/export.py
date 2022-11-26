import imageio
from glob import glob
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import json
import cv2
import io


def put_text(img, text, font_color=(0, 0, 0)):
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.7
  font_thickness = 1
  text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
  org = (5, text_size[1] + 5)
  # text_w, text_h = text_size
  # cv2.rectangle(img, (0, 0), (org[0] + text_w, org[1] + text_h), (0, 0, 0), -1)
  cv2.putText(img, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

def export_video():
  DATASET = 'opencv'
  HALF = False
  MASK = False
  CROP = True
  scene = 'glass'
  mode = 'test_preds_video'
  
  # dir
  root_dir_mnerf = f'/home/cgv839/logs/mipnerfplus/{scene}'
  root_dir_rnerf_woBD = f'/home/cgv839/logs/refractive-nerf-jax/{scene}/eikonal_fields'
  # root_dir_rnerf_woH = f'/home/cgv839/logs/refractive-nerf-jax/{scene}/eikonal_fields_mode-1'
  root_dir_rnerf = f'/home/cgv839/logs/refractive-nerf-jax/{scene}/radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_w-mod-bd-0.05_blur-5-3.0_uni384'

  root_dir = '/home/cgv839'
  if DATASET == 'blender':
    data_dir = f'{root_dir}/data/synthetic/nerf'
  elif DATASET == 'opencv':
    data_dir = f'{root_dir}/data/real'

  # file name
  if DATASET == 'blender':
    with open(os.path.join(data_dir, scene, 'transforms_test.json'), 'r') as f:
      data = json.load(f)
    test_fnames = [os.path.join(data_dir, scene, frame['file_path'] + '.png') for frame in data['frames']]
    if MASK or CROP:
      fn = lambda fname, former: os.path.split(fname)[0 if former else 1]
      mask_fnames = [os.path.join(data_dir, scene, fn(frame['file_path'], former=True), 'mask_' + fn(frame['file_path'], former=False) + '.png') for frame in data['frames']]
  elif DATASET == 'opencv':
    with open(os.path.join(data_dir, scene, 'transforms_test.json'), 'r') as f:
      data = json.load(f)
    test_fnames = [os.path.join(data_dir, scene, frame['file_path']) for frame in data['frames']]
    if MASK or CROP:
      fn = lambda fname, former: os.path.split(fname)[0 if former else 1]
      mask_fnames = [os.path.join(data_dir, scene, fn(frame['file_path'], former=True), 'mask_' + fn(frame['file_path'], former=False)[:-3] + 'png') for frame in data['frames']]
  fnames_mnerf = list(sorted(glob(os.path.join(root_dir_mnerf, mode, 'color_*.png'))))
  fnames_rnerf_woBD = list(sorted(glob(os.path.join(root_dir_rnerf_woBD, mode, '???.png'))))
  # fnames_rnerf_woH = list(sorted(glob(os.path.join(root_dir_rnerf_woH, mode, '???.png'))))
  fnames_rnerf = list(sorted(glob(os.path.join(root_dir_rnerf, mode, '???.png'))))

  # save frame
  imgs = []
  out_dir = f'/home/cgv839/result/{scene}_{mode}'
  os.makedirs(out_dir, exist_ok=True)
  for i in tqdm(range(len(test_fnames))):
    # test, mask
    test_im = cv2.imread(test_fnames[i])
    if MASK or CROP:
      mask_im = np.array(imageio.imread(mask_fnames[i])[..., None] / 255.0)
    if HALF:
      if DATASET == 'blender':
        test_im = cv2.resize(test_im, (400, 400), interpolation=cv2.INTER_AREA)
        if MASK or CROP:
          mask_im = cv2.resize(mask_im, (400, 400), interpolation=cv2.INTER_NEAREST)[..., None]
      elif DATASET == 'opencv':  # assume always central crop opencv dataset
        h, w = test_im.shape[:2]
        dH = int(h//2 * 0.5)
        dW = int(w//2 * 0.5)
        test_im = test_im[(h//2 - dH):(h//2 + dH), (w//2 - dW):(w//2 + dW)]
        if MASK or CROP:
          mask_im = mask_im[(h//2 - dH):(h//2 + dH), (w//2 - dW):(w//2 + dW)]
    
    h, w = test_im.shape[:2]

    im_mnerf = cv2.imread(fnames_mnerf[i])
    im_rnerf_woBD = cv2.imread(fnames_rnerf_woBD[i])
    # im_rnerf_woH = cv2.imread(fnames_rnerf_woH[i])
    im_rnerf = cv2.imread(fnames_rnerf[i])
    
    if CROP:
      cx, cy, cw, ch = cv2.boundingRect((mask_im[..., 0] * 255.0).astype(np.uint8))

      # draw rect
      # cv2.rectangle(im_mnerf, (cx, cy), (cx + cw, cy + ch), (116, 123, 247), 2)
      # cv2.rectangle(im_rnerf_woBD, (cx, cy), (cx + cw, cy + ch), (116, 123, 247), 2)
      # cv2.rectangle(im_rnerf_woH, (cx, cy), (cx + cw, cy + ch), (116, 123, 247), 2)
      # cv2.rectangle(im_rnerf, (cx, cy), (cx + cw, cy + ch), (116, 123, 247), 2)

      # blend mask
      mask = np.ones((h, w, 3))
      weight = np.ones((h, w, 3)) * 0.5
      weight[cy:(cy+ch), cx:(cx+cw)] = 1.0
      im_mnerf = np.clip(((im_mnerf / 255.0) * weight + mask * (1 - weight)) * 255.0, 0, 255).astype(np.uint8)
      im_rnerf_woBD = np.clip(((im_rnerf_woBD / 255.0) * weight + mask * (1 - weight)) * 255.0, 0, 255).astype(np.uint8)
      # im_rnerf_woH = np.clip(((im_rnerf_woH / 255.0) * weight + mask * (1 - weight)) * 255.0, 0, 255).astype(np.uint8)
      im_rnerf = np.clip(((im_rnerf / 255.0) * weight + mask * (1 - weight)) * 255.0, 0, 255).astype(np.uint8)

    # video
    put_text(im_mnerf, 'mip-NeRF', (0, 0, 0))
    put_text(im_rnerf_woBD, 'Eikonal Fields', (0, 0, 0))
    # put_text(im_rnerf_woH, 'Eikonal Fields (2nd: IoR)', (0, 0, 0))
    put_text(im_rnerf, 'Ours', (0, 0, 0))
    put_text(test_im, 'Reference', (0, 0, 0))
    
    # cv2.imwrite(os.path.join(out_dir, f'{i:03d}.png'), np.hstack([
    #   # np.clip(np.tile(mask_im, [1, 1, 3]) * 255.0, 0, 255).astype(np.uint8), np.ones((h, 5, 3), dtype=np.uint8) * 255,
    #   im_mnerf, np.ones((h, 5, 3), dtype=np.uint8) * 255,
    #   im_rnerf_woBD, np.ones((h, 5, 3), dtype=np.uint8) * 255,
    #   im_rnerf_woH, np.ones((h, 5, 3), dtype=np.uint8) * 255,
    #   im_rnerf, np.ones((h, 5, 3), dtype=np.uint8) * 255,
    #   test_im,
    # ]))
    # video
    imgs.append(np.hstack([
      # np.clip(np.tile(mask_im, [1, 1, 3]) * 255.0, 0, 255).astype(np.uint8), np.ones((h, 5, 3), dtype=np.uint8) * 255,
      im_mnerf, np.ones((h, 5, 3), dtype=np.uint8) * 255,
      im_rnerf_woBD, np.ones((h, 5, 3), dtype=np.uint8) * 255,
      # im_rnerf_woH, np.ones((h, 5, 3), dtype=np.uint8) * 255,
      im_rnerf, np.ones((h, 5, 3), dtype=np.uint8) * 255,
      test_im,
    ])[..., ::-1])
  # video
  imageio.mimwrite(os.path.join(out_dir, f'/home/cgv839/result/{scene}_{mode}.mp4'), imgs, fps=15)

def get_img_from_fig(fig, dpi=180):
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=dpi)
  buf.seek(0)
  img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  buf.close()
  img = cv2.imdecode(img_arr, 1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img

def export_sample():
  fpath = f'/home/cgv839/logs/refractive-nerf-jax/star-lamp_skydome-bkgd_no-partial-reflect_cycles/radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128/debug/ray_064_210_244.pkl'
  with open(fpath, 'rb') as f:
    data = pickle.load(f)

  nmax = np.max(data['ray_pos_c'].reshape(-1, 3), axis=0)
  nmin = np.min(data['ray_pos_c'].reshape(-1, 3), axis=0)
  center = np.mean(data['ray_pos_c'].reshape(-1, 3), axis=0)
  side = np.max(nmax - nmin)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(projection='3d', computed_zorder=False)

  # text
  # plt.title('Path Sampling')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  # ax.text(data['ray_pos_c'][0, 0, 0], data['ray_pos_c'][0, 0, 1], data['ray_pos_c'][0, 0, 2] + 0.05, 'start', None)
  # ax.text(data['ray_pos_c'][0, -1, 0], data['ray_pos_c'][0, -1, 1], data['ray_pos_c'][0, -1, 2] + 0.05, 'end', None)

  # plot
  ax.scatter(data['ray_pos_c'][0, :, 0:1], data['ray_pos_c'][0, :, 1:2], data['ray_pos_c'][0, :, 2:3],
             facecolors=np.tile(np.array([[255, 255, 255]]) / 255.0, [data['ray_pos_c'].shape[1], 1]),
             edgecolors=np.tile(np.array([[139, 206, 151]]) / 255.0, [data['ray_pos_c'].shape[1], 1]), s=50, depthshade=True, zorder=4.4)
  ax.plot(data['ray_pos_c'][0, :, 0], data['ray_pos_c'][0, :, 1], np.ones_like(data['ray_pos_c'][0, :, 2]) * (center[2] - side * 0.5),
          color="#8bce97")
  for i in list(range(0, data['ray_pos_c'].shape[1], 16)) + [-1]:
    ax.plot([data['ray_pos_c'][0, i, 0], data['ray_pos_c'][0, i, 0]],
            [data['ray_pos_c'][0, i, 1], data['ray_pos_c'][0, i, 1]],
            [data['ray_pos_c'][0, i, 2], center[2] - side * 0.5], 'k:')
  ax.scatter(data['ray_pos'][0, :, 0:1], data['ray_pos'][0, :, 1:2], data['ray_pos'][0, :, 2:3],
             facecolors=np.tile(np.array([[208, 156, 211]]) / 255.0, [data['ray_pos'].shape[1], 1]),
             edgecolors=np.tile(np.array([[0, 0, 0]]) / 255.0, [data['ray_pos'].shape[1], 1]), s=50, depthshade=True, zorder=4.5)
  ax.plot(data['ray_pos'][0, :, 0], data['ray_pos'][0, :, 1], np.ones_like(data['ray_pos'][0, :, 2]) * (center[2] - side * 0.5),
          color="#d09cd3")

  # equal aspect ratio
  ax.set_xlim(center[0] - side * 0.5, center[0] + side * 0.5)
  ax.set_ylim(center[1] - side * 0.5, center[1] + side * 0.5)
  ax.set_zlim(center[2] - side * 0.5, center[2] + side * 0.5)
  ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

  # style
  # ax.xaxis.set_ticklabels([])
  # ax.yaxis.set_ticklabels([])
  # ax.zaxis.set_ticklabels([])

  # for line in ax.xaxis.get_ticklines():
  #   line.set_visible(False)
  # for line in ax.yaxis.get_ticklines():
  #   line.set_visible(False)
  # for line in ax.zaxis.get_ticklines():
  #   line.set_visible(False)
      
  ax.grid(False)
  ax.xaxis.pane.set_edgecolor('black')
  ax.yaxis.pane.set_edgecolor('black')
  ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.5))
  ax.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.9))
  ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.5))
  # ax.xaxis.pane.fill = False
  # ax.yaxis.pane.fill = False
  # ax.zaxis.pane.fill = False
  ax.view_init(elev=20, azim=145)

  plt.tight_layout()

  # output
  # for name, elev, azim in zip(['top', 'right', 'front', 'free'], [90., 0., 0., 30.], [0., 0., 90., -60.]):
  #   ax.view_init(elev=elev, azim=azim)
  #   plt.draw()
  #   imageio.imwrite(f'{out_dir}/{name}.png', plt_utils.get_img_from_fig(fig, dpi=180))
  plt.draw()
  imageio.imwrite(os.path.join(os.path.dirname(fpath), 'Figure_1.png'), get_img_from_fig(fig, dpi=150))
  plt.show()

if __name__ == '__main__':
  export_video()
