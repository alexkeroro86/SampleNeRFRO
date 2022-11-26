import os
import imageio
import torch
import lpips
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import json
# from skimage.metrics import structural_similarity as ssim_fn

import flip
import ssim


# ------ Util ------

def put_text(img, text, font_color=(0, 0, 0)):
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.7
  font_thickness = 1
  text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
  org = (5, text_size[1] + 5)
  # text_w, text_h = text_size
  # cv2.rectangle(img, (0, 0), (org[0] + text_w, org[1] + text_h), (0, 0, 0), -1)
  cv2.putText(img, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# ------ File I/O ------

def load_img(fpath, white_bkgd=False):
  """[H, W, C], [0, 1]"""
  img = imageio.imread(fpath)
  if white_bkgd:
      img = img[..., :3] * (img[..., -1:] / 255.0) + (255 - img[..., -1:])
  else:
      img = img[..., :3]
  img = np.array(img, dtype=np.float32) / 255.0
  return img

def save_img(fpath, img):
  imageio.imsave(fpath, np.clip(255.0 * img, 0, 255).astype(np.uint8))

def save_err(fpath, img):
  img = flip.CHWtoHWC(flip.index2color(np.clip(255.0 * img, 0.0, 255.0), flip.get_magma_map()))
  imageio.imsave(fpath, np.clip(255.0 * img, 0, 255).astype(np.uint8))

# ------ Metric ------

def compute_psnr(ref, src):
  """[B, C, H, W], [0, 1]"""
  mse = torch.mean((ref - src)**2)
  err = -20 * torch.log10(torch.sqrt(mse))
  err = err.item()
  return err, torch.mean((ref[0] - src[0])**2, axis=0).cpu().numpy()

def compute_ssim(ssim_model, ref, src):
  """[B, C, H, W], [0, 1]"""
  err, emap = ssim_model(ref, src)
  err = err.item()
  emap = torch.clip(emap, 0.0, 1.0).squeeze().cpu().numpy()
  return err, emap

def compute_lpips(lpips_model0, lpips_model1, ref, src):
  """[B, C, H, W], [0, 1]"""
  err = lpips_model0(ref, src, normalize=True).item()
  emap = lpips_model1(ref, src, normalize=True)
  emap = torch.clip(emap, 0.0, 1.0).squeeze().cpu().numpy()
  return err, emap

def compute_flip(ref, src):
  """[H, W, C], [0, 1]"""
  monitor_distance = 0.3
  monitor_width = 0.5
  monitor_resolution_x = 400
  pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

  emap = flip.compute_ldrflip(flip.HWCtoCHW(ref), flip.HWCtoCHW(src), pixels_per_degree)[0]
  return np.mean(emap), emap

# ------ Main ------

def main():
  # Config
  WHITE_BKGD = False
  TEST = True
  NSVF = False # whether using NSVF dataset
  MIP = True  # whether using mip-nerf [ICCV 2021]
  DATASET = 'blender'  # dataset format
  METHOD = 'radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_bbox-1.0'  # the stage of ours training strategy
  HALF = True
  MASK = False  # whether apply mask before evaluation
  CROP = False  # whether crop mask bounding box before evaluation
  NAME = 'mipnerfplus'
  SCENE_LIST = [
    'torus_skydome-bkgd_no-partial-reflect_cycles',
    # 'ship_skydome-bkgd_no-partial-reflect_cycles',
    # 'star-lamp_skydome-bkgd_no-partial-reflect_cycles',
    # 'deer-globe_skydome-bkgd_no-partial-reflect_cycles',
    # 'heart',
    # 'yellow',
    # 'dolphin',
    # 'ball',
    # 'glass',
    # 'pen',
    # 'torus_skydome-bkgd_cycles'
  ]

  root_dir = '/home/cgv839'
  log_dir = f'{root_dir}/logs/{NAME}'
  if DATASET == 'blender':
    data_dir = f'{root_dir}/data/synthetic/{"nsvf" if NSVF else "nerf"}'
  elif DATASET == 'opencv':
    data_dir = f'{root_dir}/data/real'

  ssim_model = ssim.SSIM(data_range=1.0).cuda()
  lpips_model0 = lpips.LPIPS(net='alex', verbose=False).cuda()
  lpips_model1 = lpips.LPIPS(net='alex', spatial=True, verbose=False).cuda()

  for dirname in os.listdir(log_dir):
    if not os.path.isdir(os.path.join(log_dir, dirname)):
      continue
    if dirname not in SCENE_LIST:
      continue

    # Get files
    if TEST:
      if NSVF:  # if dataset is provided by NSVF [NIPS 2020]
        test_fnames = sorted(glob(os.path.join(data_dir, dirname, 'rgb', '2_*.png')))
      else:
        if DATASET == 'blender':
          with open(os.path.join(data_dir, dirname, 'transforms_test.json'), 'r') as f:
            data = json.load(f)
          test_fnames = [os.path.join(data_dir, dirname, frame['file_path'] + '.png') for frame in data['frames']]
          if MASK or CROP:
            fn = lambda fname, former: os.path.split(fname)[0 if former else 1]
            mask_fnames = [os.path.join(data_dir, dirname, fn(frame['file_path'], former=True), 'mask_' + fn(frame['file_path'], former=False) + '.png') for frame in data['frames']]
        elif DATASET == 'opencv':
          with open(os.path.join(data_dir, dirname, 'transforms_test.json'), 'r') as f:
            data = json.load(f)
          test_fnames = [os.path.join(data_dir, dirname, frame['file_path']) for frame in data['frames']]
          if MASK or CROP:
            fn = lambda fname, former: os.path.split(fname)[0 if former else 1]
            mask_fnames = [os.path.join(data_dir, dirname, fn(frame['file_path'], former=True), 'mask_' + fn(frame['file_path'], former=False)[:-3] + 'png') for frame in data['frames']]
      if MIP:  # if model is mip-nerf [ICCV 2021]
        pred_fnames = sorted(glob(os.path.join(log_dir, dirname, 'test_preds_default', 'color_*.png')))
      else:
        pred_fnames = sorted(glob(os.path.join(log_dir, dirname, METHOD, 'test_preds_default', '???.png')))
    else:
      if NSVF:  # if dataset is provided by NSVF [NIPS 2020]
        test_fnames = sorted(glob(os.path.join(data_dir, dirname, 'rgb', '0_*.png')))
      else:
        if DATASET == 'blender':
          test_fnames = sorted(glob(os.path.join(data_dir, dirname, 'train', '*.png')), key=lambda s: int(s.split('.')[0].split('_')[-1]))
        elif DATASET == 'opencv':
          with open(os.path.join(data_dir, dirname, 'transforms_train.json'), 'r') as f:
            data = json.load(f)
          test_fnames = [os.path.join(data_dir, dirname, frame['file_path']) for frame in data['frames']]
      if MIP:  # if model is mip-nerf [ICCV 2021]
        pred_fnames = sorted(glob(os.path.join(log_dir, dirname, 'train_preds', 'color_*.png')))
      else:
        pred_fnames = sorted(glob(os.path.join(log_dir, dirname, METHOD, 'train_preds', '???.png')))
    print(dirname, len(test_fnames), len(pred_fnames))
    assert len(test_fnames) == len(pred_fnames)

    suffix = "_mask" if MASK else "" + "_crop" if CROP else ""
    out_errmap_dir = os.path.join(os.path.dirname(pred_fnames[0]), 'errmap' + suffix)
    os.makedirs(out_errmap_dir, exist_ok=True)
    out_frame_dir = os.path.join(out_errmap_dir, 'frame' + suffix)
    os.makedirs(out_frame_dir, exist_ok=True)
    
    # Compute metric
    psnr_list, ssim_list, lpips_list, flip_list = [], [], [], []
    out_str = ''
    imgs = []
    i = 0
    for test_fname, pred_fname in tqdm(zip(test_fnames, pred_fnames), total=len(test_fnames)):
      if MASK or CROP:
        mask_im = np.array(imageio.imread(mask_fnames[i])[..., None] / 255.0)
        if mask_im.shape[2] > 1:
          mask_im = mask_im[:, :, :1]

      test_im = load_img(test_fname, white_bkgd=WHITE_BKGD)
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
      pred_im = load_img(pred_fname)

      if MASK:
        test_im *= mask_im
        pred_im *= mask_im

      if CROP:
        x, y, w, h = cv2.boundingRect((mask_im[..., 0] * 255.0).astype(np.uint8))
        test_im = test_im[y:(y+h), x:(x+w)]
        mask_im = mask_im[y:(y+h), x:(x+w)]
        pred_im = pred_im[y:(y+h), x:(x+w)]

      test_batch = torch.FloatTensor(test_im).permute(2, 0, 1)[None].cuda()
      pred_batch = torch.FloatTensor(pred_im).permute(2, 0, 1)[None].cuda()

      # Compute metric
      with torch.no_grad():
        psnr_val, psnr_map = compute_psnr(test_batch, pred_batch)
        ssim_val, ssim_map = compute_ssim(ssim_model, test_batch, pred_batch)
        lpips_val, lpips_map = compute_lpips(lpips_model0, lpips_model1, test_batch, pred_batch)
      flip_val, flip_map = compute_flip(test_im, pred_im)

      # Save result
      h, w = test_im.shape[:2]
      psnr_list.append(psnr_val)
      ssim_list.append(ssim_val)
      lpips_list.append(lpips_val)
      flip_list.append(flip_val)
      save_err(os.path.join(out_errmap_dir, f'psnr_{i:03d}.png'), psnr_map)
      save_err(os.path.join(out_errmap_dir, f'ssim_{i:03d}.png'), ssim_map)
      save_err(os.path.join(out_errmap_dir, f'lpips_{i:03d}.png'), lpips_map)
      save_err(os.path.join(out_errmap_dir, f'flip_{i:03d}.png'), flip_map)

      # put_text(test_im, 'reference')
      # put_text(pred_im, NAME)

      merge = [test_im, np.ones((h, 5, 3)), pred_im, np.ones((h, 5, 3))]
      for name in ['psnr', 'ssim', 'lpips', 'flip']:
        im = load_img(os.path.join(out_errmap_dir, f'{name}_{i:03d}.png'))
        pad_im = np.zeros((h, w, 3))
        pad_im[:im.shape[0], :im.shape[1]] = im
        # put_text(pad_im, name, (255, 255, 255))
        merge.append(pad_im)
        merge.append(np.ones((h, 5, 3)))
      merge = np.hstack(merge)
      save_img(os.path.join(out_frame_dir, f'frame_{i:03d}.png'), merge)
      imgs.append(merge)
      
      out_str += f'{i:3d}{psnr_val:6.2f}{ssim_val:6.3f}{lpips_val:6.3f}{flip_val:6.3f}\n'
      i += 1
    
    with open(os.path.join(os.path.dirname(pred_fnames[0]), f'metric_list{suffix}.txt'), 'w') as f:
      f.write(out_str)
    
    with open(os.path.join(os.path.dirname(pred_fnames[0]), f'result{suffix}.txt'), 'w') as f:
      f.write(f'{np.mean(psnr_list):6.2f}{np.mean(ssim_list):6.3f}{np.mean(lpips_list):6.3f}{np.mean(flip_list):6.3f}\n')

    # if not CROP:
    #   imageio.mimwrite(os.path.join(log_dir, dirname, f'summary_{METHOD}{suffix}.mp4'), imgs)

if __name__ == '__main__':
  main()
