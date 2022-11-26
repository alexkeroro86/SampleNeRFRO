import os
import imageio
import torch
import lpips
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import json

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

def get_fnames(data_dir, SCENE, log_dir, TEST, NSVF, DATASET, MASK, MIP, STAGE, CROP):
  mask_fnames = []
  if TEST:
    if NSVF:  # if dataset is provided by NSVF [NIPS 2020]
      test_fnames = sorted(glob(os.path.join(data_dir, SCENE, 'rgb', '2_*.png')))
    else:
      if DATASET == 'blender':
        with open(os.path.join(data_dir, SCENE, 'transforms_test.json'), 'r') as f:
          data = json.load(f)
        test_fnames = [os.path.join(data_dir, SCENE, frame['file_path'] + '.png') for frame in data['frames']]
        if MASK or CROP:
          fn = lambda fname, former: os.path.split(fname)[0 if former else 1]
          mask_fnames = [os.path.join(data_dir, SCENE, fn(frame['file_path'], former=True), 'mask_' + fn(frame['file_path'], former=False) + '.png') for frame in data['frames']]
      elif DATASET == 'opencv':
        with open(os.path.join(data_dir, SCENE, 'transforms_test.json'), 'r') as f:
          data = json.load(f)
        test_fnames = [os.path.join(data_dir, SCENE, frame['file_path']) for frame in data['frames']]
        if MASK or CROP:
          fn = lambda fname, former: os.path.split(fname)[0 if former else 1]
          mask_fnames = [os.path.join(data_dir, SCENE, fn(frame['file_path'], former=True), 'mask_' + fn(frame['file_path'], former=False)[:-3] + 'png') for frame in data['frames']]
    if MIP:  # if model is mip-nerf [ICCV 2021]
      pred_fnames = sorted(glob(os.path.join(log_dir, SCENE, 'test_preds', 'color_*.png')))
    else:
      pred_fnames = sorted(glob(os.path.join(log_dir, SCENE, STAGE, 'test_preds', '???.png')))
  else:
    if NSVF:  # if dataset is provided by NSVF [NIPS 2020]
      test_fnames = sorted(glob(os.path.join(data_dir, SCENE, 'rgb', '0_*.png')))
    else:
      if DATASET == 'blender':
        test_fnames = sorted(glob(os.path.join(data_dir, SCENE, 'train', '*.png')), key=lambda s: int(s.split('.')[0].split('_')[-1]))
      elif DATASET == 'opencv':
        with open(os.path.join(data_dir, SCENE, 'transforms_train.json'), 'r') as f:
          data = json.load(f)
        test_fnames = [os.path.join(data_dir, SCENE, frame['file_path']) for frame in data['frames']]
    if MIP:  # if model is mip-nerf [ICCV 2021]
      pred_fnames = sorted(glob(os.path.join(log_dir, SCENE, 'train_preds', 'color_*.png')))
    else:
      pred_fnames = sorted(glob(os.path.join(log_dir, SCENE, STAGE, 'train_preds', '???.png')))
  print(SCENE, len(test_fnames), len(pred_fnames))
  assert len(test_fnames) == len(pred_fnames)
  return pred_fnames, mask_fnames, test_fnames

# ------ Metric ------

def compute_psnr(ref, src):
  """[B, C, H, W], [0, 1]"""
  mse = torch.mean((ref - src)**2)
  err = -20 * torch.log10(torch.sqrt(mse))
  err = err.item()
  return err, torch.mean((ref[0] - src[0])**2, axis=0).cpu().numpy()

def compute_dssim(ssim_model, ref, src):
  """[B, C, H, W], [0, 1]"""
  err, emap = ssim_model(ref, src)
  err = err.item()
  emap = torch.clip(emap, 0.0, 1.0).squeeze().cpu().numpy()
  return 1 - err, 1 - emap

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
  MIP = False  # whether using mip-nerf [ICCV 2021]
  DATASET = 'blender'  # dataset format
  SCENE = 'ship_skydome-bkgd_no-partial-reflect_cycles'  # the stage of ours training strategy
  HALF = True
  MASK = False  # whether apply mask before evaluation
  CROP = False  # whether crop mask bounding box before evaluation
  NAME = 'refractive-nerf-jax'
  METHOD1 = 'radiance_01.small-step-size.pe.delay-anneal-reverse-0.25.beta-9-1-0.0075.sparsity-log-boundary-0.0075'
  METHOD2 = 'radiance_01.small-step-size.pe'

  root_dir = '/home/cgv'
  log_dir = f'{root_dir}/logs/{NAME}'
  if DATASET == 'blender':
    data_dir = f'{root_dir}/data/synthetic/{"nsvf" if NSVF else "nerf"}'
  elif DATASET == 'opencv':
    data_dir = f'{root_dir}/data/real'

  # Get files
  pred_fnames1, mask_fnames, test_fnames = get_fnames(
    data_dir, SCENE, log_dir,
    TEST, NSVF, DATASET, MASK, MIP, METHOD1, CROP)
  pred_fnames2, mask_fnames, test_fnames = get_fnames(
    data_dir, SCENE, log_dir,
    TEST, NSVF, DATASET, MASK, MIP, METHOD2, CROP)
  
  suffix = "_mask" if MASK else "" + "_crop" if CROP else ""
  out_dir = os.path.join(os.path.dirname(pred_fnames1[0]), f'compare_{METHOD2}{suffix}')
  os.makedirs(out_dir, exist_ok=True)
  out_frame_dir = os.path.join(out_dir, f'frame{suffix}')
  os.makedirs(out_frame_dir, exist_ok=True)

  # Compute metric
  ssim_model = ssim.SSIM(data_range=1.0).cuda()
  lpips_model0 = lpips.LPIPS(net='alex', verbose=False).cuda()
  lpips_model1 = lpips.LPIPS(net='alex', spatial=True, verbose=False).cuda()
  
  imgs = []
  i = 0
  for test_fname, pred_fname1, pred_fname2 in tqdm(zip(test_fnames, pred_fnames1, pred_fnames2), total=len(test_fnames)):
    if MASK or CROP:
      mask_im = np.array(imageio.imread(mask_fnames[i])[..., None] / 255.0)

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
        test_im = test_im[(h//2 - dH):(h//2 + dH - 1), (w//2 - dW):(w//2 + dW - 1)]
        if MASK or CROP:
          mask_im = mask_im[(h//2 - dH):(h//2 + dH - 1), (w//2 - dW):(w//2 + dW - 1)]
    
    pred_im1 = load_img(pred_fname1)
    pred_im2 = load_img(pred_fname2)

    if MASK:
      test_im *= mask_im
      pred_im1 *= mask_im
      pred_im2 *= mask_im

    if CROP:
      x, y, w, h = cv2.boundingRect((mask_im[..., 0] * 255.0).astype(np.uint8))
      test_im = test_im[y:(y+h), x:(x+w)]
      mask_im = mask_im[y:(y+h), x:(x+w)]
      pred_im1 = pred_im1[y:(y+h), x:(x+w)]
      pred_im2 = pred_im2[y:(y+h), x:(x+w)]

    test_batch = torch.FloatTensor(test_im).permute(2, 0, 1)[None].cuda()
    pred_batch1 = torch.FloatTensor(pred_im1).permute(2, 0, 1)[None].cuda()
    pred_batch2 = torch.FloatTensor(pred_im2).permute(2, 0, 1)[None].cuda()

    with torch.no_grad():
      psnr_val1, psnr_map1 = compute_psnr(test_batch, pred_batch1)
      dssim_val1, dssim_map1 = compute_dssim(ssim_model, test_batch, pred_batch1)
      lpips_val1, lpips_map1 = compute_lpips(lpips_model0, lpips_model1, test_batch, pred_batch1)
    flip_val1, flip_map1 = compute_flip(test_im, pred_im1)

    with torch.no_grad():
      psnr_val2, psnr_map2 = compute_psnr(test_batch, pred_batch2)
      dssim_val2, dssim_map2 = compute_dssim(ssim_model, test_batch, pred_batch2)
      lpips_val2, lpips_map2 = compute_lpips(lpips_model0, lpips_model1, test_batch, pred_batch2)
    flip_val2, flip_map2 = compute_flip(test_im, pred_im2)

    # Save result
    h, w = test_im.shape[:2]
    merge = []
    for name, map1, map2 in zip(['psnr', 'dssim', 'lpips', 'flip'],
                                [psnr_map1, dssim_map1, lpips_map1, flip_map1],
                                [psnr_map2, dssim_map2, lpips_map2, flip_map2]):
      non = np.abs(map1 - map2)[..., None] < 1e-3
      pos = (1 - non) * (map1 <= map2)[..., None]
      neg = (1 - non) * (map1 > map2)[..., None]

      im = np.array([239, 138, 98])[None, None] / 255.0 * pos + \
           np.array([247, 247, 247])[None, None] / 255.0 * non + \
           np.array([103, 169, 207])[None, None] / 255.0 * neg
      save_img(os.path.join(out_dir, f'{name}_{i:03d}.png'), im)

      pad_im = np.ones((h, w, 3))
      pad_im[:im.shape[0], :im.shape[1]] = im
      put_text(pad_im, name, (0, 0, 0))
      merge.append(pad_im)
      merge.append(np.ones((h, 5, 3)))
    merge = np.hstack(merge)
    save_img(os.path.join(out_frame_dir, f'frame_{i:03d}.png'), merge)
    imgs.append(merge)

    i += 1
  
  if not CROP:
    imageio.mimwrite(os.path.join(log_dir, SCENE, f'compare_{METHOD1}{suffix}.mp4'), imgs)

if __name__ == '__main__':
  main()
