import os
import json
import numpy as np
from glob import glob
import cv2


def main():
  pred_root = 'C:/Users/User/Meetings/CGVlab/20220324/nthu-glass/mipnerfplus/test_preds'
  test_root = 'C:/Users/User/Workspaces/Python/refractive-nerf-jax/calib/20220322_rs_test'
  test_out_root = os.path.join(test_root, 'test_crops')
  pred_out_root = os.path.join(pred_root, 'pred_crops')
  os.makedirs(test_out_root, exist_ok=True)
  os.makedirs(pred_out_root, exist_ok=True)

  with open(os.path.join(test_root, 'calib.json'), 'r') as f:
    meta = json.load(f)

  # pred_fnames = sorted(glob(os.path.join(pred_root, '???.png')))
  pred_fnames = sorted(glob(os.path.join(pred_root, 'color_*.png')))

  for i, frame in enumerate(meta['frames']):
    dictionary, fname = os.path.split(frame['file_path'])
    test_fname = os.path.join(test_root, fname)
    mask_fname = os.path.join(test_root, 'mask_' + fname[:-3] + 'png')
    pred_fname = pred_fnames[i]

    test_im = cv2.imread(test_fname)
    mask_im = cv2.imread(mask_fname)[..., 0]
    pred_im = cv2.imread(pred_fname)

    h, w = test_im.shape[:2]
    dH = int(h//2 * 0.5)
    dW = int(w//2 * 0.5)
    test_im = test_im[(h//2 - dH):(h//2 + dH - 1), (w//2 - dW):(w//2 + dW - 1)]
    mask_im = mask_im[(h//2 - dH):(h//2 + dH - 1), (w//2 - dW):(w//2 + dW - 1)]

    x, y, w, h = cv2.boundingRect(mask_im)
    test_im = test_im[y:(y+h), x:(x+w)]
    mask_im = mask_im[y:(y+h), x:(x+w)]
    pred_im = pred_im[y:(y+h), x:(x+w)]

    cv2.imwrite(os.path.join(test_out_root, f'{i:03d}.png'), test_im)
    cv2.imwrite(os.path.join(test_out_root, f'mask_{i:03d}.png'), mask_im)
    cv2.imwrite(os.path.join(pred_out_root, f'{i:03d}.png'), pred_im)

if __name__ == '__main__':
  main()
