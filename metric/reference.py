import json
import os
import imageio
import cv2


def main():
  out_dir = '/home/cgv/logs/refractive-nerf-jax/ship_skydome-bkgd_no-partial-reflect_cycles/ablation - radiance (wo-empty)/test_preds'
  json_fname = '/home/cgv/data/synthetic/nerf/ship_skydome-bkgd_no-partial-reflect_cycles/transforms_test.json'
  with open(json_fname, 'r') as f:
    data = json.load(f)

  fnames = [os.path.join(os.path.dirname(json_fname), meta['file_path'] + '.png') for meta in data['frames']]
  skip = 20
  fnames = fnames[::skip]
  for i, fn in enumerate(fnames):
    img = imageio.imread(fn)
    h, w = img.shape[:2]

    # dH = int(h//2 * 0.5)
    # dW = int(w//2 * 0.5)
    # img = img[(h//2 - dH):(h//2 + dH - 1), (w//2 - dW):(w//2 + dW - 1)]

    img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    
    imageio.imsave(os.path.join(out_dir, f'gt_{i:03d}.png'), img)

if __name__ == '__main__':
  main()
