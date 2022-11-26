"""
Package: pupil-apriltags
Reference: https://developer.ridgerun.com/wiki/index.php?title=Full_Body_Pose_Estimation_for_Sports_Analysis_-_AprilTag_Camera_Calibration
"""
import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from glob import glob
from pupil_apriltags import Detector

import cfg


def resize_images():
  fnames = glob(os.path.join(cfg.root, '*.jpg'))
  for fname in fnames:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(fname, img)

def create_apriltag():
  tag_id = 0
  for k in range(12):
    bit = 9
    scale = 20
    row = 3
    col = 3
    face = np.zeros((row * bit, col * bit, 3), dtype=np.uint8)
    for i in range(row):
      for j in range(col):
        tag = cv2.imread(os.path.join(cfg.root, f'tag41_12_{tag_id:05d}.png'))
        face[(j * bit):((j + 1) * bit), (i * bit):((i + 1) * bit)] = tag
        tag_id += 1
    face = cv2.resize(face, (9 * bit * scale, 9 * bit * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(cfg.root, f'face_{k:05d}.png'), face)

def main():
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
          [x + scale * 0.7, y + scale * 0.2, anchor_z],
          [x + scale * 0.7, y + scale * 0.7, anchor_z],
          [x + scale * 0.2, y + scale * 0.7, anchor_z],
          [x + scale * 0.2, y + scale * 0.2, anchor_z]], dtype=np.float32)
  # Face 8
  anchor_x = scale * 5.4
  anchor_y, anchor_z = 2.7, 2.7
  k = 8
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      y = scale * anchor_y + scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [anchor_x, y + scale * 0.2, z - scale * 0.7],
        [anchor_x, y + scale * 0.7, z - scale * 0.7],
        [anchor_x, y + scale * 0.7, z - scale * 0.2],
        [anchor_x, y + scale * 0.2, z - scale * 0.2]], dtype=np.float32)
  # Face 10
  anchor_x = scale * 2.7
  anchor_y, anchor_z = 5.4, 2.7
  k = 10
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      y = scale * anchor_y - scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [anchor_x, y - scale * 0.2, z - scale * 0.7],
        [anchor_x, y - scale * 0.7, z - scale * 0.7],
        [anchor_x, y - scale * 0.7, z - scale * 0.2],
        [anchor_x, y - scale * 0.2, z - scale * 0.2]], dtype=np.float32)
  # Face 9
  anchor_y = scale * 5.4
  anchor_z, anchor_x = 2.7, 5.4
  k = 9
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      x = scale * anchor_x - scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [x - scale * 0.2, anchor_y, z - scale * 0.7],
        [x - scale * 0.7, anchor_y, z - scale * 0.7],
        [x - scale * 0.7, anchor_y, z - scale * 0.2],
        [x - scale * 0.2, anchor_y, z - scale * 0.2]], dtype=np.float32)
  # Face 11
  anchor_y = scale * 2.7
  anchor_z, anchor_x = 2.7, 2.7
  k = 11
  for i in range(3):
    for j in range(3):
      z = scale * anchor_z - scale * 0.9 * j
      x = scale * anchor_x + scale * 0.9 * i
      tag_pts_dict[f'{j + i * 3 + 9 * k}'] = np.array([
        [x + scale * 0.2, anchor_y, z - scale * 0.7],
        [x + scale * 0.7, anchor_y, z - scale * 0.7],
        [x + scale * 0.7, anchor_y, z - scale * 0.2],
        [x + scale * 0.2, anchor_y, z - scale * 0.2]], dtype=np.float32)

  # fig = plt.figure(figsize=(12, 8))
  # ax = fig.add_subplot(projection='3d')
  # for key, pts in tag_pts_dict.items():
  #   ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
  # plt.show()
  # exit()

  at_detector = Detector(
    families='tagStandard41h12',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0)

  fnames = list(sorted(glob(os.path.join(cfg.root, '*.jpg'))))
  image_size = None
  uvs_all = []
  pts_all = []
  for fname in fnames:
    # Open the image
    img = cv2.imread(fname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find apriltag markers in the query image
    tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

    # Detect single view corners
    uvs_frame = []
    pts_frame = []
    for tag in tags:
      uvs_frame.append(tag.corners.astype(np.float32))
      pts_frame.append(tag_pts_dict[f'{tag.tag_id}'])

      cv2.circle(img, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)
      cv2.circle(img, tuple(tag.corners[1].astype(int)), 4, (0, 255, 0), 2)
      cv2.circle(img, tuple(tag.corners[2].astype(int)), 4, (0, 0, 255), 2)
      cv2.circle(img, tuple(tag.corners[3].astype(int)), 4, (255, 0, 255), 2)

    if len(uvs_frame) < 3:
      print(f'Number of corners detected in "{fname}" is less than 6!')
      continue

    # Collect multi-view cameras
    uvs_all.append(np.concatenate(uvs_frame, axis=0))
    pts_all.append(np.concatenate(pts_frame, axis=0))

    # If our image size is unknown, set it now
    if not image_size:
      image_size = gray.shape[::-1]

    # Reproportion the image, maxing width or height at 1000
    proportion = max(img.shape) / 1000.0
    img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

    # Pause to display each image, waiting for key press
    cv2.imshow('AprilTag', img)
    cv2.waitKey(0)

  # Destroy any open CV windows
  cv2.destroyAllWindows()

  # Make sure at least one image was found
  if len(fnames) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images were found. Add images and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

  # Make sure we were able to calibrate on at least one apriltag by checking
  # if we ever determined the image size
  if not image_size:
    # Calibration failed because we didn't see any apriltags of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect apriltags in any of the images supplied. Try different pictures of apriltags.")
    # Exit for failure
    exit()

  # Now that we've seen all of our images, perform the camera calibration
  # based on the set of points we've discovered
  K = np.array([[1.2 * max(image_size[0], image_size[1]), 0, image_size[0]/2],
                [0, 1.2 * max(image_size[0], image_size[1]), image_size[1]/2],
                [0, 0, 1]], dtype=np.float32)
  calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=pts_all, imagePoints=uvs_all,
    imageSize=image_size,
    cameraMatrix=K,
    distCoeffs=np.zeros((5,), dtype=np.float32),
    flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_INTRINSIC_GUESS)
  
  # Print matrix and distortion coefficient to the console
  print(cameraMatrix)
  print(distCoeffs)

  # Save values to be used where matrix+dist is required, for instance for posture estimation
  with open(os.path.join(os.path.split(fnames[0])[0], 'calib.pkl'), 'wb') as f:
    pickle.dump({
      'fnames': fnames,
      'cameraMatrix': cameraMatrix,
      'distCoeffs': distCoeffs,
      'rvecs': rvecs,
      'tvecs': tvecs,
    }, f)

if __name__ == '__main__':
  # resize_images()
  # create_apriltag()
  main()
