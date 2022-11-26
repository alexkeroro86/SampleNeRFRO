"""
Package: opencv-contrib-python
Reference: https://github.com/kyle-bersani/opencv-examples/blob/master/CalibrationByCharucoBoard/CalibrateCamera.py
"""
import cv2
from cv2 import aruco
from glob import glob
import numpy as np
import pickle
import os

import cfg

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
CHARUCO_BOARD = cv2.aruco.CharucoBoard_create(5, 7, 0.04, 0.02, ARUCO_DICT)


def resize_images():
  fnames = glob(os.path.join(cfg.root, '*.jpg'))
  for fname in fnames:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(fname, img)

  fnames = glob(os.path.join(cfg.root, 'mask_*.png'))
  for fname in fnames:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(fname, img)

def create_charuco_board():
  boardImg = CHARUCO_BOARD.draw((500, 600), 10, 1)
  cv2.imwrite('BoardImage.png', boardImg)

def detect_charuco_board_and_calibrate_camera():
  # Create the arrays and variables we'll use to store info like corners and IDs from images processed
  corners_all = [] # Corners discovered in all images processed
  ids_all = [] # Aruco ids corresponding to corners discovered
  image_size = None # Determined at runtime

  fnames = glob(os.path.join(cfg.root, '*.jpg'))

  for fname in fnames:
    # Open the image
    img = cv2.imread(fname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find aruco markers in the query image
    corners, ids, _ = cv2.aruco.detectMarkers(
      image=gray,
      dictionary=ARUCO_DICT)

    # Outline the aruco markers found in our query image
    img = aruco.drawDetectedMarkers(
      image=img, 
      corners=corners)

    # Get charuco corners and ids from detected aruco markers
    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
      markerCorners=corners,
      markerIds=ids,
      image=gray,
      board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    if response > 0:
      # Add these corners and ids to our calibration arrays
      corners_all.append(charuco_corners)
      ids_all.append(charuco_ids)
      # Check enough detected keypoints (> 6)
      print(fname, len(charuco_ids))
      
      # Draw the Charuco board we've detected to show our calibrator the board was properly detected
      img = aruco.drawDetectedCornersCharuco(
        image=img,
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids)
      
      # If our image size is unknown, set it now
      if not image_size:
        image_size = gray.shape[::-1]
  
      # Reproportion the image, maxing width or height at 1000
      proportion = max(img.shape) / 1000.0
      img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
      # Pause to display each image, waiting for key press
      cv2.imshow('Charuco board', img)
      cv2.waitKey(0)
    else:
      print(f'Not able to detect a charuco board {response} in image: {fname}')

  # Destroy any open CV windows
  cv2.destroyAllWindows()

  # Make sure at least one image was found
  if len(fnames) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

  # Make sure we were able to calibrate on at least one charucoboard by checking
  # if we ever determined the image size
  if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()

  # Now that we've seen all of our images, perform the camera calibration
  # based on the set of points we've discovered
  calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=corners_all,
    charucoIds=ids_all,
    board=CHARUCO_BOARD,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=np.zeros((5,)),  # Disable distortion model
    flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)

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
  create_charuco_board()
  detect_charuco_board_and_calibrate_camera()
