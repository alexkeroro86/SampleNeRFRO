import numpy as np

root = '/home/cgv839/data/real/dolphin'

# ------ Visualizer ------
near = 0.2
far = 1.2
# ------ Visual hull ------
num_voxels = 512
# AprilTag
# min_point = np.array([0.105, 0.105, 0.105])
# max_point = np.array([0.705, 0.705, 0.705])
# LLFF
min_point = None
max_point = None
threshold = 0.9
