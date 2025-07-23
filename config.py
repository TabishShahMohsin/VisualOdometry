import numpy as np
HEIGHT = 15 # in centimeters
WIDTH = 10
fx = fy = 800
cx = cy = 400
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
distCoeffs = np.zeros((4,1))  # or use your actual distortion
IMAGE_SIZE = (800, 800)