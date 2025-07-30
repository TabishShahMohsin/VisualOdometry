import numpy as np
HEIGHT = 15 # in centimeters
WIDTH = 10
# HEIGHT = 10 # in centimeters
# WIDTH = 15
fx = fy = 800
cx = cy = 400
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
distCoeffs = np.zeros((4, 1), dtype=np.float32) # Don't change this line, this was after hours of debugging
IMAGE_SIZE = (800, 800)