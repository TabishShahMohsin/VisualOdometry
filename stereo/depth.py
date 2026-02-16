# https://www.youtube.com/watch?v=uKDAVcSaNZA&t=6s
import json
import numpy as np
import cv2 as cv

with open("params.json") as f:
    params = json.load(f)
    K1, D1, K2, D2, R, T, baseline = params['K1'], params['D1'], params['K2'], params['D2'], params['R'], params['T'], params['baseline']

def depth_map(img_left, img_right):
    img_size = img_left.shape[1], img_left.shape[0]

    # Rectification of images to ensure epipolar lines are parallel

    stereo = cv.StereoSGBM_create() # Needs args
    disparity = stereo.compute(img_left, img_right).astype(np.float32)/16
    points_3d = cv.reprojectImageTo3D(disparity, Q)
    d_map = points_3d[:, :, 2]
    d_map[disparity < 0] = 0

    return d_map, img_left