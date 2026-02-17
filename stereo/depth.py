# https://www.youtube.com/watch?v=uKDAVcSaNZA&t=6s
import numpy as np
import cv2 as cv

min_disp = 0
block_size = 7
num_disp = 16*28

stereo = cv.StereoSGBM_create(
    min_disp,
    num_disp,
    block_size,
    P1 = 8*8*block_size**2,
    P2 = 32*8*block_size**2,
    disp12MaxDiff=15,
    uniquenessRatio = 1,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

cv_file = cv.FileStorage('stereoMap.xml', cv.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

def undistortRectify(frameL, frameR):
    undistortedL = cv.remap(frameL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4)
    undistortedR = cv.remap(frameR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4)
    return undistortedL, undistortedR

# imgL = cv.imread('Calibration/left/left.png')
# imgR = cv.imread('Calibration/right/right.png')
imgL = cv.imread('Calibration/left/x.jpg')
imgR = cv.imread('Calibration/right/x.jpg')

rectifiedL, rectifiedR = undistortRectify(imgL, imgR)

grayL = cv.cvtColor(rectifiedL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(rectifiedR, cv.COLOR_BGR2GRAY)

disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

dis_vis = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
vis = np.hstack((rectifiedL, rectifiedR))
for y in range(0, vis.shape[0], 50):
    cv.line(vis, (0, y), (vis.shape[1], y), (0, 255, 0))
cv.imshow('Matches', vis)
cv.imshow('grayL', grayL)
cv.imshow('grayR', grayR)
cv.imshow('disparity', dis_vis)
cv.waitKey(0)
cv.destroyAllWindows()

import time
while True:
    cv.imshow('show', dis_vis)
    if cv.waitKey(500) == ord('q'):
        break
    cv.imshow('show', rectifiedL)
    if cv.waitKey(500) == ord('q'):
        break
cv.destroyAllWindows()

print(np.min(disparity), np.max(disparity))