import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

def plot_camera_pose(rot, trans):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, 0, 0, 50, color='red', label='Left Cam')
    ax.scatter(0, 0, 0, color='red')
    t = trans.flatten()
    z_dir = rot @ np.array([0, 0, 50])
    ax.quiver(t[0], t[1], t[2], z_dir[0], z_dir[1], z_dir[2], color='blue', label='Right Cam')
    ax.scatter(t[0], t[1], t[2], color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend(); plt.show()

chessboardSize = (9, 6)
frame_size = () # Would need to check
square_size = 24.00 # mm, need this so that the baseline comes in mm

# Termination criteria, stop after 30 iterations or improvement < 0.001
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points for the grid in the range
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= square_size
plt.scatter(objp[:, 0], objp[:, 1])
plt.title('Object points at z=0')
plt.show()

# Arrays to store obj and img points:
objpoints = [] # 3d points in real world space
imgpointL = [] # 2d points in the left frame on image plane
imgpointR = [] # 2d points in the right frame on image plane

imagesLeft = glob.glob('left/*.jpg')
imagesRight = glob.glob('right/*.jpg')

for img_left, img_right in zip(imagesLeft, imagesRight):
    
    imgL = cv.imread(img_left)
    imgR = cv.imread(img_right)
    grayL = cv.cvtColor(imgL, cv.COLOR_RGB2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_RGB2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize)

    if retL and retR:
        objpoints.append(objp) # Doing this for maintaining dimesions, even though with repetition

        # Refining for better accuracy:
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointL.append(cornersL)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointR.append(cornersR)

        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('Left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('Right', imgR)
        cv.waitKey(10)

cv.destroyAllWindows()

# Obtaining the new camera matrix for precision
# Find out the correct alpha
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointL, grayL.shape[::-1], None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roiL = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 0, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointR, grayR.shape[::-1], None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roiR = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 0, (widthR, heightR))

dstL = cv.undistort(imgL, cameraMatrixL, distL, None, newCameraMatrixL)
dstR = cv.undistort(imgR, cameraMatrixR, distR, None, newCameraMatrixR)
cv.rectangle(dstL, (roiL[0], roiL[1]), (roiL[0] + roiL[2], roiL[1] + roiL[3]), (255, 0, 0), 3)
cv.rectangle(dstR, (roiR[0], roiR[1]), (roiR[0] + roiR[2], roiR[1] + roiR[3]), (255, 0, 0), 3)
cv.imshow('Region of Interest after finding the newK L', dstL)
cv.imshow('Region of Interest after finding the newK R', dstR)
cv.waitKey(0)

# Getting the transformation between the 2 cameras
flags = cv.CALIB_FIX_INTRINSIC # We know that the K1, K2 are good enough to remain fix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundametalMatrix = cv.stereoCalibrate(objpoints, imgpointL, imgpointR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], flags=flags, criteria=criteria)
plot_camera_pose(rot, trans)

# Stereo Rectification
rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0, 0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

rectImgL = cv.remap(imgL, stereoMapL[0], stereoMapL[1], cv.INTER_LINEAR)
rectImgR = cv.remap(imgR, stereoMapR[0], stereoMapR[1], cv.INTER_LINEAR)
vis = np.hstack((rectImgL, rectImgR))
for y in range(0, vis.shape[0], 50):
    cv.line(vis, (0, y), (2* vis.shape[1], y), (0, 255, 0), 3)
cv.imshow('Showing perfect epipolar lines.', vis)
cv.waitKey(0)

# Saving the exact pixel(original-distorted) to undistorted-pixel map
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)
cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

cv_file.release()
