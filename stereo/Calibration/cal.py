import numpy as np
import cv2 as cv
import glob

chessboardSize = (9, 6)
frame_size = () # Would need to check
square_size = 24.00 # mm, need this so that the baseline comes in mm

# Termination criteria, stop after 30 iterations or improvement < 0.001
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points for the grid in the range
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= square_size

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
        cv.waitKey(100)

cv.destroyAllWindows()

# --- STEREO CALIBRATION ---
# Perform initial individual calibration to get better starting intrinsics
retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointR, grayR.shape[::-1], None, None)

# The "magic" function that finds the relationship between the two cameras
flags = cv.CALIB_FIX_INTRINSIC
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpointL, imgpointR, mtxL, distL, mtxR, distR, 
    grayL.shape[::-1], criteria=criteria, flags=flags)

# --- SAVE PARAMETERS ---
print("Calibration complete. Saving to stereo_rectify_maps.npz")
np.savez('stereo_params.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print(f"{mtxL=}, \n {distL=}, \n {mtxR=}, \n {distR=}, \n {R=}, \n {T=}")