# Trying to implement a function that on the basis of the given intersection points would return a R:T matrix   
import math
import numpy as np
import random
from config import HEIGHT, WIDTH, K, IMAGE_SIZE
import cv2
distCoeffs = np.zeros((1, 4))  # or use your actual distortion
distCoeffs = np.asarray(distCoeffs, dtype=np.float64)

def ransac(intersections:list) -> np.array:
    # This function should take the intersections and return a np.array

    # First making the first point choice random:
    random.shuffle(intersections)
    
    # The rest choices would be like 180 roll, 180 pitch, or simply translation.
    objectPoints = np.array([
        [0, 0, 0],               # p1
        [0, WIDTH, 0],           # p2
        [-HEIGHT, WIDTH, 0],      # p3
        [-HEIGHT, 0, 0]           # p4
    ], dtype=np.float32)

    for point in intersections:
        # Getting the poitns in a cyclic order with the closest point as the first
        points = get_8_closest_cyclic(point, intersections) # Remember they are going to be counter - clockwise
        if points == None: 
            continue

        for i in range(0, 8, 2): # This is the order that ensures that width is the first point
            imagePoints = np.array([
                point,  # corresponding to p4
                points[i],  # corresponding to p1
                points[(i + 1)],  # corresponding to p2
                points[(i + 2) % 8],  # corresponding to p3
            ], dtype=np.float32)
            try:
                success, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, K, distCoeffs)
                if success:
                    return rvec, tvec, imagePoints, point
                # PNP is failing in many cases
                # t = np.rint(tvec % np.array([[10], [15], [1]])).astype(np.int16)
                # r = np.rint(np.rad2deg(rvec) % np.array([180])).astype(np.int16)
                # if success:
                #     print('t', t,'\n', 'r',  r)
            except:
                pass
    # Ransac is to be used here




def get_8_closest_cyclic(target_point, point_list):
    x0, y0 = target_point
    # Add some condition so that there are no points at the ends of the picture
    # Should add determinant or some property of rectangle here, to be implemneted later especially for the non - ideal images
    if not(0.25 * IMAGE_SIZE[1] < x0 < 0.75 * IMAGE_SIZE[1] and 0.25 * IMAGE_SIZE[0] < y0 < 0.75 * IMAGE_SIZE[0]):
        return None

    # Step 1: Get 8 closest (excluding the point itself)
    closest = sorted(point_list, key=lambda p: math.hypot(p[0] - x0, p[1] - y0))[1:9]

    # Step 3: Sort by angle around centroid
    def angle(p):
        return math.atan2(p[1] - y0, p[0] - x0)
    
    closest.sort(key=angle)

    # Step 4: Find the point closest to the target to start the cycle
    min_idx = min(range(8), key=lambda i: (closest[i][0] - x0)**2 + (closest[i][1] - y0)**2)
    
    # Step 5: Rotate list to start from closest to target
    ordered = closest[min_idx:] + closest[:min_idx]

    

    return ordered

