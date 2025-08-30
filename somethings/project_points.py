import numpy as np
from config import K, HEIGHT, WIDTH
import cv2

# TODO: Add njit & parallel later for speedup (requires fixed shapes/types)


def project_points(points_3d, R, t, K):
    # Transform world points to camera coordinates (no inverse!)
    points_cam = (R @ points_3d.T).T + t.T  # t should be (3, 1)

    # Need to filter out the points
    valid_points = points_cam

    if valid_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Project onto image plane
    points_proj = (K @ valid_points.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32)


def proj_pts(rvec, t):
    # Validate rvec and t shapes
    rvec = np.asarray(rvec, dtype=np.float32).reshape(3, 1)
    t = np.asarray(t, dtype=np.float32).reshape(3, 1)

    # Define camera-to-world rotation
    # R_c = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, -1, 0],
    #         [0, 0, -1],
    #     ],
    #     dtype=np.float32
    # )
    # R = cv2.Rodrigues(rvec)[0] @ R_c
    R = cv2.Rodrigues(rvec)[0] 

    # Grid parameters
    tile_w, tile_h = WIDTH, HEIGHT
    rows, cols = 16, 16 #TODO: Check why odd doesn't work here 
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2

    all_masks = []
    all_2d_points = list()

    for i in range(rows):
        for j in range(cols):
            x0 = j * tile_w - grid_cx
            y0 = i * tile_h - grid_cy

            corners = np.array([
                [x0, y0, 0],
                [x0 + tile_w, y0, 0],
                [x0 + tile_w, y0 + tile_h, 0],
                [x0, y0 + tile_h, 0],
            ], dtype=np.float32)
            pts_2d = project_points(corners, R, t, K)
            all_2d_points.append(pts_2d[0]) 
            all_2d_points.append(pts_2d[1]) 
            all_2d_points.append(pts_2d[2]) 
            all_2d_points.append(pts_2d[3]) 

    return np.array(all_2d_points)