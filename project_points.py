import numpy as np
from config import K, HEIGHT, WIDTH
# Future: Make it fast using njit, parallel, and make it robust


def rotation_matrix(roll, pitch, yaw):
    # Correction for looking down
    R_c = np.array(
        [
            [1, 0,0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )

    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx @ R_c


def project_points(points_3d, R, t, K):
    # Inverse transform: world to camera coordinates
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv
    mask = points_cam[:, 2] > 0  # keep only points in front of the camera
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32), mask


def proj_pts(roll, pitch, yaw, tx, ty, tz):
    # Rotation and translation of the camera
    R = rotation_matrix(roll, pitch, yaw)
    t = np.array([tx, ty, tz])

    # Grid setup
    tile_w, tile_h = WIDTH, HEIGHT  # cm
    rows, cols = 8, 8
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2

    for i in range(rows):
        for j in range(cols):
            # Grid centered at origin
            x0 = j * tile_w - grid_cx
            y0 = i * tile_h - grid_cy
            corners = np.array(
                [
                    [x0, y0, 0],
                    [x0 + tile_w, y0, 0],
                    [x0 + tile_w, y0 + tile_h, 0],
                    [x0, y0 + tile_h, 0],
                ]
            )
            pts_2d, mask = project_points(corners, R, t, K)


    return pts_2d, mask

