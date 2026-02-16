import cv2
import numpy as np
import os
import csv
from scipy.spatial.transform import Rotation as R_scipy
from pathlib import Path

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

# --- Video Settings ---
VIDEO_FILENAME = "crab_walk_video.mp4"
GT_CSV_FILENAME = "gt_trajectory.csv"
VIDEO_FPS = 30
VIDEO_DURATION_S = 10 # Duration for one full loop of the rectangle
IMG_WIDTH = 800
IMG_HEIGHT = 800

# --- Dataset Path ---
OUTPUT_PATH = Path(__file__).resolve().parent.parent / 'dataset'

# --- Grid and Tile Settings (in cm) ---
TILE_WIDTH = 5.0
TILE_HEIGHT = 3.0
GRID_ROWS = 50
GRID_COLS = 50

# --- Camera Intrinsics ---
FX = 800
FY = 800
CX = IMG_WIDTH / 2
CY = IMG_HEIGHT / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

# --- Camera Path Settings ---
RECT_WIDTH = 40 # cm
RECT_HEIGHT = 20 # cm
Z_DISTANCE = 50 # cm

# ==============================================================================
#                   CORE PROJECTION & DRAWING FUNCTIONS
# ==============================================================================

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Converts Euler angles to a 3x3 rotation matrix."""
    R_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    r = R_scipy.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix() @ R_c

def project_points(points_3d, R, t, K):
    """Projects 3D points to 2D image plane."""
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv
    mask = points_cam[:, 2] > 0

    if not np.any(mask):
        return np.array([]), mask

    points_proj = (K @ points_cam[mask].T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32), mask

def draw_scene(R, t):
    """Draws the projected grid onto a grayscale image canvas."""
    # Create a single-channel (grayscale) image
    img = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) * 220 # Light grey background

    grid_cx = GRID_COLS * TILE_WIDTH / 2
    grid_cy = GRID_ROWS * TILE_HEIGHT / 2

    x = np.arange(0, (GRID_COLS + 1) * TILE_WIDTH, TILE_WIDTH) - grid_cx
    y = np.arange(0, (GRID_ROWS + 1) * TILE_HEIGHT, TILE_HEIGHT) - grid_cy
    xx, yy = np.meshgrid(x, y)
    points_3d = np.vstack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]).T

    pts_2d, mask = project_points(points_3d, R, t, K)

    if pts_2d.shape[0] > 0:
        # The original line had a shape error. This line correctly creates
        # an array of shape (num_points, 2) to match the output of project_points.
        all_pts = np.full((points_3d.shape[0], 2), -1, dtype=np.int32)
        
        all_pts[mask] = pts_2d
        grid_pts_2d = all_pts.reshape(xx.shape + (2,))

        # Draw horizontal lines
        for i in range(GRID_ROWS + 1):
            row_pts = grid_pts_2d[i, :, :][grid_pts_2d[i, :, 0] != -1]
            if len(row_pts) > 1:
                # Use a grayscale color (0 for black)
                cv2.polylines(img, [row_pts], False, 0, 1)

        # Draw vertical lines
        for j in range(GRID_COLS + 1):
            col_pts = grid_pts_2d[:, j, :][grid_pts_2d[:, j, 0] != -1]
            if len(col_pts) > 1:
                # Use a grayscale color (0 for black)
                cv2.polylines(img, [col_pts], False, 0, 1)
    
    return img

# ==============================================================================
#                       VIDEO GENERATION SCRIPT
# ==============================================================================

def main():
    """Main function to generate the video and ground truth CSV."""
    print("Starting video generation...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    video_path = str(OUTPUT_PATH / VIDEO_FILENAME)
    gt_csv_path = str(OUTPUT_PATH / GT_CSV_FILENAME)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (IMG_WIDTH, IMG_HEIGHT), isColor=False)

    # Define the rectangular path
    path_points = [
        np.array([0, 0, Z_DISTANCE]),
        np.array([RECT_WIDTH, 0, Z_DISTANCE]),
        np.array([RECT_WIDTH, RECT_HEIGHT, Z_DISTANCE]),
        np.array([0, RECT_HEIGHT, Z_DISTANCE]),
        np.array([0, 0, Z_DISTANCE])
    ]
    
    total_frames = VIDEO_FPS * VIDEO_DURATION_S
    frames_per_segment = total_frames // (len(path_points) - 1)

    # Fixed rotation (no rotation)
    R_mat = euler_to_rotation_matrix(0, 0, 0)

    with open(gt_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "x", "y", "z"])

        frame_num = 0
        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i+1]
            
            for frame_idx in range(frames_per_segment):
                alpha = frame_idx / frames_per_segment
                t_vec = (1 - alpha) * start_point + alpha * end_point
                
                image = draw_scene(R_mat, t_vec)
                video_writer.write(image)
                
                csv_writer.writerow([frame_num, t_vec[0], t_vec[1], t_vec[2]])
                frame_num += 1

    video_writer.release()
    print("-" * 30)
    print(f"Video generation complete! Saved to:")
    print(video_path)
    print(f"Ground truth trajectory saved to:")
    print(gt_csv_path)
    print("-" * 30)

if __name__ == "__main__":
    main()
