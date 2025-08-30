import cv2
import numpy as np
import os
import csv
import random
from scipy.spatial.transform import Rotation as R_scipy
from pathlib import Path

folder_path = Path(__file__).resolve().parent

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

# --- Dataset Settings ---
NUM_IMAGES = 5000  # Total number of images to generate
# IMPORTANT: Use an ABSOLUTE path to your output folder
OUTPUT_PATH = folder_path.parent / 'dataset' 

# --- Grid and Tile Settings (in cm) ---
TILE_WIDTH = 5.0
TILE_HEIGHT = 3.0
GRID_ROWS = 50
GRID_COLS = 50

# --- Render Settings ---
IMG_WIDTH = 800
IMG_HEIGHT = 800

# --- Camera Intrinsics ---
FX = 800
FY = 800
CX = IMG_WIDTH / 2
CY = IMG_HEIGHT / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

# --- Camera Pose Randomization Ranges ---
ROLL_RANGE = (-0, 0)   # degrees
PITCH_RANGE = (-0, 0)  # degrees
YAW_RANGE = (-0, 0) # degrees
TX_RANGE = (-20, 20)     # cm
TY_RANGE = (-20, 20)     # cm
TZ_RANGE = (50, 50)     # cm (distance from grid)

# ==============================================================================
#                   CORE PROJECTION & DRAWING FUNCTIONS
# ==============================================================================

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Converts Euler angles to a 3x3 rotation matrix."""
    # Correction for looking down at the XY plane from a +Z camera view
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

    # Project only the points that are in front of the camera
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
        # === FIX START ===
        # The original line had a shape error. This line correctly creates
        # an array of shape (num_points, 2) to match the output of project_points.
        all_pts = np.full((points_3d.shape[0], 2), -1, dtype=np.int32)
        # === FIX END ===
        
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
#                       DATASET GENERATION SCRIPT
# ==============================================================================

def setup_directories(base_path):
    """Creates the output directories for images and the label file."""
    image_dir = os.path.join(base_path, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    return image_dir, os.path.join(base_path, "labels.csv")

def main():
    """Main function to generate the dataset."""
    print("Starting dataset generation...")
    image_dir, label_file_path = setup_directories(OUTPUT_PATH)

    with open(label_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = [
            "image_name", "tile_w", "tile_h", "fx", "fy", "cx", "cy",
            "label_tx", "label_ty", "tz", "qw", "qx", "qy", "qz"
        ]
        csv_writer.writerow(header)

        for i in range(NUM_IMAGES):
            print(f"Generating image {i+1}/{NUM_IMAGES}...")

            roll = random.uniform(*ROLL_RANGE)
            pitch = random.uniform(*PITCH_RANGE)
            yaw = random.uniform(*YAW_RANGE)
            tx = random.uniform(*TX_RANGE)
            ty = random.uniform(*TY_RANGE)
            tz = random.uniform(*TZ_RANGE)

            R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
            t_vec = np.array([tx, ty, tz])

            image = draw_scene(R_mat, t_vec)

            image_name = f"image_{i:05d}.png"
            cv2.imwrite(os.path.join(image_dir, image_name), image)

            r = R_scipy.from_matrix(R_mat)
            quat = r.as_quat() # (x, y, z, w)
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

            label_tx = tx % TILE_WIDTH
            label_ty = ty % TILE_HEIGHT

            label_row = [
                image_name, TILE_WIDTH, TILE_HEIGHT, FX, FY, CX, CY,
                label_tx, label_ty, tz,
                qw, qx, qy, qz
            ]
            csv_writer.writerow(label_row)

    print("-" * 30)
    print(f"Dataset generation complete! {NUM_IMAGES} images and 1 label file saved to:")
    print(OUTPUT_PATH)
    print("-" * 30)

if __name__ == "__main__":
    main()