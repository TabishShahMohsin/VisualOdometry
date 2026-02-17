# -*- coding: utf-8 -*-
"""
Camera Pose Estimation from a Grid Pattern.

This script estimates the 3D position (x, y, z) and orientation (yaw) of a
camera relative to a regular 2D grid on a flat plane.

It operates in two main stages:
1.  **FFT Initial Guess:** Uses a 2D Fast Fourier Transform on the central
    region of the image to get a fast, robust initial estimate of the
    camera's height (z) and rotation (yaw).
2.  **Physics-Based Refinement:** Uses the initial guess to seed a numerical
    optimizer (`scipy.optimize.minimize`). The optimizer refines the full
    6-DOF pose by projecting a 3D model of the grid and minimizing the
    distance between the projected lines and the actual edges detected
    in the image.

Dependencies:
- opencv-python
- numpy
- matplotlib
- scikit-learn
- scipy

Usage:
    python your_script_name.py
    (Ensure 'datasets/synthetic/pics/image_2.jpg' exists or change the path)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans

def detect_grid_fft(image, tile_world, focal_length, center_roi=0.5):
    """
    Detect grid orientation and spacing using Fourier Transform for an initial estimate.

    Args:
        image (np.array): The input image.
        tile_world (tuple): (width, height) of grid tiles in world units (e.g., cm).
        focal_length (float): Camera focal length in pixels.
        center_roi (float): Fraction of the image's center to use for analysis.

    Returns:
        tuple: (yaw, z_estimate) in radians and world units.
    """
    # Convert to grayscale and crop the central region to minimize lens distortion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi_size = int(min(h, w) * center_roi)
    y1, y2 = h//2 - roi_size//2, h//2 + roi_size//2
    x1, x2 = w//2 - roi_size//2, w//2 + roi_size//2
    crop = gray[y1:y2, x1:x2]

    # Compute 2D FFT and shift the zero-frequency component to the center
    fft = np.fft.fft2(crop)
    fshift = np.fft.fftshift(fft)
    
    # Create a high-pass filter to remove the DC component (center)
    mask = np.ones(crop.shape, np.uint8)
    cy, cx = crop.shape[0]//2, crop.shape[1]//2
    # A fixed small radius is often more robust for removing the DC component
    filter_size = 25
    cv2.circle(mask, (cx, cy), filter_size, 0, -1)
    
    fshift_filtered = fshift * mask
    magnitude_filtered = np.log(np.abs(fshift_filtered) + 1)
    
    # Find prominent frequency peaks using non-maximum suppression
    peaks = []
    # A larger margin helps separate distinct frequency peaks
    peak_margin = 10
    temp_magnitude = magnitude_filtered.copy()
    for _ in range(20): # Find up to 20 peaks
        _, max_val, _, max_loc_tuple = cv2.minMaxLoc(temp_magnitude)
        # Stop if peaks are no longer significant
        if max_val < 7.0:
            break
        max_loc = (max_loc_tuple[1], max_loc_tuple[0]) # (row, col)
        peaks.append(max_loc)
        # Suppress the area around the found peak to find the next one
        cv2.circle(temp_magnitude, (max_loc[1], max_loc[0]), peak_margin, 0, -1)
        
    if len(peaks) < 2:
        return 0.0, 150.0 # Return default values if pattern is not found

    # Convert peak locations to vectors originating from the spectrum's center
    peak_vectors = np.array([(p[1] - cx, cy - p[0]) for p in peaks])
    
    # Cluster peaks into two groups representing the two primary grid directions
    if len(peak_vectors) < 2:
        return 0.0, 150.0

    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0).fit(peak_vectors)
    cluster_centers = kmeans.cluster_centers_

    # Identify clusters for horizontal vs. vertical lines based on their angle.
    # Vertical grid lines produce peaks near the horizontal frequency axis.
    angles = np.arctan2(cluster_centers[:, 1], cluster_centers[:, 0])
    vert_lines_idx = np.argmin(np.abs(angles))
    horiz_lines_idx = 1 - vert_lines_idx

    # Yaw is the rotation of the vertical lines.
    yaw = angles[vert_lines_idx]

    # Calculate spatial period (pixel spacing) from frequency magnitude.
    # Period T = N / f, where N is ROI size and f is frequency magnitude.
    magnitudes = np.linalg.norm(cluster_centers, axis=1)
    
    spacing_vert_lines_px = roi_size / (magnitudes[vert_lines_idx] + 1e-8)
    spacing_horiz_lines_px = roi_size / (magnitudes[horiz_lines_idx] + 1e-8)
    
    # Estimate depth Z = f * (X_world / X_image) = f / scale
    scale_vert = spacing_vert_lines_px / tile_world[0]
    scale_horiz = spacing_horiz_lines_px / tile_world[1]
    
    valid_scales = [s for s in [scale_horiz, scale_vert] if s > 1e-5]
    if not valid_scales:
        return yaw, 150.0 # Return default z if scale is invalid
        
    z_estimate = focal_length / np.mean(valid_scales)
    
    return yaw, z_estimate


def physics_loss(params, edge_dt, K, grid_lines_3d, img_shape):
    """Robust physics-based loss function for refinement."""
    tx, ty, tz, yaw = params
    R = rotation_matrix(yaw)
    t = np.array([tx, ty, tz])
    
    total_dt = 0
    total_points = 0
    
    for line in grid_lines_3d:
        pts_2d, mask = project_points(line, R, t, K)
        if not mask.all():
            continue
            
        p1, p2 = pts_2d[0], pts_2d[1]
        length = np.linalg.norm(p2 - p1)
        num_samples = max(2, int(length / 5))
        t_vals = np.linspace(0, 1, num_samples)
        points = np.column_stack([
            p1[0]*(1-t_vals) + p2[0]*t_vals,
            p1[1]*(1-t_vals) + p2[1]*t_vals
        ])
        
        valid_x = (points[:, 0] >= 0) & (points[:, 0] < img_shape[1])
        valid_y = (points[:, 1] >= 0) & (points[:, 1] < img_shape[0])
        valid_points = points[valid_x & valid_y].astype(int)
        
        if len(valid_points) == 0:
            continue
            
        dt_values = edge_dt[valid_points[:, 1], valid_points[:, 0]]
        # Clamp distance to prevent single outliers from dominating the loss
        MAX_DIST = 20.0
        dt_values = np.clip(dt_values, 0, MAX_DIST)
        
        total_dt += np.sum(dt_values)
        total_points += len(dt_values)
    
    if total_points == 0:
        return 1e6 # Large penalty if no grid lines are visible

    mean_dt = total_dt / total_points
    
    # Regularization terms to encourage plausible solutions
    z_reg = 0.01 * abs(tz - 150) # Prefer z around 150cm
    yaw_reg = 0.1 * abs(yaw)     # Prefer small yaw angles
    
    return mean_dt + z_reg + yaw_reg


def rotation_matrix(yaw):
    """Creates a rotation matrix for yaw and camera coordinate conversion."""
    # This matrix converts from a CV coordinate system (Z forward, Y down)
    # to a world system (e.g., Z up, Y forward) via a 180-deg rotation on X-axis.
    R_cam_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_yaw @ R_cam_to_world


def project_points(points_3d, R, t, K):
    """Projects 3D points to 2D image plane with safety checks."""
    R_inv = R.T
    t_inv = -R_inv @ t
    points_cam = (R_inv @ points_3d.T).T + t_inv
    
    mask = points_cam[:, 2] > 1e-5
    
    points_proj = (K @ points_cam.T).T
    
    # Perspective divide with an epsilon to prevent division by zero
    points_2d = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)
    
    return points_2d.astype(np.float32), mask


def create_grid_lines(tile_w, tile_h, rows=80, cols=80):
    """Creates a large 3D grid model to ensure it covers the camera's view."""
    grid_w = cols * tile_w
    grid_h = rows * tile_h
    lines = []
    
    for i in range(rows + 1):
        y = i * tile_h - grid_h / 2
        lines.append([[-grid_w / 2, y, 0], [grid_w / 2, y, 0]])
    
    for j in range(cols + 1):
        x = j * tile_w - grid_w / 2
        lines.append([[x, -grid_h / 2, 0], [x, grid_h / 2, 0]])
        
    return np.array(lines, dtype=np.float32)


def main():
    """Main execution function."""
    # --- Configuration ---
    IMAGE_PATH = 'datasets/synthetic/pics/image_2.jpg'
    # World dimensions of a single grid tile (width, height) in cm
    TILE_DIMENSIONS = (5.0, 3.0)
    # Plausible physical range for camera height (z-axis) in cm
    PLAUSIBLE_Z_RANGE = (50.0, 300.0)

    # --- Load Image ---
    try:
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            raise FileNotFoundError(f"Image not found at {IMAGE_PATH} or is corrupted.")
    except Exception as e:
        print(f"Error: {e}")
        return
        
    # --- Setup ---
    h, w = img.shape[:2]
    # Estimate camera intrinsics. fx=fy=image_width is a common heuristic.
    fx = fy = w
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # --- Stage 1: Initial Estimation ---
    yaw_rad, z_estimate = detect_grid_fft(img, TILE_DIMENSIONS, fx)
    print(f"Fourier Initial Estimate: yaw={np.degrees(yaw_rad):.1f}°, z={z_estimate:.1f}cm")

    # Clip the estimate to the plausible range to ensure valid optimizer bounds
    z_estimate = np.clip(z_estimate, PLAUSIBLE_Z_RANGE[0], PLAUSIBLE_Z_RANGE[1])
    print(f"Clipped Initial Estimate for Optimizer: z={z_estimate:.1f}cm")

    # --- Stage 2: Refinement ---
    # Prepare image for the loss function
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Distance transform is key: it tells projected lines how far they are from a real edge
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    
    grid_lines_3d = create_grid_lines(TILE_DIMENSIONS[0], TILE_DIMENSIONS[1])
    
    initial_params = [0, 0, z_estimate, yaw_rad]
    
    bounds = [
        (-50, 50),     # tx: horizontal translation range
        (-50, 50),     # ty: vertical translation range (along grid plane)
        PLAUSIBLE_Z_RANGE,
        (yaw_rad - np.radians(20), yaw_rad + np.radians(20))
    ]
    
    # Run the optimizer
    result = minimize(
        physics_loss, initial_params,
        args=(dist_transform, K, grid_lines_3d, img.shape),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-4}
    )
    
    # --- Visualization ---
    tx, ty, tz, yaw = result.x
    print(f"Optimized Parameters: tx={tx:.1f}cm, ty={ty:.1f}cm, tz={tz:.1f}cm, yaw={np.degrees(yaw):.1f}°")
    
    R = rotation_matrix(yaw)
    t = np.array([tx, ty, tz])
    result_img = img.copy()
    
    for line in grid_lines_3d:
        pts_2d, mask = project_points(line, R, t, K)
        if mask.all():
            p1 = tuple(pts_2d[0].astype(int))
            p2 = tuple(pts_2d[1].astype(int))
            # Draw only if the line is reasonably within the frame
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(result_img, p1, p2, (0, 255, 0), 2)
    
    text_color = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_img, f"Z: {tz:.1f}cm", (15, 30), font, 0.7, text_color, 2)
    cv2.putText(result_img, f"Yaw: {np.degrees(yaw):.1f}deg", (15, 60), font, 0.7, text_color, 2)
    cv2.putText(result_img, f"X: {tx:.1f}cm", (15, 90), font, 0.7, text_color, 2)
    cv2.putText(result_img, f"Y: {ty:.1f}cm", (15, 120), font, 0.7, text_color, 2)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Final Grid Projection')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()