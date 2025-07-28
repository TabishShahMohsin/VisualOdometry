import numpy as np
import cv2

# === Step 1: Define world points (object frame) ===
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
], dtype=np.float32)

# === Step 2: Define camera intrinsics (camera matrix) ===
focal_length = 800  # in pixels
image_center = (320, 240)  # assume 640x480 image
K = np.array([
    [focal_length, 0, image_center[0]],
    [0, focal_length, image_center[1]],
    [0, 0, 1]
], dtype=np.float32)

# === Step 3: Define a downward + yawed camera rotation ===

# Rotation: 180 degrees around X-axis (flip to look down)
Rx = np.array([
    [1, 0,  0],
    [0, -1, 0],
    [0, 0, -1]
], dtype=np.float32)

# Add yaw: rotate around Z axis by 30 degrees
yaw_deg = 30
yaw_rad = np.deg2rad(yaw_deg)
Rz = np.array([
    [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
    [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
    [0, 0, 1]
], dtype=np.float32)

# Combined rotation: first yaw, then flip to look down
R_yaw_down = Rx @ Rz  # Important: Order matters!

# Translation vector: camera is at (0.5, 0.5, 1.0)
tvec_true = np.array([[0.5], [0.5], [1.0]])

# Convert rotation matrix to rvec
rvec_true, _ = cv2.Rodrigues(R_yaw_down)

# === Step 4: Project 3D points to image points ===
image_points, _ = cv2.projectPoints(object_points, rvec_true, tvec_true, K, None)

# === Step 5: Recover pose using solvePnP ===
success, rvec_est, tvec_est = cv2.solvePnP(object_points, image_points, K, None)

# === Step 6: Recover estimated camera world position ===
R_est, _ = cv2.Rodrigues(rvec_est)
camera_position_est = -R_est.T @ tvec_est

# === Step 7: Print all values ===
print("=== Ground Truth ===")
print("True rvec:\n", rvec_true)
print("True tvec:\n", tvec_true)

print("\n=== Estimated ===")
print("Estimated rvec:\n", rvec_est)
print("Estimated tvec:\n", tvec_est)

print("\n=== Camera Position (World Frame) ===")
print("Estimated Camera Position:\n", camera_position_est.T)