import cv2
import numpy as np

# Goals:
    # Moving the AUV in a rectangle
    # Extracting pitures out of the motion
    # Tracking corners with corner id
    # Corner should have a bool for in_frame
    
# Future:
    # Adding noise in everything including roll and pitch
    # 

# ==== Editable Grid Parameters ====
tile_w = 10     # Tile width (along x) in cm
tile_h = 15     # Tile height (along y) in cm
rows = 30       # Number of rows to show in both +y and -y directions (i.e. 2*rows in total)
cols = 30       # Number of columns to show in both +x and -x directions (i.e. 2*cols in total)

# ==== Function to get rotation matrix with yaw (Z-axis rotation) ====
def rotation_matrix_yaw_only(yaw_deg):
    # Convert degrees to radians, negative because we are going from world to camera
    yaw = np.radians(-yaw_deg)

    # Standard Z-axis rotation matrix (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    # Since the camera is facing downward (Z_cam aligned with -Z_world),
    # we apply a rotation to align Z_cam with -Z_world
    R_align = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    # Return the combined rotation matrix: R_align * Rz
    return R_align @ Rz

# ==== Function to project 3D world points into 2D camera image points ====
def project_points(points_3d, R, t, K):
    # Convert rotation matrix from world-to-camera
    Rt = R.T

    # Inverse translation vector in camera coordinates
    t_inv = -Rt @ t

    # Apply rotation and translation to convert world points to camera coordinates
    # X_cam = R_inv(X_world - t), as yaw is about the point t and the cam is over t
    points_cam = (Rt @ points_3d.T).T + t_inv

    # Mask to keep only the points in front of the camera (z > 0)
    mask = points_cam[:, 2] > 0

    # Project points into image plane using intrinsic matrix K
    # X_proj = K X_cam, where the result would be a homogeneous matrix
    points_proj = (K @ points_cam.T).T

    # Normalize homogeneous coordinates (divide by z)
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]

    return points_2d.astype(np.int32), mask

# ==== Function to render the scene from the camera's perspective ====
def draw_scene(yaw, tx, ty, tz):
    # ---- Camera Intrinsic Parameters (fx, fy = focal lengths; cx, cy = principal point) ----
    fx = fy = 800
    cx = cy = 400
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # ---- Compute camera rotation and translation ----
    R = rotation_matrix_yaw_only(yaw)
    t = np.array([tx, ty, tz])  # Translation vector

    # ---- Create blank white canvas (800x800 pixels, 1 channel = grayscale) ----
    img = np.ones((800, 800), dtype=np.uint8) * 255  # 255 = white
    # The image size is a camera property, and hence changing it would need changing K

    # ---- Draw tiles centered around the origin ----
    for i in range(-rows, rows):         # y-axis tiles (top to bottom)
        for j in range(-cols, cols):     # x-axis tiles (left to right)
            # Compute the bottom-left corner of the tile
            x0 = j * tile_w
            y0 = i * tile_h

            # Define 4 corners of the tile in 3D (Z=0, i.e. on the ground)
            corners = np.array([
                [x0,           y0,            0],
                [x0 + tile_w,  y0,            0],
                [x0 + tile_w,  y0 + tile_h,   0],
                [x0,           y0 + tile_h,   0]
            ])

            # Project the 3D corners to 2D image
            pts_2d, mask = project_points(corners, R, t, K)

            # Only draw if all 4 corners are in front of the camera
            if mask.all():
                # Draw polygon (tile outline) on the image
                cv2.polylines(img, [pts_2d], isClosed=True, color=(0, 0, 0), thickness=1)

    # ---- Draw red circle at the origin ----
    origin_2d, visible = project_points(np.array([[0, 0, 0]]), R, t, K)
    if visible[0]:
        # Mark origin with a red dot
        cv2.circle(img, tuple(origin_2d[0]), radius=5, color=(0, 0, 255), thickness=-1)
        # Label it as "Origin"
        cv2.putText(img, "Origin", tuple(origin_2d[0] + [10, -10]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 0),
                    thickness=1)

    return img

# ==== Callback for GUI sliders (trackbars) ====
# This is the "function reference" that OpenCV calls when a slider changes
# `_` is a dummy argument passed by OpenCV (the new trackbar value)
# def update(_=None):
#     # Read current values from sliders
#     yaw = cv2.getTrackbarPos("Yaw", win)           # [0–360]
#     tx  = cv2.getTrackbarPos("Tx", win) - 100      # [-100 to +100]
#     ty  = cv2.getTrackbarPos("Ty", win) - 100      # [-100 to +100]
#     tz  = cv2.getTrackbarPos("Tz", win) + 1        # [51 to 301], avoid 0 to prevent divide-by-zero

#     # Redraw the image with updated parameters
#     img = draw_scene(yaw, tx, ty, tz)

#     # Show the new image in the same window
#     cv2.imshow(win, img)
#     # cv2.imwrite("changed.jpeg", img) # Can save images just like that with a particular parameters

# # ==== GUI setup ====
# win = "AUV Camera View"                # Name of the OpenCV window
# cv2.namedWindow(win)                   # Create the window

# # Each trackbar is linked to the update() function, which is called when the slider is moved
# # This is called a "callback" — we pass the function `update` as a reference (not update())
# cv2.createTrackbar("Yaw", win, 0, 360, update)     # Starts at 180 → maps to yaw = 0
# cv2.createTrackbar("Tx",  win, 100, 200, update)     # Maps to tx = -100 to +100
# cv2.createTrackbar("Ty",  win, 100, 200, update)     # Maps to ty = -100 to +100
# cv2.createTrackbar("Tz",  win, 50, 300, update)      # Maps to tz = 51 to 301 (always positive)

# # Initial draw: we manually call update() once to draw the first frame before any slider is moved
# update()

# # ==== Event loop to keep the window open ====
# # waitKey(30) waits for 30 milliseconds for a key press
# # If you press ESC (key code 27), it breaks the loop and closes the window
# while True:
#     if cv2.waitKey(30) == 27:  # 27 = ESC key
#         break

# cv2.destroyAllWindows()  # Clean up all OpenCV windows


yaw = 0
# For now the distance is in cm
tx = 0
ty = 0
tz = 50

img = draw_scene(yaw, tx, ty, tz)

# cv2.imshow("scene", img)
cv2.waitKey(0)

import os
os.makedirs("data", exist_ok=True)
base_path = os.path.join('.', 'data', 'image_')

cv2.imwrite(base_path + '0.jpg', img)
image_count = 0

for i in range(100):
    tx += 1
    image_count += 1
    cv2.imwrite(base_path + str(image_count) + '.jpg', draw_scene(yaw, tx, ty, tz))

for i in range(100):
    ty += 1
    image_count += 1
    cv2.imwrite(base_path + str(image_count) + '.jpg', draw_scene(yaw, tx, ty, tz))

for i in range(90):
    yaw += np.deg2rad(2)
    image_count += 1
    cv2.imwrite(base_path + str(image_count) + '.jpg', draw_scene(yaw, tx, ty, tz))


for i in range(100):
    tx -= 1
    image_count += 1
    cv2.imwrite(base_path + str(image_count) + '.jpg', draw_scene(yaw, tx, ty, tz))

for i in range(100):
    ty -= 1
    image_count += 1
    cv2.imwrite(base_path + str(image_count) + '.jpg', draw_scene(yaw, tx, ty, tz))

    
for i in range(90):
    yaw -= np.deg2rad(2)
    image_count += 1
    cv2.imwrite(base_path + str(image_count) + '.jpg', draw_scene(yaw, tx, ty, tz))
    
