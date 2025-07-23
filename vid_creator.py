import cv2
import numpy as np
# Future: Make it fast using njit, parallel, and make it robust

# === Editable Grid Parameters ===
tile_w = 10     # cm
tile_h = 15     # cm
rows = 30       # in +y and -y
cols = 30       # in +x and -x
fx = fy = 800
cx = cy = 400
K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]])

# === Yaw-only Rotation Matrix ===
def rotation_matrix_yaw_only(yaw_deg):
    yaw = np.radians(-yaw_deg)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    R_align = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    return R_align @ Rz

# === Project 3D to 2D ===
def project_points(points_3d, R, t, K):
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv
    mask = points_cam[:, 2] > 0
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32), mask

# === Draw Scene from AUV Pose ===
def draw_scene(yaw, tx, ty, tz):
    
    R = rotation_matrix_yaw_only(yaw)
    t = np.array([tx, ty, tz])

    # === Create white 3-channel canvas ===
    img = np.ones((800, 800, 3), dtype=np.uint8) * 255

    for i in range(-rows, rows):
        for j in range(-cols, cols):
            x0 = j * tile_w
            y0 = i * tile_h
            corners = np.array([
                [x0,           y0,            0],
                [x0 + tile_w,  y0,            0],
                [x0 + tile_w,  y0 + tile_h,   0],
                [x0,           y0 + tile_h,   0]
            ])
            pts_2d, mask = project_points(corners, R, t, K)
            if mask.all():
                cv2.polylines(img, [pts_2d], isClosed=True, color=(0, 0, 0), thickness=1)

    # Draw origin
    origin_2d, visible = project_points(np.array([[0, 0, 0]]), R, t, K)
    if visible[0]:
        cv2.circle(img, tuple(origin_2d[0]), 5, (0, 0, 255), -1)
        cv2.putText(img, "Origin", tuple(origin_2d[0] + [10, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

# === Main Motion and Video Creation ===
yaw, tx, ty, tz = 0, 0, 0, 125

frame = draw_scene(yaw, tx, ty, tz)
h, w = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('simple.mp4', fourcc, 30, (w, h))

video_writer.write(frame)

# Move in +x
for _ in range(100):
    tx += 1
    video_writer.write(draw_scene(yaw, tx, ty, tz))

# Move in +y
for _ in range(100):
    ty += 1
    video_writer.write(draw_scene(yaw, tx, ty, tz))

# Rotate +180°
# for _ in range(90):
#     yaw += 2  # degrees
#     video_writer.write(draw_scene(yaw, tx, ty, tz))

# Move back in -x
for _ in range(100):
    tx -= 1
    video_writer.write(draw_scene(yaw, tx, ty, tz))

# Move back in -y
for _ in range(100):
    ty -= 1
    video_writer.write(draw_scene(yaw, tx, ty, tz))

# Rotate back -180°
# for _ in range(90):
#     yaw -= 2
#     video_writer.write(draw_scene(yaw, tx, ty, tz))

video_writer.release()
print("✅ Video saved as 'output_video.mp4'")