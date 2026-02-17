import cv2
import numpy as np


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


def draw_scene(roll, pitch, yaw, tx, ty, tz):
    # Camera intrinsics
    fx = fy = 800
    cx = cy = 400
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Rotation and translation of the camera
    R = rotation_matrix(roll, pitch, yaw)
    t = np.array([tx, ty, tz])

    # Grid setup
    tile_w, tile_h = 5, 3  # cm
    rows, cols = 8, 8
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2

    img = np.ones((800, 800, 3), dtype=np.uint8) * 255

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
            if mask.all():
                cv2.polylines(img, [pts_2d], True, (0, 0, 0), 1)

    # Draw the origin marker if visible
    origin_2d, visible = project_points(np.array([[0, 0, 0]]), R, t, K)
    if visible[0]:
        cv2.circle(img, tuple(origin_2d[0]), 5, (0, 0, 255), -1)
        cv2.putText(
            img,
            "Origin",
            tuple(origin_2d[0] + [10, -10]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    return img


# GUI update callback
def update(_=None):
    roll = cv2.getTrackbarPos("Roll", win) - 180
    pitch = cv2.getTrackbarPos("Pitch", win) - 180
    yaw = cv2.getTrackbarPos("Yaw", win) - 180
    tx = cv2.getTrackbarPos("Tx", win) - 100
    ty = cv2.getTrackbarPos("Ty", win) - 100
    tz = cv2.getTrackbarPos("Tz", win)

    img = draw_scene(roll, pitch, yaw, tx, ty, tz)
    cv2.imshow(win, img)
    # cv2.imwrite("sample.png", img)


# Create window and trackbars
win = "Camera View Explorer"
cv2.namedWindow(win)

cv2.createTrackbar("Roll", win, 180, 360, update)
cv2.createTrackbar("Pitch", win, 180, 360, update)
cv2.createTrackbar("Yaw", win, 180, 360, update)
cv2.createTrackbar("Tx", win, 100, 200, update)
cv2.createTrackbar("Ty", win, 100, 200, update)
cv2.createTrackbar("Tz", win, 50, 300, update)

update()  # Draw first frame

# Main loop
while True:
    if cv2.waitKey(30) == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
