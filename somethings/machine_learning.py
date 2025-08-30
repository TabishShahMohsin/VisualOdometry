import cv2
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
# IN chaining in video loop we would need to use the prev values to init the next values


def rotation_matrix(roll, pitch, yaw):
    # Correction for looking down
    Rc = np.array([
            [1, 0,0],
            [0, -1, 0],
            [0, 0, -1],
    ])
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array(
        [[1, 0, 0],
         [0, np.cos(roll), -np.sin(roll)],
         [0, np.sin(roll), np.cos(roll)]
    ])
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
    return Rz @ Ry @ Rx @ Rc


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
    rows, cols = 25, 15 
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2

    img = np.zeros((800, 800), dtype=np.uint8)

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
                cv2.polylines(img, [pts_2d], True, 255, 1)

    return img

def loss_function(pose, img2: np.array) -> float:    
    roll, pitch, yaw, tx, ty, tz = pose
    img1 = draw_scene(roll, pitch, yaw, tx, ty, tz)
    # lines1 = cv2.HoughLines(img1, 1, np.pi / 180, 100)
    # lines2 = cv2.HoughLines(img2, 1, np.pi / 180, 100)
    cv2.imshow('something', img1)
    cv2.waitKey(0)
    cv2.imshow('something', img2)
    cv2.waitKey(0)
    def chamfer_loss(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """
        Computes Chamfer Distance from edges in img_a to nearest edges in img_b.
        Both inputs must be binary edge maps (0 or 255), e.g., after cv2.Canny.
        Returns mean squared distance.
        """
        # Ensure binary format (0/255)
        edges_a = (img_a > 0).astype(np.uint8) * 255
        edges_b = (img_b > 0).astype(np.uint8) * 255

        # Check for empty edge maps
        if np.count_nonzero(edges_a) == 0 or np.count_nonzero(edges_b) == 0:
            return float('inf')

        # Distance transform of the inverse of img_b:
        # each pixel holds distance to nearest edge in img_b
        dt = cv2.distanceTransform(255 - edges_b, cv2.DIST_L2, 3)

        # Sample distances at edge locations in img_a
        distances = dt[edges_a > 0]

        # Return mean squared distance
        return np.mean(distances ** 2)
    return chamfer_loss(img2, img1)
    


roll, pitch, yaw, tx, ty, tz = 0, 0, 90, 0, 0, 15
img = cv2.imread('original.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 100, 200)

init_pose = np.array([0, 0, 0, 0, 0, 20]) 
print(loss_function(init_pose, img))

bounds = [
        (-10, 10),    # roll
        (-10, 10),    # pitch
        (-270, 270),  # yaw
        (-10, 10),  # tx
        (-10, 10),  # ty
        (1, 50)     # tz
    ]


result = differential_evolution(
    loss_function,
    bounds=bounds,
    args=(img,),  # your image as fixed argument
    maxiter=1,
    popsize=15,
    polish=True  # optional: refine with local search at the end
)

# result = dual_annealing(
#     func=loss_function,
#     bounds=bounds,
#     args=(img,),  # pass extra args to loss function
#     maxiter=30
# )

optimized_pose = result.x
print(optimized_pose)