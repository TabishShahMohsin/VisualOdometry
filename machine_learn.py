"""
Pose-estimation demo
–––––––––––––––––––
Render an artificial tiled-grid, compare it with an edge-image of a real grid,
and optimise the six camera pose parameters (roll, pitch, yaw, tx, ty, tz)
so the two grids align.  The objective function is a one-way Chamfer distance.
"""

import cv2
import numpy as np
from scipy.optimize import differential_evolution    # global optimiser
# --- CONFIG --------------------------------------------------------------- #

IMAGE_SIZE      = 800                 # square canvas, 800×800 px
FX = FY         = 800                 # focal length in pixels
CX = CY         = IMAGE_SIZE // 2     # principal point at image centre
TILE_W, TILE_H  = 5, 3                # physical tile size (cm)
ROWS, COLS      = 25, 15              # grid dimensions
DEBUG_WINDOWS   = False               # toggle GUI pop-ups

# --- MATH / GEOMETRY ------------------------------------------------------ #

def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a 3 × 3 rotation matrix from roll-pitch-yaw (°).
    Uses aerospace convention (XYZ → roll, pitch, yaw).  A final
    static correction matrix flips Y and Z so the virtual camera
    “looks down” like OpenCV’s conventional view.
    """
    # Fixed “look-down” correction.
    Rc = np.diag([ 1, -1, -1 ])

    # Convert to radians.
    roll, pitch, yaw = np.radians([roll, pitch, yaw])

    # Intrinsic rotations about camera axes.
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [ 0,             1, 0            ],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])

    # Final rotation:  R = Rz · Ry · Rx · Rc
    return Rz @ Ry @ Rx @ Rc


def project_points(points_3d: np.ndarray,
                   R: np.ndarray,
                   t: np.ndarray,
                   K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual alternative to cv2.projectPoints: transform 3-D world points
    into the camera frame, keep positive-Z points, and apply perspective
    division to get pixel coords.
    """
    Rt = R.T              # inverse rotation
    t_cam = -Rt @ t       # inverse translation

    points_cam = (Rt @ points_3d.T).T + t_cam          # 3-D in camera frame
    mask = points_cam[:, 2] > 1e-6                     # Z>0 → in front

    proj = (K @ points_cam.T).T                        # homogeneous pixels
    pts_2d = proj[:, :2] / proj[:, 2:3]                # divide by z
    return pts_2d.astype(np.int32), mask


# --- RENDERER ------------------------------------------------------------- #

def draw_grid(roll: float, pitch: float, yaw: float,
              tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Render a wireframe grid (as white edges on black background) given a
    camera pose.  Grid lies on the world-XY plane (Z = 0).
    """
    # Camera intrinsics matrix.
    K = np.array([[FX,  0, CX],
                  [ 0, FY, CY],
                  [ 0,  0,  1]])

    R = rotation_matrix(roll, pitch, yaw)
    t = np.array([tx, ty, tz])

    # Physical extent of the grid, centred at origin.
    grid_cx = COLS * TILE_W / 2
    grid_cy = ROWS * TILE_H / 2

    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)

    for r in range(ROWS):
        for c in range(COLS):
            # Four 3-D corners of one rectangle.
            x0 = c * TILE_W - grid_cx
            y0 = r * TILE_H - grid_cy
            quad = np.array([[x0,             y0,              0],
                             [x0 + TILE_W,    y0,              0],
                             [x0 + TILE_W,    y0 + TILE_H,     0],
                             [x0,             y0 + TILE_H,     0]])

            pts_2d, valid = project_points(quad, R, t, K)
            if valid.all():                                   # all corners visible
                cv2.polylines(canvas, [pts_2d], True, 255, 1) # white lines

    return canvas


# --- CHAMFER LOSS --------------------------------------------------------- #

def chamfer_loss(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    """
    One-way Chamfer loss: average *squared* distance from every white
    pixel in edge_a to the nearest white pixel in edge_b.
    Both inputs must already be 0/255 edge maps (e.g. Canny output).
    """
    if not np.any(edge_a) or not np.any(edge_b):
        return float('inf')  # degenerate case

    dt = cv2.distanceTransform(255 - edge_b, cv2.DIST_L2, 3)  # distance image
    return np.mean(dt[edge_a > 0] ** 2)


# --- OBJECTIVE FUNCTION -------------------------------------------------- #

def objective(pose: np.ndarray, target_edges: np.ndarray) -> float:
    """
    Wrapper for the optimiser:  pose → synthetic grid → edges → Chamfer loss.
    """
    roll, pitch, yaw, tx, ty, tz = pose
    synthetic = draw_grid(roll, pitch, yaw, tx, ty, tz)
    synthetic_edges = cv2.Canny(synthetic, 100, 200)   # edge map

    if DEBUG_WINDOWS:                                  # optional visual check
        cv2.imshow("synthetic", synthetic_edges)
        cv2.imshow("target", target_edges)
        cv2.waitKey(1)

    return chamfer_loss(synthetic_edges, target_edges) + chamfer_loss(target_edges, synthetic_edges)


# --- ENTRY-POINT --------------------------------------------------------- #

if __name__ == "__main__":

    # --- Load & preprocess the real image ------------------------------- #
    target = cv2.imread("original.jpeg", cv2.IMREAD_GRAYSCALE)
    target_edges = cv2.Canny(target, 100, 200)                           # edge map

    # --- Global search bounds (deg / cm) -------------------------------- #
    PARAM_BOUNDS = [(-10,  10),   # roll
                    (-10,  10),   # pitch
                    (-270, 270),  # yaw
                    (-10,  10),   # tx
                    (-10,  10),   # ty
                    (  1,  50)]   # tz (depth)

    # --- Run Differential Evolution global optimiser -------------------- #
    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        args=(target_edges,),
        maxiter=100,          # iterations per population ≈ evaluations/NP
        popsize=15,
        polish=True,          # local refine at the end
        disp=True             # progress in console
    )

    print("Optimised pose  :", result.x)
    print("Final loss      :", result.fun)

    # --- Visualise alignment -------------------------------------------- #
    aligned = draw_grid(*result.x)
    cv2.imshow("Aligned grid", aligned)
    cv2.imshow("Target edges", target_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()