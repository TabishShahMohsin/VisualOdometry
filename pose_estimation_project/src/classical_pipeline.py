import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
import os

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

VIDEO_PATH = "dataset/crab_walk_video.mp4"
GT_CSV_PATH = "dataset/gt_trajectory.csv"
OUTPUT_PLOT_PATH = "dataset/classical_trajectory_comparison.png"
DEBUG_FRAME_DIR = "dataset/debug_frames"

# --- Grid Settings ---
TILE_WIDTH = 5.0
TILE_HEIGHT = 3.0

# --- Camera and Image Intrinsics ---
IMG_WIDTH = 800
IMG_HEIGHT = 800
K = np.array([[800, 0, IMG_WIDTH/2], [0, 800, IMG_HEIGHT/2], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))

# --- Processing Parameters ---
SAVE_DEBUG_FRAME_INTERVAL = 50

# ==============================================================================
#                       HELPER FUNCTIONS
# ==============================================================================

def get_intersections_from_lines(lines, min_dist=20):
    """Finds intersection points of horizontal and vertical lines."""
    horz_lines, vert_lines = [], []
    if lines is None: return None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            horz_lines.append(line[0])
        else:
            vert_lines.append(line[0])

    def cluster_lines(lines, is_horizontal=True):
        if not lines: return []
        lines = sorted(lines, key=lambda l: l[1 if is_horizontal else 0])
        clustered = [[lines[0]]]
        for line in lines[1:]:
            avg_pos = np.mean([l[1 if is_horizontal else 0] for l in clustered[-1]])
            if abs(line[1 if is_horizontal else 0] - avg_pos) < min_dist:
                clustered[-1].append(line)
            else:
                clustered.append([line])
        avg_lines = []
        for cluster in clustered:
            if is_horizontal:
                avg_y = np.mean([l[1] for l in cluster])
                avg_lines.append([0, avg_y, IMG_WIDTH, avg_y])
            else:
                avg_x = np.mean([l[0] for l in cluster])
                avg_lines.append([avg_x, 0, avg_x, IMG_HEIGHT])
        return np.array(avg_lines, dtype=np.int32)

    avg_horz = cluster_lines(horz_lines, is_horizontal=True)
    avg_vert = cluster_lines(vert_lines, is_horizontal=False)

    intersections = []
    for h_line in avg_horz:
        for v_line in avg_vert:
            x1, y1, x2, y2 = h_line
            x3, y3, x4, y4 = v_line
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom != 0:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersections.append([int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))])
    return np.array(intersections, dtype=np.float32) if intersections else None

# ==============================================================================
#                         CLASSICAL PIPELINE SCRIPT
# ==============================================================================

def main():
    print("Starting Final Classical Pipeline...")
    os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    gt_trajectory_df = pd.read_csv(GT_CSV_PATH)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {num_frames} frames...")

    reconstructed_trajectory = [np.array([0.0, 0.0, 0.0])]
    last_t_ocs = None
    frames_processed = 0

    for frame_idx in tqdm(range(num_frames), desc="Analyzing Video"):
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            intersections = get_intersections_from_lines(lines)
            if intersections is not None and len(intersections) >= 4:
                try:
                    tl_point = min(intersections, key=lambda p: np.linalg.norm(p))
                    candidates = [p for p in intersections if not np.array_equal(p, tl_point)]
                    if len(candidates) < 3: continue

                    est_w = TILE_WIDTH / (last_t_ocs[2] if last_t_ocs is not None else 50) * K[0,0]
                    est_h = TILE_HEIGHT / (last_t_ocs[2] if last_t_ocs is not None else 50) * K[1,1]

                    tr = min(candidates, key=lambda p: np.linalg.norm(p - (tl_point + [est_w, 0])))
                    bl = min(candidates, key=lambda p: np.linalg.norm(p - (tl_point + [0, est_h])))
                    br_expected = tl_point + (tr - tl_point) + (bl - tl_point)
                    br = min(candidates, key=lambda p: np.linalg.norm(p - br_expected))

                    image_points = np.array([tl_point, tr, bl, br], dtype=np.float32)
                    objp = np.array([[0,0,0], [TILE_WIDTH,0,0], [0,TILE_HEIGHT,0], [TILE_WIDTH,TILE_HEIGHT,0]], dtype=np.float32)

                    _, rvec, tvec = cv2.solvePnP(objp, image_points, K, DIST_COEFFS)
                    current_t_ocs = tvec.flatten()

                    if last_t_ocs is not None:
                        delta_t_ocs = current_t_ocs - last_t_ocs
                        tile_shift_x = np.round(delta_t_ocs[0] / TILE_WIDTH)
                        tile_shift_y = np.round(delta_t_ocs[1] / TILE_HEIGHT)
                        delta_t_wcs_x = delta_t_ocs[0] - tile_shift_x * TILE_WIDTH
                        delta_t_wcs_y = delta_t_ocs[1] - tile_shift_y * TILE_HEIGHT
                        
                        last_point = reconstructed_trajectory[-1]
                        new_point = last_point + np.array([delta_t_wcs_x, delta_t_wcs_y, 0])
                        reconstructed_trajectory.append(new_point)
                    
                    last_t_ocs = current_t_ocs
                    frames_processed += 1

                    if frame_idx % SAVE_DEBUG_FRAME_INTERVAL == 0:
                        debug_frame = frame.copy()
                        for point in image_points:
                            cv2.circle(debug_frame, tuple(point.astype(int)), 5, (0,0,255), -1)
                        cv2.imwrite(os.path.join(DEBUG_FRAME_DIR, f"frame_{frame_idx}.png"), debug_frame)
                except (ValueError, cv2.error):
                    continue

    cap.release()

    if frames_processed < 2: 
        print("Could not process enough frames.")
        return

    reconstructed_trajectory = np.array(reconstructed_trajectory)
    gt_traj = gt_trajectory_df[['x', 'y']].values
    gt_traj -= gt_traj[0, :]
    reconstructed_trajectory -= reconstructed_trajectory[0, :]

    plt.figure(figsize=(10, 10))
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], label='Ground Truth Trajectory', color='g', linewidth=3, alpha=0.8)
    plt.plot(reconstructed_trajectory[:, 0], reconstructed_trajectory[:, 1], label='Reconstructed Trajectory (Classical)', color='b', linestyle='--', marker='o', markersize=4)
    plt.title('Camera Trajectory Reconstruction (Final Classical Method)')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"Trajectory plot saved to {OUTPUT_PLOT_PATH}")

if __name__ == '__main__':
    main()
