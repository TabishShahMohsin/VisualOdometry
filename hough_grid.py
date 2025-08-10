import cv2
import numpy as np
import time
from numba import njit
from multiprocessing import Pool, cpu_count

# ==============================================================================
# OPTIMIZED CORE FUNCTIONS (NUMBA JIT)
# ==============================================================================

@njit(fastmath=True, cache=True)
def rotation_matrix_numba(roll, pitch, yaw):
    """
    Numba-optimized version of the rotation matrix calculation.
    """
    # Correction for a downward-looking camera
    R_c = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])

    roll_rad, pitch_rad, yaw_rad = np.radians(np.array([roll, pitch, yaw]))

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0.0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    Ry = np.array([
        [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)]
    ])
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return Rz @ Ry @ Rx @ R_c

@njit(fastmath=True, cache=True)
def project_points_numba(points_3d, R, t, K):
    """
    Numba-optimized batch projection of 3D points to 2D.
    Processes all points in a single vectorized operation.
    """
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv

    # Use a 1D array for z-coordinates to work around Numba's indexing limitations
    z = points_cam[:, 2]
    validity_mask = z > 1e-6 # Points must be in front of camera
    
    # Create a copy to modify for safe division
    z_safe = z.copy()
    z_safe[z_safe <= 1e-6] = 1.0 # Avoid division by zero for invalid points

    points_proj = (K @ points_cam.T).T
    # Reshape z_safe to a column vector (n, 1) for broadcasting during division
    points_2d = points_proj[:, :2] / z_safe.reshape(-1, 1)
    
    return points_2d.astype(np.int32), validity_mask


@njit(fastmath=True, cache=True)
def draw_line_numba(img, x0, y0, x1, y1):
    """
    Numba-compatible Bresenham's line algorithm.
    """
    h, w = img.shape
    if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
        return # Skip lines where at least one endpoint is out of bounds.

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < h and 0 <= x0 < w:
            img[y0, x0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

@njit(fastmath=True, cache=True)
def score_pose_numba(params, K, edges, grid_points_3d):
    """
    Calculates the score for a single pose using vectorized operations.
    """
    roll, pitch, yaw, tx, ty, tz = params
    R = rotation_matrix_numba(roll, pitch, yaw)
    t = np.array([tx, ty, tz], dtype=np.float64)
    
    all_2d_points, valid_mask = project_points_numba(grid_points_3d, R, t, K)
    
    synthetic_grid = np.zeros(edges.shape, dtype=np.uint8)
    num_lines = len(all_2d_points) // 2
    for i in range(num_lines):
        p1_idx, p2_idx = 2 * i, 2 * i + 1
        if valid_mask[p1_idx] and valid_mask[p2_idx]:
            p1 = all_2d_points[p1_idx]
            p2 = all_2d_points[p2_idx]
            draw_line_numba(synthetic_grid, p1[0], p1[1], p2[0], p2[1])
            
    return np.sum(np.bitwise_and(synthetic_grid, edges))

@njit(fastmath=True, cache=True)
def generate_grid_points():
    """Generates the 3D coordinates for all grid line endpoints."""
    tile_w, tile_h = 5.0, 3.0
    rows, cols = 8, 8
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2
    
    num_h_lines = rows + 1
    num_v_lines = cols + 1
    total_lines = num_h_lines + num_v_lines
    all_3d_points = np.empty((2 * total_lines, 3), dtype=np.float64)

    y_coords = np.linspace(-grid_cy, grid_cy, num_h_lines)
    all_3d_points[:2*num_h_lines:2, 0] = -grid_cx
    all_3d_points[1:2*num_h_lines:2, 0] = grid_cx
    all_3d_points[:2*num_h_lines, 1] = np.repeat(y_coords, 2)
    
    x_coords = np.linspace(-grid_cx, grid_cx, num_v_lines)
    start_idx = 2 * num_h_lines
    all_3d_points[start_idx:, 0] = np.repeat(x_coords, 2)
    all_3d_points[start_idx:start_idx+2*num_v_lines:2, 1] = -grid_cy
    all_3d_points[start_idx+1:start_idx+2*num_v_lines:2, 1] = grid_cy
    all_3d_points[:, 2] = 0.0
    return all_3d_points

# ==============================================================================
# MULTIPROCESSING WORKER SETUP
# ==============================================================================
# Global variables are used to avoid passing large data to each worker process
g_edges = None
g_K = None
g_grid_points_3d = None

def init_worker(edges, K, grid_points_3d):
    """Initializes global variables for each worker process."""
    global g_edges, g_K, g_grid_points_3d
    g_edges = edges
    g_K = K
    g_grid_points_3d = grid_points_3d

def process_param_chunk(params):
    """Worker function for multiprocessing. Calculates score for one pose."""
    return score_pose_numba(params, g_K, g_edges, g_grid_points_3d)

# ==============================================================================
# MAIN SEARCH ALGORITHM
# ==============================================================================

def find_pose_from_grid_optimized(image):
    """
    Estimates camera pose using an iterative, probabilistic search.
    """
    # --- Setup ---
    fx = fy = 800.0
    cx = cy = 400.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    grid_points_3d = generate_grid_points()

    # --- Iterative Refinement Search Parameters ---
    num_generations = 3
    samples_per_gen = 20000
    top_k_to_keep = 100
    
    # initial_bounds = {
    #     'roll': (-4, 4), 'pitch': (-4, 4), 'yaw': (0, 90),
    #     'tx': (0, 5), 'ty': (0, 3), 'tz': (25, 150)
    # }
    initial_bounds = {
        'roll': (-0, 0), 'pitch': (-0, 0), 'yaw': (0, 90),
        'tx': (0, 5), 'ty': (0, 3), 'tz': (25, 150)
    }
    param_keys = list(initial_bounds.keys())
    current_bounds = initial_bounds.copy()
    
    best_overall_params = None
    num_processes = cpu_count()
    print(f"--- Starting Iterative Search ---\nUsing {num_processes} processes, {num_generations} generations, {samples_per_gen} samples/gen.")

    # --- Main Search Loop ---
    with Pool(processes=num_processes, initializer=init_worker, initargs=(edges, K, grid_points_3d)) as pool:
        for gen in range(num_generations):
            print(f"\n--- Generation {gen + 1}/{num_generations} ---")

            samples = [np.random.uniform(low, high, samples_per_gen) for low, high in current_bounds.values()]
            param_samples = list(zip(*samples))

            scores = pool.map(process_param_chunk, param_samples)
            scored_samples = sorted(zip(scores, param_samples), key=lambda x: x[0], reverse=True)
            
            if not scored_samples or scored_samples[0][0] == 0:
                print("Warning: No matching candidates found. Stopping search.")
                break

            top_candidates = [s[1] for s in scored_samples[:top_k_to_keep]]
            best_overall_params = scored_samples[0][1]
            
            print(f"Best score in Gen {gen+1}: {scored_samples[0][0]}")
            print(f"Best params: {np.round(np.array(best_overall_params), 2)}")

            if gen == num_generations - 1: break

            # --- Refine search space for the next generation ---
            top_array = np.array(top_candidates)
            mean_params = np.mean(top_array, axis=0)
            # Add a small epsilon for robustness against zero standard deviation
            std_params = np.std(top_array, axis=0) + 1e-6 

            factor = 2.0 
            for i, key in enumerate(param_keys):
                low = mean_params[i] - factor * std_params[i]
                high = mean_params[i] + factor * std_params[i]
                original_low, original_high = initial_bounds[key]
                current_bounds[key] = (max(low, original_low), min(high, original_high))
            print("Refined search bounds for next generation.")

    return best_overall_params

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def draw_grid_scene_optimized(params, K, img_size=(800, 800)):
    """Draws the final grid using the same optimized, vectorized functions."""
    img = np.zeros(img_size, dtype=np.uint8)
    grid_points_3d = generate_grid_points()
    
    roll, pitch, yaw, tx, ty, tz = params
    R = rotation_matrix_numba(roll, pitch, yaw)
    t = np.array([tx, ty, tz], dtype=np.float64)
    
    all_2d_points, valid_mask = project_points_numba(grid_points_3d, R, t, K)
    
    num_lines = len(all_2d_points) // 2
    for i in range(num_lines):
        p1_idx, p2_idx = 2 * i, 2 * i + 1
        if valid_mask[p1_idx] and valid_mask[p2_idx]:
            p1 = tuple(all_2d_points[p1_idx])
            p2 = tuple(all_2d_points[p2_idx])
            cv2.line(img, p1, p2, 255, 1) # Use cv2.line for anti-aliasing in final render
    return img

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    # --- Generate a sample image with a known pose ---
    # known_params = (2, 2, 45, 2, 1, 100) # (roll, pitch, yaw, tx, ty, tz)
    known_params = (0, 0, 35, 2, 1, 70) # (roll, pitch, yaw, tx, ty, tz)
    K_gen = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float64)
    
    sample_image = np.ones((800, 800, 3), dtype=np.uint8) * 255
    grid_to_draw = draw_grid_scene_optimized(known_params, K_gen)
    sample_image[grid_to_draw == 255] = [0, 0, 0]

    cv2.imshow("Input Grid Image", sample_image)
    cv2.waitKey(1)

    # --- Find the pose using the optimized method ---
    start_time = time.time()
    estimated_params = find_pose_from_grid_optimized(sample_image)
    end_time = time.time()
    print(f"\nTotal estimation time: {end_time - start_time:.2f} seconds")

    if estimated_params:
        print("\n--- Estimated Pose (Optimized) ---")
        print(f"Roll: {estimated_params[0]:.2f}, Pitch: {estimated_params[1]:.2f}, Yaw: {estimated_params[2]:.2f}")
        print(f"Tx: {estimated_params[3]:.2f}, Ty: {estimated_params[4]:.2f}, Tz: {estimated_params[5]:.2f}")

        # --- Visualize the result ---
        overlay_grid = draw_grid_scene_optimized(estimated_params, K_gen)
        overlay_grid_colored = cv2.cvtColor(overlay_grid, cv2.COLOR_GRAY2BGR)
        overlay_grid_colored[np.where((overlay_grid_colored == [255,255,255]).all(axis=2))] = [0,0,255] # Red
        result_image = cv2.addWeighted(sample_image, 0.7, overlay_grid_colored, 0.3, 0)
        
        cv2.imshow("Result (Original + Estimated Grid)", result_image)
    else:
        print("\nCould not estimate the pose.")

    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
