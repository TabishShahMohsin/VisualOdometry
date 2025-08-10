import cv2
import numpy as np
from scipy.optimize import minimize

def rotation_matrix(yaw):
    R_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    yaw_rad = np.radians(yaw)
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    return Rz @ R_c

def project_points(points_3d, R, t, K):
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv
    mask = points_cam[:, 2] > 0
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32), mask

def create_grid_lines(tile_w=10, tile_h=15, rows=8, cols=8):
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2
    lines = []
    
    # Horizontal lines
    for i in range(rows + 1):
        y = i * tile_h - grid_cy
        lines.append([[-grid_cx, y, 0], [grid_cx, y, 0]])
    
    # Vertical lines
    for j in range(cols + 1):
        x = j * tile_w - grid_cx
        lines.append([[x, -grid_cy, 0], [x, grid_cy, 0]])
    
    return np.array(lines)

def sample_line_segments(segments_2d, step_px=5):
    points = []
    for seg in segments_2d:
        p1, p2 = seg[0], seg[1]
        length = np.linalg.norm(p2 - p1)
        if length == 0:
            continue
        num_samples = max(2, int(length / step_px))
        t = np.linspace(0, 1, num_samples)
        x = p1[0] * (1 - t) + p2[0] * t
        y = p1[1] * (1 - t) + p2[1] * t
        points.extend(np.column_stack((x, y)))
    return np.array(points)

def enhanced_loss(params, edge_dt, K, grid_lines_3d, img_shape):
    tx, ty, tz, yaw = params
    R = rotation_matrix(yaw)
    t = np.array([tx, ty, tz])
    
    segs_2d = []
    visible_segments = 0
    total_dt = 0
    valid_points = 0
    
    # Max distance to consider (in pixels)
    MAX_DIST = 20.0
    
    for line in grid_lines_3d:
        pts_2d, mask = project_points(line, R, t, K)
        
        # Check if both points are in front of camera
        if not mask.all():
            continue
            
        # Check if at least one point is within image bounds
        in_bounds = False
        for pt in pts_2d:
            if 0 <= pt[0] < img_shape[1] and 0 <= pt[1] < img_shape[0]:
                in_bounds = True
                break
        if not in_bounds:
            continue
            
        segs_2d.append(pts_2d)
        visible_segments += 1
        
        # Sample points along the segment
        sampled_points = sample_line_segments([pts_2d])
        for pt in sampled_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                dt_val = edge_dt[y, x]
                # Clamp large distance values
                if dt_val > MAX_DIST:
                    dt_val = MAX_DIST
                total_dt += dt_val
                valid_points += 1
    
    # Penalty for insufficient visible segments
    if visible_segments < 5:  # Require at least 5 visible segments
        return 1e6
    
    # Penalty for too few sampled points
    if valid_points < 20:
        return 1e6
    
    # Calculate mean distance transform value
    mean_dt = total_dt / valid_points
    
    # Regularization terms
    z_reg = 0.01 * abs(tz - 100)  # Prefer z around 100cm
    yaw_reg = 0.001 * abs(yaw)     # Prefer small yaw angles
    
    # Weight by visibility (more segments = better)
    visibility_weight = visible_segments / len(grid_lines_3d)
    
    # Combine components
    loss = mean_dt / visibility_weight + z_reg + yaw_reg
    
    return loss

def main():
    # Load image and compute edge map
    img = cv2.imread('datasets/synthetic/pics/image_2.jpg')  # Replace with your image path
    if img is None:
        print("Error: Image not found")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    
    # Normalize distance transform for visualization
    dt_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('Distance Transform', dt_vis)
    
    # Camera intrinsics
    h, w = img.shape[:2]
    fx = fy = min(h, w) * 0.8  # Adjust based on image size
    cx, cy = w // 2, h // 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Create 3D grid lines
    grid_lines_3d = create_grid_lines()
    
    # Initial guess with reasonable constraints
    initial_params = [0, 0, 100, 0]  # [tx, ty, tz, yaw]
    
    # Set parameter bounds (x, y, z, yaw)
    bounds = [
        (-100, 100),   # tx: ±100 cm
        (-100, 100),   # ty: ±100 cm
        (50, 300),     # tz: 50-300 cm
        (-45, 45)      # yaw: ±45 degrees
    ]
    
    # Optimize with bounds
    result = minimize(
        enhanced_loss,
        initial_params,
        args=(dist_transform, K, grid_lines_3d, img.shape),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-4}
    )
    
    print("Optimization result:")
    print(f"Status: {result.message}")
    print(f"Parameters: tx={result.x[0]:.2f}, ty={result.x[1]:.2f}, tz={result.x[2]:.2f}, yaw={result.x[3]:.2f}")
    
    # Visualize result
    tx, ty, tz, yaw = result.x
    R = rotation_matrix(yaw)
    t = np.array([tx, ty, tz])
    
    # Draw grid lines
    for line in grid_lines_3d:
        pts_2d, mask = project_points(line, R, t, K)
        if mask.all():
            color = (0, 255, 0)  # Green for visible lines
            cv2.line(img, tuple(pts_2d[0]), tuple(pts_2d[1]), color, 2)
    
    # Draw origin
    origin_2d, visible = project_points(np.array([[0, 0, 0]]), R, t, K)
    if visible[0]:
        cv2.circle(img, tuple(origin_2d[0]), 8, (0, 0, 255), -1)
    
    # Display results
    cv2.putText(img, f"Z: {tz:.1f}cm", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"Yaw: {yaw:.1f}deg", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Edges', edges)
    cv2.imshow('Grid Projection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()