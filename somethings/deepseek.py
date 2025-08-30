import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def circular_mean(angles):
    """Compute circular mean for angles in [0, π)"""
    # FIX: This function is correct and essential for averaging angles.
    double_angles = 2 * np.array(angles)
    C = np.mean(np.cos(double_angles))
    S = np.mean(np.sin(double_angles))
    mean_double = np.arctan2(S, C)
    return (mean_double / 2) % np.pi

def merge_similar_lines(lines, rho_threshold=5, angle_threshold=np.pi/180 * 5):
    """Merge lines that are very close in (rho, theta)"""
    if lines is None or len(lines) == 0:
        return np.array([])
        
    lines = np.array(lines)
    # Ensure all thetas are in [0, pi) for consistency
    lines[:, 0][lines[:, 0] < 0] *= -1
    lines[:, 1][lines[:, 0] < 0] = (lines[:, 1][lines[:, 0] < 0] + np.pi) % np.pi
    lines[:, 1] = lines[:, 1] % np.pi

    used = np.zeros(len(lines), dtype=bool)
    merged_lines = []
    
    for i in range(len(lines)):
        if used[i]:
            continue
            
        rho_i, theta_i = lines[i]
        group_indices = [i]
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
                
            rho_j, theta_j = lines[j]
            angle_diff = abs(theta_i - theta_j)
            # Handle angle wrap-around at 0 and pi
            angle_diff = min(angle_diff, np.pi - angle_diff)
            
            if angle_diff < angle_threshold and abs(rho_i - rho_j) < rho_threshold:
                group_indices.append(j)
        
        group_lines = lines[group_indices]
        avg_rho = np.mean(group_lines[:, 0])
        # FIX: Use circular mean for angles instead of a simple arithmetic mean.
        avg_theta = circular_mean(group_lines[:, 1])
        merged_lines.append([avg_rho, avg_theta])

        # Mark all lines in the group as used
        for idx in group_indices:
            used[idx] = True
            
    return np.array(merged_lines)

def fit_cluster(rhos, min_spacing=5.0):
    """Fit rhos to linear model: rho = offset + k*spacing
    Returns (offset, spacing)"""
    rhos = np.array(rhos)
    if len(rhos) < 2:
        return (np.mean(rhos), 0.0) if len(rhos) == 1 else (0.0, 0.0)

    rhos = np.sort(rhos)
    diffs = np.diff(rhos)
    
    # Estimate spacing as the median of significant differences
    valid_diffs = diffs[diffs > min_spacing]
    if len(valid_diffs) == 0:
        # Fallback if no diffs are large enough
        est_spacing = (rhos[-1] - rhos[0]) / (len(rhos) - 1) if len(rhos) > 1 else 0
    else:
        est_spacing = np.median(valid_diffs)
    
    if est_spacing < min_spacing:
        return np.mean(rhos), 0.0

    # Estimate integer indices `k` for each line
    k = np.round((rhos - rhos[0]) / est_spacing).astype(int)
    
    # Refine offset and spacing with a linear fit
    A = np.vstack([np.ones_like(k), k]).T
    try:
        offset, spacing = np.linalg.lstsq(A, rhos, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.mean(rhos), 0.0

    spacing_abs = max(abs(spacing), min_spacing)
    
    # FIX: Adjust offset to be in [-spacing/2, spacing/2] using a more concise formula.
    offset_adjusted = (offset + spacing_abs / 2) % spacing_abs - spacing_abs / 2
    
    return offset_adjusted, spacing_abs

def show(img: np.array, title="Untitled") -> None:
    cv2.imshow(title, img)
    cv2.waitKey(0)

def detect_grid_parameters(image_path, min_spacing=5.0):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    print(f"Image resolution: {img.shape}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use CLAHE for better contrast in varied lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    show(enhanced)
    edges = cv2.Canny(enhanced, 20, 60, apertureSize=3)
    show(edges)

    # First attempt with standard Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    if lines is not None:
        lines = lines[:, 0, :]
    else: # Fallback to Probabilistic Hough Transform if standard fails
        # lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
        #                         minLineLength=50, maxLineGap=10)
        lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                        minLineLength=300, maxLineGap=50)
        if lines_p is None:
            raise ValueError("No lines detected - adjust Hough parameters")
            
        # FIX: Correctly convert (x1,y1,x2,y2) to (rho, theta)
        infinite_lines = []
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            # Angle of the normal vector to the line
            theta = np.arctan2(-dx, dy) + np.pi / 2
            # Perpendicular distance from origin to the line
            rho = x1 * np.cos(theta) + y1 * np.sin(theta)
            
            # Normalize to OpenCV's convention (rho >= 0, theta in [0, pi))
            if rho < 0:
                rho *= -1
                theta = (theta - np.pi)
            
            theta = theta % np.pi
            infinite_lines.append([rho, theta])
        lines = np.array(infinite_lines)

    # Merge nearby lines to get a cleaner set
    merged_lines = merge_similar_lines(lines, rho_threshold=10, angle_threshold=np.pi/180 * 5)
    
    if len(merged_lines) < 4:
        raise ValueError(f"Only {len(merged_lines)} lines remain after merging - insufficient for grid")

    # Cluster lines into two orthogonal families using K-Means
    angles = merged_lines[:, 1]
    # Map angles to points on a circle to cluster them
    X = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)])
    
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(X)
    labels = kmeans.labels_
    
    cluster1 = merged_lines[labels == 0]
    cluster2 = merged_lines[labels == 1]
    
    if len(cluster1) < 2 or len(cluster2) < 2:
        raise ValueError(f"Insufficient lines in clusters: {len(cluster1)} and {len(cluster2)}")

    # FIX: Calculate parameters for each cluster independently
    theta1 = circular_mean(cluster1[:, 1])
    c1, p = fit_cluster(cluster1[:, 0], min_spacing)
    
    theta2 = circular_mean(cluster2[:, 1])
    c2, q = fit_cluster(cluster2[:, 0], min_spacing)
    
    # Sanity check the detected parameters
    angle_diff = abs(theta1 - theta2)
    angular_separation = min(angle_diff, np.pi - angle_diff)
    
    if not np.isclose(angular_separation, np.pi/2, atol=0.2): # Allow ~11.5 degrees of skew
        print(f"Warning: Grid may not be orthogonal. Angular separation = {np.rad2deg(angular_separation):.1f}°")
    
    if p < min_spacing or q < min_spacing:
        raise ValueError(f"Invalid spacing detected: p={p:.2f}, q={q:.2f}")

    # FIX: Return independent angles for each grid direction
    return theta1, c1, p, theta2, c2, q, img

def draw_grid_lines(image, theta1, c1, p, theta2, c2, q):
    """Draw grid lines on the image using the fitted parameters"""
    img_with_grid = image.copy()
    height, width = image.shape[:2]
    
    # Define colors for the two line sets
    color1, color2 = (0, 0, 255), (0, 255, 0) # Red, Green
    
    # Function to draw a set of parallel lines
    def draw_line_set(theta, offset, spacing, color):
        if spacing <= 0: return
        max_rho = np.sqrt(width**2 + height**2)
        # Determine how many lines to draw to cover the image
        line_range = int(np.ceil(max_rho / spacing)) + 2

        for k in range(-line_range, line_range + 1):
            rho = offset + k * spacing
            a, b = np.cos(theta), np.sin(theta)
            # Find two points on the line far outside the image
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 2 * max_rho * (-b)), int(y0 + 2 * max_rho * (a)))
            pt2 = (int(x0 - 2 * max_rho * (-b)), int(y0 - 2 * max_rho * (a)))
            
            # FIX: Use cv2.clipLine to robustly clip the line to the image rectangle.
            clipped, p1_clipped, p2_clipped = cv2.clipLine((0, 0, width, height), pt1, pt2)
            if clipped:
                cv2.line(img_with_grid, p1_clipped, p2_clipped, color, 1, cv2.LINE_AA)

    # FIX: Draw each set of lines using its own fitted angle (theta1, theta2)
    draw_line_set(theta1, c1, p, color1)
    draw_line_set(theta2, c2, q, color2)
    
    return img_with_grid

if __name__ == "__main__":
    import sys
    # Provide a default image path for convenience
    # image_path = "datasets/synthetic/pics/image_5.jpg" # <--- CHANGE THIS TO YOUR IMAGE PATH
    # image_path = "datasets/synthetic/pics/image_417.jpg" # <--- CHANGE THIS TO YOUR IMAGE PATH
    image_path = "datasets/meston_pool/pics/7130.jpg" # <--- CHANGE THIS TO YOUR IMAGE PATH
    # image_path = sys.argv[1] if len(sys.argv) > 1 else "grid_image.png"
    # min_spacing = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    try:
        # FIX: Unpack the new return values including separate angles
        theta1, c1, p, theta2, c2, q, img = detect_grid_parameters(image_path, min_spacing=10.0)
        
        print("Grid parameters extracted:")
        print(f"Set 1: θ₁={np.rad2deg(theta1):.1f}°, c₁={c1:.2f}, p={p:.2f}")
        print(f"Set 2: θ₂={np.rad2deg(theta2):.1f}°, c₂={c2:.2f}, q={q:.2f}")
        
        # Draw the detected grid on the image
        img_with_grid = draw_grid_lines(img, theta1, c1, p, theta2, c2, q)
        
        # Display the results
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_with_grid, cv2.COLOR_BGR2RGB))
        plt.title("Detected Grid Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("grid_comparison.png")
        print("\nComparison image saved as 'grid_comparison.png'")
        plt.show()
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        print("Try adjusting parameters in detect_grid_parameters(), such as:")
        print("- Hough transform thresholds")
        print("- Canny edge detection thresholds")
        print("- min_spacing value")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")