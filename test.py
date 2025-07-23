import math
import random
import matplotlib.pyplot as plt

def get_8_closest_cyclic(target_point, point_list):
    x0, y0 = target_point

    # Step 1: Get 8 closest (excluding the point itself)
    closest = sorted(point_list, key=lambda p: math.hypot(p[0] - x0, p[1] - y0))[1:9]

    # Step 3: Sort by angle around centroid
    def angle(p):
        return math.atan2(p[1] - y0, p[0] - x0)
    
    closest.sort(key=angle)

    # Step 4: Find the point closest to the target to start the cycle
    min_idx = min(range(8), key=lambda i: (closest[i][0] - x0)**2 + (closest[i][1] - y0)**2)
    
    # Step 5: Rotate list to start from closest to target
    ordered = closest[min_idx:] + closest[:min_idx]

    return ordered

# ---- TEST CASE ----

def generate_noisy_grid(n_rows, n_cols, jitter=0.2):
    points = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j + random.uniform(-jitter, jitter)
            y = i + random.uniform(-jitter, jitter)
            points.append((x, y))
    return points

# Generate a 5x5 grid
grid_points = generate_noisy_grid(5, 5)

# Pick a point from the middle
target = grid_points[12]

# Run the function
ordered_points = get_8_closest_cyclic(target, grid_points)

# ---- VISUALIZE RESULT ----

plt.figure(figsize=(6, 6))
# Plot all points
xs, ys = zip(*grid_points)
plt.scatter(xs, ys, label="All Points", color='gray')

# Plot the target
plt.scatter(*target, color='red', label='Target', zorder=5)

# Plot ordered 8 neighbors
for i, p in enumerate(ordered_points):
    plt.scatter(*p, color='blue')
    plt.text(p[0]+0.1, p[1]+0.1, str(i), color='blue')
    plt.arrow(target[0], target[1], p[0] - target[0], p[1] - target[1],
              head_width=0.05, fc='black', ec='black', length_includes_head=True)

plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title("Cyclically Ordered 8 Nearest Points")
plt.show()