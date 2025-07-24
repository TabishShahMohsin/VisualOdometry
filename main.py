from hough_corner_detector import HoughCornerDetecter
import cv2
import numpy as np
from pnp_grid_ransac import ransac
import config
import matplotlib.pyplot as plt

HEIGHT = 15 # in centimeters
WIDTH = 10
fx = fy = 800
cx = cy = 400
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
distCoeffs = np.zeros((4, 1), dtype=np.float32) # Don't change this line, this was after hours of debugging


def run_hough_lines_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return

    processor = HoughCornerDetecter()
    
    trajectory = []
    last_R = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        intersections = processor.process_image(frame)
        if not intersections:
            continue

        result = ransac(intersections)
        if result is None:
            continue

        rvec, tvec, imagePoints, point = result
        
        # Convert rvec to a full rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # This is the pose of the WORLD coordinate system relative to the CAMERA
        # R_cam_world = R
        # t_cam_world = tvec

        # To get the camera's pose in the world, we need to invert this transformation
        # R_world_cam = R.T
        # t_world_cam = -R.T @ tvec

        # --- Pose Stabilization ---
        # solvePnP can be unstable and flip 180 degrees. We check for this.
        if last_R is not None:
            # Check if the z-axis of the new rotation is flipped compared to the last one
            if np.dot(R[:, 2], last_R[:, 2]) < 0:
                # If it's flipped, rotate it 180 degrees around its own z-axis
                R_flip = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                R = R @ R_flip
                tvec = R_flip @ tvec # Also correct the translation

        # Calculate camera position in world coordinates
        camera_position = -R.T @ tvec
        trajectory.append(camera_position.flatten())
        last_R = R
        
        # --- Visualization ---
        for x, y in intersections:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        if imagePoints is not None:
            for i in imagePoints:
                cv2.circle(frame, (int(i[0]), int(i[1])), 15, (0, 255, 0), -1)
        if point is not None:
            cv2.circle(frame, (point[0], point[1]), 7, (255, 0, 0), -1)

        cv2.imshow('Hough Corner Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return trajectory


def show_trajectory(trajectory):
    """
    Plots a 3D trajectory from a list of coordinates with a color gradient to show order.
    
    Args:
        trajectory (list): List of [x, y, z] points.
    """
    if not trajectory:
        print("Trajectory is empty.")
        return

    trajectory = np.array(trajectory)
    if trajectory.shape[1] != 3:
        print("Only 3D trajectories supported.")
        return

    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color gradient based on time/order
    N = len(x)
    colors = plt.cm.viridis(np.linspace(0, 1, N - 1))

    # Plot segments with color fading
    for i in range(N - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=2)

    # Mark start and end
    ax.scatter(x[0], y[0], z[0], color='blue', label='Start', s=100)
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='End', s=100)

    ax.set_title("3D Trajectory (Color = Time Order)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    traj = run_hough_lines_on_video('test4.mp4')  # Change to your video file path
    show_trajectory(traj)