from hough_corner_detector import HoughCornerDetecter
import cv2
import numpy as np
from pnp_grid_ransac import ransac
from config import HEIGHT, WIDTH, K, distCoeffs
import config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os



def run_hough_lines_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return

    processor = HoughCornerDetecter()

    initial_pose = np.eye(4)
    trajectory = [initial_pose]

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        intersections = processor.process_image(frame)
        for x, y in intersections:
            cv2.circle(frame, (x, y), 9, (0, 0, 255), -1)

        rvec, tvec, imagePoints, point = ransac(intersections)
        # print("ccs_Yaw:", np.rad2deg(rvec)[2])
        # print("ccs_roll:", np.rad2deg(rvec)[0])
        # print("ccs_pitch:", np.rad2deg(rvec)[1])
        # rvec = wrap_centered(rvec, np.pi)

        R, _ = cv2.Rodrigues(rvec)

        # if wrap_centered(rvec, np.pi / 2) > :
        # print(rvec)
        c_pose = np.eye(4)
        c_pose[:3, :3] = R
        c_pose[:3, 3] = tvec.ravel()

        pose = np.linalg.inv(c_pose)
        
        r_ocs = cv2.Rodrigues(pose[:3, :3])[0]
        if r_ocs[2] > np.pi / 2:
            flip_xy = np.array([
                [-1,  0,  0, 0],
                [ 0, -1,  0, 0],
                [ 0,  0,  1, 0],
                [ 0,  0,  0, 1]
            ])
            corrected_pose = flip_xy @ pose
        else: corrected_pose = pose

        # print("x:", wrap_centered(pose[0][3], HEIGHT))

        # delta = find_delta(pose, trajectory[-1])
        delta = find_delta(corrected_pose, trajectory[-1])

        # trajectory.append(trajectory[-1] @ delta)
        trajectory.append(delta)

        for i, pt in enumerate(imagePoints):
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 7, (0, 255, 0), -1)
        cv2.circle(frame, (point[0], point[1]), 5, (255, 0, 0), -1)
        cv2.imshow('Hough Corner Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break
    cap.release()
    cv2.destroyAllWindows()
    return trajectory

def find_delta(final_pose_ocs, initial_pose) -> np.array:
    # 1. Calculate the relative motion between local poses
    # This delta's translation might have a large jump if we crossed a tile boundary
    # delta_pose = np.linalg.inv(initial_pose) @ final_pose_ocs
    
    # Uncorrected new pose
    # uncorrected_new_pose = initial_pose @ delta_pose


    # relative_pose = np.linalg.inv(initial_pose) @ final_pose_ocs
    # # Wrap translation components
    # relative_pose[0, 3] = wrap_centered(relative_pose[0, 3], WIDTH)
    # relative_pose[1, 3] = wrap_centered(relative_pose[1, 3], HEIGHT)
    # return relative_pose

    world_displacement = final_pose_ocs[:3, 3] - initial_pose[:3, 3]

    world_displacement[0] = wrap_centered(world_displacement[0], WIDTH)
    world_displacement[1] = wrap_centered(world_displacement[1], HEIGHT)

    corrected_position = initial_pose[:3, 3] + world_displacement
    print("Initial Pose:", np.rad2deg(cv2.Rodrigues(initial_pose[:3, :3])[0]))
    print("Final Pose:", np.rad2deg(cv2.Rodrigues(final_pose_ocs[:3, :3])[0]))
    print("World displacement:", np.round(world_displacement, 2))

    final_new_pose = np.copy(final_pose_ocs)
    final_new_pose[:3, 3] = corrected_position
    return final_new_pose


    
    # 2. Get the rotation part of the delta
    R_delta = delta_pose[:3, :3]
    
    # 3. Get the translation part and wrap it correctly
    # This is the key step: wrap the raw translation delta
    t_delta = delta_pose[:3, 3]
    t_delta_wrapped_x = wrap_centered(t_delta[0], WIDTH)
    t_delta_wrapped_y = wrap_centered(t_delta[1], HEIGHT)
    
    # 4. Reconstruct the corrected delta_pose
    corrected_delta_pose = np.eye(4)
    corrected_delta_pose[:3, :3] = R_delta
    corrected_delta_pose[0, 3] = t_delta_wrapped_x
    corrected_delta_pose[1, 3] = t_delta_wrapped_y
    corrected_delta_pose[2, 3] = t_delta[2] # Z is not wrapped

    return corrected_delta_pose

    # Making z +ve
    # sign_i = 1 if initial_pose[2][3] > 0 else -1
    # initial_pose = initial_pose * sign_i
    # sign_f = 1 if final_pose[2][3] > 0 else -1
    # final_pose = final_pose * sign_f
    delta_pose = np.linalg.inv(initial_pose) @ final_pose
    delta_pose[0][3] = wrap_centered(delta_pose[0][3], WIDTH)
    delta_pose[1][3] = wrap_centered(delta_pose[1][3], HEIGHT)
    return delta_pose

    
def wrap_centered(value, max_value):
    return ((value + max_value / 2) % max_value) - max_value / 2


def show_trajectory(trajectory):
    """
    Plots a 3D trajectory from a list of coordinates with a color gradient to show order.
    
    Args:
        trajectory (list): List of [x, y, z] points.
    """
    trajectory = np.array(trajectory)

    # x, y, z = trajectory[0, 3], trajectory[1, 3], trajectory[1, 3]
    x = [pose[0, 3] for pose in trajectory]
    y = [pose[1, 3] for pose in trajectory]
    z = [pose[2, 3] for pose in trajectory]

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

def visualize_trajectory_video(
    poses: list[np.ndarray],
    axis_length: float = 10,
    output_path: str = "camera_trajectory.mp4",
    fps: int = 15
):
    if not poses:
        print("Warning: The provided list of poses is empty.")
        return

    # Prepare trajectory path
    path = np.array([p[:3, 3] for p in poses])
    x_min, x_max = np.min(path[:, 0]), np.max(path[:, 0])
    y_min, y_max = np.min(path[:, 1]), np.max(path[:, 1])
    z_min, z_max = np.min(path[:, 2]), np.max(path[:, 2])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fix axes for consistent animation
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Camera Flythrough Animation")

    line, = ax.plot([], [], [], lw=2, color='cyan', label="Trajectory")
    cam_axes = {'x': None, 'y': None, 'z': None}

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        ax.cla()
        ax.plot(path[:frame+1, 0], path[:frame+1, 1], path[:frame+1, 2], lw=2, color='cyan')
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], c='lime', s=80, label="Start")
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='magenta', s=80, label="End")

        pose = poses[frame]
        R = pose[:3, :3]
        origin = pose[:3, 3]

        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2],
                  length=axis_length, color='r', label='X-axis')
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2],
                  length=axis_length, color='g', label='Y-axis')
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2],
                  length=axis_length, color='b', label='Z-axis')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Camera Flythrough Animation")
        ax.legend()
        ax.grid(True)

        return []

    ani = animation.FuncAnimation(fig, update, frames=len(poses), init_func=init, blit=False)

    # Save animation using OpenCV
    print(f"[INFO] Saving animation to: {output_path}")
    temp_path = "temp_anim_frames"
    os.makedirs(temp_path, exist_ok=True)

    for i in range(len(poses)):
        update(i)
        frame_path = os.path.join(temp_path, f"{i:04d}.png")
        plt.savefig(frame_path)

    # Read saved images and write video
    frame_paths = sorted([os.path.join(temp_path, f) for f in os.listdir(temp_path) if f.endswith(".png")])
    frame = cv2.imread(frame_paths[0])
    h, w, _ = frame.shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for fpath in frame_paths:
        img = cv2.imread(fpath)
        writer.write(img)

    writer.release()

    # Cleanup
    for f in frame_paths:
        os.remove(f)
    os.rmdir(temp_path)
    print(f"[INFO] Animation saved successfully.")

    plt.show()

def visualize_trajectory_with_poses(
    poses: list[np.ndarray],
    axis_length: float = 10,
    skip_frames: int = 15
):
    """
    Visualizes a 3D trajectory path and the camera's orientation at intervals.

    Args:
        poses (list[np.ndarray]): A list of 4x4 pose matrices (T_world_from_camera).
        axis_length (float): The visual length of the orientation axes.
        skip_frames (int): The number of frames to skip between drawing pose axes.
    """
    if not poses:
        print("Warning: The provided list of poses is empty.")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the trajectory path (translation components)
    path = np.array([p[:3, 3] for p in poses])

    # Plot the full trajectory path
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Trajectory Path', color='cyan')

    # Plot the orientation axes for a subset of poses
    for i, pose in enumerate(poses):
        if i % skip_frames != 0:
            continue
            
        # Origin of the camera's coordinate system in the world frame
        origin = pose[:3, 3]
        
        # Rotation matrix (orientation of the camera)
        R = pose[:3, :3]
        
        # Direction of the camera's axes in the world frame
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        # Draw the axes using quiver plots
        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2],
                  length=axis_length, color='r', label='X-axis' if i == 0 else "")
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2],
                  length=axis_length, color='g', label='Y-axis' if i == 0 else "")
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2],
                  length=axis_length, color='b', label='Z-axis' if i == 0 else "")

    # Mark the start and end points
    ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='lime', marker='o', s=100, label='Start')
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color='magenta', marker='*', s=150, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory and Pose Visualization')
    ax.legend()
    ax.grid(True)
    
    # Set aspect ratio to be equal
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        # A fallback for some matplotlib versions
        ax.set_box_aspect([1, 1, 1])

    plt.show()

if __name__ == "__main__":
    import sys
    traj = run_hough_lines_on_video(sys.argv[1])  # Change to your video file path
    visualize_trajectory_video(traj)
    # visualize_trajectory_with_poses(traj)
    # show_trajectory(traj)