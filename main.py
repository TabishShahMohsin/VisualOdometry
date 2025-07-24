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
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return

    processor = HoughCornerDetecter()

    t_total = np.zeros((3, 1))
    trajectory = [t_total]
    yaw_trajectory = [0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        intersections = processor.process_image(frame)
        for x, y in intersections:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        # print("intersection_count: ", len(intersections))

        rvec, tvec, imagePoints, point = ransac(intersections)

        sign = yaw_sign(yaw_trajectory[-1])

        # print('tvec:', tvec, '\n', 'trajectory:',  trajectory)
        x, y, z, yaw = translation_delta(tvec, trajectory[-1], rvec[2], yaw_trajectory[-1])
        x = x * sign
        y = y * sign
        print('x, y, z, yaw:', x, y, z, yaw)

        # print(rvec, tvec, sep='\n') 
        trajectory.append(trajectory[-1] + np.array([x, y, z]))
        yaw_trajectory.append(yaw_trajectory[-1] + yaw)
        # print(trajectory)
        print('current coordinates', trajectory[-1])
        print('current_yaw', yaw_trajectory[-1])


        for i in imagePoints:
            cv2.circle(frame, (int(i[0]), int(i[1])), 15, (0, 255, 0), -1)
        cv2.circle(frame, (point[0], point[1]), 7, (0, 0, 255), -1)
        cv2.imshow('Hough Corner Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break
    cap.release()
    cv2.destroyAllWindows()
    return trajectory

def translation_delta(final_tvec: np.array, initial_tvec:np.array, final_yaw: np.array, initial_yaw:np.array) -> np.array:

    # Making z +ve
    sign_i = 1 if initial_tvec[2] > 0 else -1
    sign_f = 1 if final_tvec[2] > 0 else -1
    initial_tvec = initial_tvec * sign_i
    final_tvec = final_tvec * sign_f

    initial_yaw = initial_yaw * sign_i
    final_yaw = final_yaw * sign_f

    print('final_x:', final_tvec[0], 'initial_x', initial_tvec[0])
    x = wrap_centered(final_tvec[0] - initial_tvec[0], config.WIDTH)
    y = wrap_centered(final_tvec[1] - initial_tvec[1], config.HEIGHT)
    z = final_tvec[2] - initial_tvec[2] 
    yaw = wrap_centered(final_yaw - initial_yaw, np.pi / 2)
    # print('x, y, z:', x, y, z)
    return np.round(x, 5), np.round(y, 5), np.round(z, 5), np.round(yaw, 5)

def wrap_centered(value, max_value):
    return ((value + max_value / 2) % max_value) - max_value / 2
    


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


def yaw_sign(yaw):
    return 1
    if 0 <= yaw < np.pi / 4:
        return 1
    else:
        return -1
    # return 1 if wrap_centered(yaw, 2 * np.pi) > 0 else -1


if __name__ == "__main__":
    traj = run_hough_lines_on_video('test4.mp4')  # Change to your video file path
    show_trajectory(traj)