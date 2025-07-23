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

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        intersections = processor.process_image(frame)
        for x, y in intersections:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        # print("intersection_count: ", len(intersections))

        rvec, tvec, imagePoints, point = ransac(intersections)

        # print('tvec:', tvec, '\n', 'trajectory:',  trajectory)
        x, y, z = translation_delta(tvec, trajectory[-1])
        print('x, y, z:', x, y, z)

        # print(rvec, tvec, sep='\n') 
        trajectory.append(trajectory[-1] + np.array([x, y, z]))
        # print(trajectory)
        print(trajectory[-1])

        for i in imagePoints:
            cv2.circle(frame, (int(i[0]), int(i[1])), 15, (0, 255, 0), -1)
        cv2.circle(frame, (point[0], point[1]), 7, (0, 0, 255), -1)
        # cv2.imshow('Hough Corner Detection', frame)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break
    cap.release()
    cv2.destroyAllWindows()
    return trajectory

def translation_delta(final_tvec: np.array, initial_tvec:np.array) -> np.array:

    # Making z +ve
    sign_i = 1 if initial_tvec[2] > 0 else -1
    sign_f = 1 if final_tvec[2] > 0 else -1
    initial_tvec = initial_tvec * sign_i
    final_tvec = final_tvec * sign_f

    def wrap_centered(value, max_value):
        return ((value + max_value / 2) % max_value) - max_value / 2
    print('final_x:', final_tvec[0], 'initial_x', initial_tvec[0])
    x = wrap_centered(final_tvec[0] - initial_tvec[0], config.WIDTH)
    y = wrap_centered(final_tvec[1] - initial_tvec[1], config.HEIGHT)
    z = final_tvec[2] - initial_tvec[2] 
    # print('x, y, z:', x, y, z)
    return np.round(x, 5), np.round(y, 5), np.round(z, 5)
    


def show_trajectory(trajectory):
    """
    Plots a 2D or 3D trajectory from a list of coordinates.
    
    Args:
        trajectory (list): List of [x, y] or [x, y, z] points.
    """
    if not trajectory:
        print("Trajectory is empty.")
        return

    from mpl_toolkits.mplot3d import Axes3D
    x, y, z = zip(*trajectory)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='+', linestyle='-', color='g')
    ax.set_title("3D Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    traj = run_hough_lines_on_video('simple.mp4')  # Change to your video file path
    show_trajectory(traj)