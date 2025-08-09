import cv2
from hough_corner_detector import HoughCornerDetecter
import cv2

def visualize_hough_corners(video_path=0):
    """
    Visualizes Hough-based corner detection live on a video or webcam feed.
    Args:
        video_path (str or int): Path to video file or 0 for webcam.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video source.")
        return

    detector = HoughCornerDetecter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))


        # Detect intersections (corners)
        corners, lines, edges = detector.process_image(frame)
        cv2.imshow('something', edges)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]  # line is [[x1, y1, x2, y2]]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Draw red circles on detected corners
        for (x, y) in corners:
            cv2.circle(frame, (x, y), 9, (0, 0, 255), -1)  # Red dot

        cv2.imshow('Hough Corner Detection', frame)
        # cv2.waitKey(0)
        # cv2.imshow('canny', canny)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_hough_corners(sys.argv[1])  # Pass video path as CLI argument
    else:
        visualize_hough_corners(0)  # Default to webcam