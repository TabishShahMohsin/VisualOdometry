import cv2
import numpy as np

# Path to the video
video_path = "datasets/meston_pool/videos/pool_corner.MOV"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))

def find_intersections(lines, shape):
    """Find intersections between near-horizontal and near-vertical lines."""
    intersections = []
    if lines is None:
        return intersections
    
    horizontals = []
    verticals = []

    for rho, theta in lines[:, 0]:
        if abs(theta) < np.pi / 4 or abs(theta - np.pi) < np.pi / 4:
            verticals.append((rho, theta))
        elif abs(theta - np.pi / 2) < np.pi / 4:
            horizontals.append((rho, theta))

    for rho1, theta1 in verticals:
        for rho2, theta2 in horizontals:
            # Solve for intersection of the two lines
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([rho1, rho2])
            try:
                x0, y0 = np.linalg.solve(A, b)
                if 0 <= x0 < shape[1] and 0 <= y0 < shape[0]:
                    intersections.append((int(x0), int(y0)))
            except np.linalg.LinAlgError:
                continue
    return intersections

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    enhanced = clahe.apply(blur)

    edges = cv2.Canny(enhanced, 10, 30, apertureSize=3)
    cv2.imshow("edges", edges)

    # Non-probabilistic Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=70)

    # Draw lines for visualization
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Find grid intersections
    intersections = find_intersections(lines, frame.shape)

    for (x, y) in intersections:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # red dot

    cv2.imshow("Marked points", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()