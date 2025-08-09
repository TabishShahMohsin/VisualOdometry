import cv2
import numpy as np
import math

# 3D real-world coordinates of A4 (in mm)
A4_WIDTH = 210
A4_HEIGHT = 297
object_points = np.array(
    [
        [0, 0, 0],  # Top-left
        [A4_WIDTH, 0, 0],  # Top-right
        [A4_WIDTH, A4_HEIGHT, 0],  # Bottom-right
        [0, A4_HEIGHT, 0],  # Bottom-left
    ],
    dtype=np.float32,
)

# Dummy camera intrinsics (you should use actual calibrated values)
camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no distortion


def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_a4_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))
    return None


def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, length=50):
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)
    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(
        np.vstack((origin, axis)), rvec, tvec, camera_matrix, dist_coeffs
    )

    corner = tuple(imgpts[0].ravel().astype(int))
    image = cv2.line(
        image, corner, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3
    )  # X: red
    image = cv2.line(
        image, corner, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 3
    )  # Y: green
    image = cv2.line(
        image, corner, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 3
    )  # Z: blue
    return image


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not read image.")
        return

    image_display = image.copy()
    corners = detect_a4_corners(image)

    if corners is None:
        print("❌ Could not detect A4 sheet.")
        return

    # Draw detected corners
    for i, pt in enumerate(corners):
        cv2.circle(image_display, tuple(pt.astype(int)), 6, (0, 255, 0), -1)
        cv2.putText(
            image_display,
            f"{i+1}",
            tuple(pt.astype(int) + np.array([5, -5])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points, corners, camera_matrix, dist_coeffs
    )
    if not success:
        print("❌ solvePnP failed.")
        return

    # Display pose
    R, _ = cv2.Rodrigues(rvec)
    euler = rotationMatrixToEulerAngles(R)

    print("\n📍 Camera Pose Relative to A4 Sheet:")
    print(f"Translation Vector (tvec) [mm]:\n{tvec.ravel()}")
    print(
        f"Rotation (Euler Angles in degrees):\nRoll={euler[0]:.2f}, Pitch={euler[1]:.2f}, Yaw={euler[2]:.2f}"
    )

    image_display = draw_axes(image_display, camera_matrix, dist_coeffs, rvec, tvec)
    cv2.imshow("A4 Detection + Axes", image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_pose_a4.py <image.jpg>")
    else:
        main(sys.argv[1])
