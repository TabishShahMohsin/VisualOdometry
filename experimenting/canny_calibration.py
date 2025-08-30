import cv2
import numpy as np
from pathlib import Path

file_path = Path(__file__).resolve().parent

# Load image
image = cv2.imread(file_path / 'frames' / 'frame_00199.jpg')  # Change this to your image path
if image is None:
    raise ValueError("Image not found!")

kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations=2)

# Resize for display if needed
image = cv2.resize(image, (640, 480))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

# Trackbar callback function (does nothing, required by createTrackbar)
def nothing(x):
    pass

# Create window
cv2.namedWindow("Canny Edge Detector")

# Create trackbars for minVal and maxVal
cv2.createTrackbar("Min Threshold", "Canny Edge Detector", 50, 255, nothing)
cv2.createTrackbar("Max Threshold", "Canny Edge Detector", 150, 255, nothing)

while True:
    # Get current positions of the trackbars
    minVal = cv2.getTrackbarPos("Min Threshold", "Canny Edge Detector")
    maxVal = cv2.getTrackbarPos("Max Threshold", "Canny Edge Detector")

    # Apply Canny Edge Detection
    canny = cv2.Canny(blurred, minVal, maxVal)

    # Stack grayscale blurred and Canny result side by side
    stacked = np.hstack((blurred, canny))

    # Show the image
    cv2.imshow("Canny Edge Detector", stacked)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()