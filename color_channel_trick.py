import cv2
import numpy as np

# --- Load image ---
img = cv2.imread("frames/frame_00004.jpg")  # Replace with your path
if img is None:
    raise ValueError("Could not load image. Check the path.")

# --- Step 1: Convert to LAB color space ---
blur = cv2.GaussianBlur(img, (5,5), 1)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# --- Step 2: Enhance contrast in the L channel ---
L_eq = cv2.equalizeHist(L)  # Histogram equalization

# --- Step 3: Optional - noise reduction ---
# blur = cv2.GaussianBlur(L_eq, (5,5), 0)

# --- Step 4: Edge detection on enhanced L channel ---
thresh = cv2.adaptiveThreshold(L_eq, 255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV,
                            15, 20)
edges = cv2.Canny(L_eq, 30, 100)  # Lower thresholds = more sensitive

# --- Step 5: Visualize results ---
cv2.imshow("Original", img)
cv2.imshow("L Channel", L)
cv2.imshow("Enhanced L Channel", L_eq)
cv2.imshow("Edges from L Channel", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()