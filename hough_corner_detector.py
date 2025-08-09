import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from numba import njit


@njit
# The task of finding intersections is to be parallized later
def find_intersections_fast(lines_list, height, width, det_thresh, buffer):
    points = []
    n = len(lines_list)
    for i in range(n - 1):
        for j in range(i + 1, n):
            x1, y1, x2, y2 = lines_list[i]
            x3, y3, x4, y4 = lines_list[j]

            A1 = y2 - y1
            B1 = x1 - x2
            C1 = A1 * x1 + B1 * y1

            A2 = y4 - y3
            B2 = x3 - x4
            C2 = A2 * x3 + B2 * y3

            det = A1 * B2 - A2 * B1
            if abs(det) > det_thresh:
                x = (B2 * C1 - B1 * C2) / det
                y = (A1 * C2 - A2 * C1) / det

                if 0 <= x < width and 0 <= y < height:
                    if min(x1, x2) - buffer <= x <= max(x1, x2) + buffer and \
                       min(y1, y2) - buffer <= y <= max(y1, y2) + buffer and \
                       min(x3, x4) - buffer <= x <= max(x3, x4) + buffer and \
                       min(y3, y4) - buffer <= y <= max(y3, y4) + buffer:
                        points.append((int(round(x)), int(round(y))))
    return points


class HoughCornerDetecter:
    """
    A class to encapsulate the image processing workflow for detecting lines
    and their intersections using Hough Transform.
    """

    def __init__(self):
        """
        Initializes the LineDetectionProcessor with input and output paths
        and default parameters.

        Args:
            image_folder (Path): Path to the folder containing input images.
            results_folder (Path): Path to the folder where results will be saved.
        """

        # Parameters for Gaussian Blur
        self.blur_kernel = (5, 5)
        self.blur_sigma = 1

        # Parameters for Canny Edge Detector
        self.canny_lower_thresh = 4 # Increased default for better edge detection
        self.canny_higher_thresh = 12 # Increased default for better edge detection

        # Parameters for HoughLinesP
        self.hough_rho = 1              # Distance resolution of the accumulator in pixels.
        self.hough_theta = np.pi / 180  # Angle resolution of the accumulator in radians.
        # self.hough_threshold = 100      # Accumulator threshold parameter. Only lines that get enough votes are returned.
        self.hough_threshold = 200      # Accumulator threshold parameter. Only lines that get enough votes are returned.
        self.hough_min_line_length = 90 # Minimum line length. Line segments shorter than this are rejected.
        self.hough_max_line_gap = 10    # Maximum allowed gap between line segments to treat them as single line.

        # Parameters for Intersection Detection
        self.min_determinant_threshold = 3000  # Avoid division by small numbers for parallel lines
        self.intersection_buffer = 90       # A small buffer around line endpoints for intersection validation
        # self.intersection_buffer = 900       # A small buffer around line endpoints for intersection validation

    def _find_line_intersections(self, lines, img_shape):
        if lines is None:
            return []

        lines_list = np.array([line[0] for line in lines])
        return find_intersections_fast(lines_list, img_shape[0], img_shape[1], self.min_determinant_threshold, self.intersection_buffer)


    def _cluster(self, points: list) -> list:
        points_np = np.array(points)
        if len(points_np) == 0:
            return []

        db = DBSCAN(eps=20, min_samples=1).fit(points_np)
        labels = db.labels_
        merged_points = np.array([points_np[labels == label].mean(axis=0) for label in np.unique(labels)])
        return [tuple(map(int, pt)) for pt in merged_points]


    def process_image(self, img: np.array) -> list:
        """
        Processes a single image to detect lines and their intersections.

        Args:
            img_path (Path): The path to the input image.
        """
        # Change this first if any problem especially with the camera matrix
        # img = cv2.resize(img, (640, 480))
        # Invert if lines are dark
        # inverted = cv2.bitwise_not(img)
        # # Top-hat to extract dark lines
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        # tophat = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel)
        # cv2.imshow('tophat', tophat)

        # 1. Grayscale Conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    15, 2.5)
        # cv2.imshow('thresh', thresh)
        # cv2.waitKey(0)

        # 2. Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(thresh, self.blur_kernel, self.blur_sigma)


        # 3. Canny Edge Detection
        edges = cv2.Canny(blurred, self.canny_lower_thresh, self.canny_higher_thresh)

        # 4. Hough Line Transform (Probabilistic)
        lines = cv2.HoughLinesP(thresh, self.hough_rho, self.hough_theta,
                                self.hough_threshold,
                                minLineLength=self.hough_min_line_length,
                                maxLineGap=self.hough_max_line_gap)

        # 6. Find Intersections
        intersections = self._find_line_intersections(lines, img.shape)
        intersections = self._cluster(intersections)

        # return intersections, lines, edges
        return intersections