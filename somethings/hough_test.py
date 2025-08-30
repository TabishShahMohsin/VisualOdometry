import cv2
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.cluster import DBSCAN

class LineDetectionProcessor:
    """
    A class to encapsulate the image processing workflow for detecting lines
    and their intersections using Hough Transform.
    """

    def __init__(self, image_folder, results_folder):
        """
        Initializes the LineDetectionProcessor with input and output paths
        and default parameters.

        Args:
            image_folder (Path): Path to the folder containing input images.
            results_folder (Path): Path to the folder where results will be saved.
        """
        self.image_folder = image_folder
        self.results_folder = results_folder
        self.results_folder.mkdir(parents=True, exist_ok=True)  # Ensure results directory exists

        # Parameters for Gaussian Blur
        self.blur_kernel = (5, 5)
        self.blur_sigma = 1.0

        # Parameters for Canny Edge Detector
        self.canny_lower_thresh = 4
        self.canny_higher_thresh = 10 # Increased default for better edge detection

        # Parameters for HoughLinesP
        self.hough_rho = 1              # Distance resolution of the accumulator in pixels.
        self.hough_theta = np.pi / 180  # Angle resolution of the accumulator in radians.
        self.hough_threshold = 100      # Accumulator threshold parameter. Only lines that get enough votes are returned.
        self.hough_min_line_length = 90 # Minimum line length. Line segments shorter than this are rejected.
        self.hough_max_line_gap = 10    # Maximum allowed gap between line segments to treat them as single line.

        # Parameters for Intersection Detection
        self.min_determinant_threshold = 3000  # Avoid division by small numbers for parallel lines
        self.intersection_buffer = 90       # A small buffer around line endpoints for intersection validation

        # Display control
        self.show_intermediate_steps = False # Set to True to display intermediate images

    def _show_image(self, img: np.array, title="Untitled"):
        """
        Helper function to display an image with a title.
        Only displays if self.show_intermediate_steps is True.
        """
        if self.show_intermediate_steps:
            cv2.imshow(title, img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                sys.exit("Exited by user.")
            cv2.destroyWindow(title) # Close individual window after viewing

    def _find_line_intersections(self, lines: np.array, img_shape: tuple) -> list:
        """
        Finds intersection points of line segments detected by HoughLinesP.

        Args:
            lines (np.array): An array of line segments in the format [[x1, y1, x2, y2], ...].
            img_shape (tuple): The (height, width) of the original image.

        Returns:
            list: A list of (x, y) tuples representing intersection points.
        """
        intersection_points = []
        if lines is None:
            return intersection_points

        # Convert lines to a more convenient list of tuples
        lines_list = [line[0] for line in lines]
        num_lines = len(lines_list)

        for i in range(num_lines - 1):
            for j in range(i + 1, num_lines):
                x1, y1, x2, y2 = lines_list[i]
                x3, y3, x4, y4 = lines_list[j]

                # Calculate slopes and intercepts (y = mx + b or x = const for vertical lines)
                # Line 1: A1*x + B1*y = C1
                A1 = y2 - y1
                B1 = x1 - x2
                C1 = A1 * x1 + B1 * y1

                # Line 2: A2*x + B2*y = C2
                A2 = y4 - y3
                B2 = x3 - x4
                C2 = A2 * x3 + B2 * y3

                determinant = A1 * B2 - A2 * B1

                # Check if lines are not parallel (or nearly parallel)
                if abs(determinant) > self.min_determinant_threshold:
                    x = (B2 * C1 - B1 * C2) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant

                    # Check if the intersection point is within the image bounds
                    if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                        # Check if the intersection point lies on both line segments
                        # We use a small buffer (intersection_buffer) to account for floating point inaccuracies
                        # and to allow intersections slightly outside the strict segment endpoints.
                        is_on_segment1 = (min(x1, x2) - self.intersection_buffer <= x <= max(x1, x2) + self.intersection_buffer) and \
                                         (min(y1, y2) - self.intersection_buffer <= y <= max(y1, y2) + self.intersection_buffer)

                        is_on_segment2 = (min(x3, x4) - self.intersection_buffer <= x <= max(x3, x4) + self.intersection_buffer) and \
                                         (min(y3, y4) - self.intersection_buffer <= y <= max(y3, y4) + self.intersection_buffer)

                        if is_on_segment1 and is_on_segment2:
                            intersection_points.append((int(round(x)), int(round(y))))
        return intersection_points

    def _cluster(self, points: list) -> list:
        points_np = np.array(points)

        # Apply DBSCAN clustering
        # eps = distance threshold in pixels (tune this, e.g., 5 to 10)
        db = DBSCAN(eps=50, min_samples=1).fit(points_np)

        # Get cluster labels
        labels = db.labels_

        # Compute the centroid for each cluster
        unique_labels = set(labels)
        merged_points = []

        for label in unique_labels:
            cluster = points_np[labels == label]
            centroid = np.mean(cluster, axis=0)
            merged_points.append(tuple(centroid.astype(int)))

        return merged_points


    def process_image(self, img_path: Path):
        """
        Processes a single image to detect lines and their intersections.

        Args:
            img_path (Path): The path to the input image.
        """
        img_name = img_path.name
        print(f"Processing {img_name}...")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Could not read image {img_path}. Skipping.")
            return

        original_display = img.copy()
        img = cv2.resize(img, (640, 480))
        self._show_image(original_display, 'Original Image')

        # 1. Grayscale Conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._show_image(gray, 'Grayscale')

        # 2. Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)
        self._show_image(blurred, 'Blurred')

        # 3. Canny Edge Detection
        edges = cv2.Canny(blurred, self.canny_lower_thresh, self.canny_higher_thresh)
        self._show_image(edges, 'Canny Edges')

        # 4. Hough Line Transform (Probabilistic)
        lines = cv2.HoughLinesP(edges, self.hough_rho, self.hough_theta,
                                self.hough_threshold,
                                minLineLength=self.hough_min_line_length,
                                maxLineGap=self.hough_max_line_gap)

        # 5. Visualize Detected Lines
        line_img = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5) # Green lines
        self._show_image(line_img, 'Detected Lines (HoughLinesP)')

        # Draw lines on the original image for combined view
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(original_display, (x1, y1), (x2, y2), (0, 255, 0), thickness=5) # Green lines

        # 6. Find and Draw Intersections
        intersections = self._find_line_intersections(lines, img.shape)
        intersections = self._cluster(intersections)
        for point in intersections:
            cv2.circle(original_display, point, 20, (0, 0, 255), thickness=-1) # Red circles

        self._show_image(original_display, 'Lines and Intersections')

        # Save the result
        output_path = self.results_folder / img_name
        cv2.imwrite(str(output_path), original_display)
        print(f"Result saved to {output_path}")

    def run(self):
        """
        Runs the line detection process for all images in the specified folder.
        """
        if not self.image_folder.exists():
            print(f"Error: Image folder '{self.image_folder}' does not exist.")
            return

        for img_name in os.listdir(self.image_folder):
            img_path = self.image_folder / img_name
            if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                self.process_image(img_path)
            else:
                print(f"Skipping non-image file: {img_name}")
        print("All images processed.")
        cv2.destroyAllWindows() # Ensure all windows are closed at the end


def run_hough_lines():
    # Setting up paths dynamically
    file_path = Path(__file__).resolve().parent
    image_folder_path = file_path / "frames"
    results_folder_path = file_path / "Results_hough"

    processor = LineDetectionProcessor(image_folder_path, results_folder_path)
    processor.show_intermediate_steps = True # Set to True to see images pop up during execution
    processor.run()

if __name__ == "__main__":
    run_hough_lines()