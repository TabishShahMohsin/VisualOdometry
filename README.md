# Sub-Odometry: Visual Odometry from a Tiled Grid

This project implements a visual odometry system to track the 3D position and orientation (pose) of a camera as it moves over a structured environment, specifically a grid of rectangular tiles. By detecting the corners of the grid in the video feed, it calculates the camera's motion frame by frame to reconstruct its trajectory.

## How It Works: The Pipeline

The core logic follows a multi-step pipeline to go from a raw video to a 3D trajectory.

### 1. Corner Detection

- **Input:** A video frame (image).
- **Process:** The system first needs to find the key feature points in the image, which are the corners of the grid tiles. This is handled by the `HoughCornerDetecter` class in `hough_corner_detector.py`.
    1.  **Preprocessing:** The image is converted to grayscale and a Gaussian blur is applied to reduce noise.
    2.  **Edge Detection:** The Canny edge detector is used to find sharp changes in intensity, which correspond to the lines of the grid.
    3.  **Line Detection:** A Probabilistic Hough Transform (`cv2.HoughLinesP`) is applied to the edge map. This detects straight line segments in the image.
    4.  **Intersection Finding:** The detected lines are analyzed to find their intersection points. A function `find_intersections_fast` calculates where pairs of lines cross.
    5.  **Clustering:** The raw intersection points are often clustered close together. The DBSCAN clustering algorithm groups these points, and the centroid of each cluster is taken as a single, more accurate corner point.
- **Output:** A list of `(x, y)` coordinates for the detected grid corners.

### 2. Pose Estimation with PnP

- **Input:** The list of 2D corner points detected in the image.
- **Process:** The `ransac` function in `pnp_grid_ransac.py` estimates the camera's 6D pose (3D rotation and 3D translation) from the 2D points.
    1.  **Define 3D Object Points:** We define the known 3D coordinates of a single tile in the real world (e.g., a 10x15 cm rectangle). These points are our reference.
    2.  **Find a Local Grid Patch:** The code iterates through the detected corners. For each corner, it finds its 8 nearest neighbors and sorts them cyclically (`get_8_closest_cyclic`) to form a local 3x3 grid patch. This assumes the camera is always looking at a part of the grid.
    3.  **Solve for Pose:** The `cv2.solvePnP` function is the core of this step. It takes the 3D coordinates of the reference tile and the corresponding 2D coordinates of the detected corners in the image. Using the camera's intrinsic parameters (`K` matrix), it calculates the rotation vector (`rvec`) and translation vector (`tvec`) that describe the camera's position and orientation relative to that tile.
- **Output:** The rotation (`rvec`) and translation (`tvec`) vectors for the current frame.

### 3. Trajectory Reconstruction

- **Input:** The pose (`rvec`, `tvec`) for the current frame and the accumulated trajectory so far.
- **Process:** The `main.py` script orchestrates the whole process.
    1.  **Delta Calculation:** The `translation_delta` function calculates the change in position and yaw (rotation around the vertical axis) between the current frame and the previous one. It includes logic to handle wrapping (e.g., moving across tile boundaries) and sign ambiguities.
    2.  **Integration:** The calculated delta is added to the previous pose to get the new, absolute pose of the camera. This is repeated for every frame.
- **Output:** A complete trajectory, which is a list of 3D points representing the camera's path.

### 4. Visualization

- **Process:** Once the video is processed, the `show_trajectory` function in `main.py` uses `matplotlib` to create a 3D plot of the reconstructed path, with a color gradient to indicate the direction of motion over time.

## File Breakdown

-   `main.py`: The main entry point of the application. It reads the video, calls the processing pipeline for each frame, and plots the final trajectory.
-   `hough_corner_detector.py`: Contains the `HoughCornerDetecter` class responsible for finding grid corners in an image.
-   `pnp_grid_ransac.py`: Implements the `ransac` and `get_8_closest_cyclic` functions to estimate the camera's pose from the detected corners using `cv2.solvePnP`.
-   `config.py`: A configuration file that stores global constants like the physical dimensions of the grid tiles (`HEIGHT`, `WIDTH`) and the camera's intrinsic matrix (`K`).
-   `vid_creator.py`: A helper script to generate a synthetic video of a camera moving over a procedurally generated grid. This is useful for testing the odometry pipeline in a controlled environment.
-   `demo_wrt_cam.py`: A utility for visualizing how a 3D grid appears from different camera positions and orientations. It helps in understanding the effects of rotation and translation and for debugging the projection logic.
-   `test.py`, `ransac_testing.py`: Scripts used for developing and testing specific parts of the system.

## Key Concepts

-   **Visual Odometry:** A technique to determine the position and orientation of a robot or vehicle by analyzing a sequence of camera images.
-   **Hough Transform:** A feature extraction technique used to identify lines, circles, or other shapes in an image.
-   **Perspective-n-Point (PnP):** A computer vision problem to find the pose of a camera given a set of n 3D points in the world and their corresponding 2D projections in an image.
-   **Camera Intrinsics (K matrix):** A matrix containing the camera's internal parameters like focal length (`fx`, `fy`) and optical center (`cx`, `cy`), which are needed to map 3D world points to 2D image points.

## How to Run

1.  **Ensure Dependencies are Installed:**
    You will need the following Python libraries:
    -   `opencv-python`
    -   `numpy`
    -   `matplotlib`
    -   `scikit-learn`

2.  **Run the Main Script:**
    Execute the `main.py` script. You may need to change the video file path inside the script.

    ```bash
    python main.py
    ```

    The script will process the video specified in `run_hough_lines_on_video('test4.mp4')`, display the video with detected corners overlaid, and finally show a 3D plot of the camera's trajectory.
