# Sub-Odometry: Visual Odometry from a Tiled Grid

This project is a visual odometry system designed to track the 3D position and orientation (pose) of a camera as it moves over a structured environment, specifically a grid of rectangular tiles. By detecting the corners of the grid in a video feed, it calculates the camera's motion frame by frame to reconstruct its trajectory.

This document explains the intended logic, the current implementation, a critical flaw in its logic, and the path to fixing it.

## Core Pipeline and Intended Logic

The system is designed to work in a multi-step pipeline:

### 1. Corner Detection (`hough_corner_detector.py`)

The first step is to find the key feature points in each video frame, which are the corners of the grid tiles.

-   **Input:** A raw video frame.
-   **Process:** The `HoughCornerDetecter` class performs a series of image processing operations:
    1.  **Grayscale & Blur:** The image is converted to grayscale and a Gaussian blur is applied to reduce image noise.
    2.  **Edge Detection:** A Canny edge detector finds the sharp intensity changes that correspond to the lines of the grid.
    3.  **Line Detection:** A Probabilistic Hough Transform (`cv2.HoughLinesP`) is applied to the edge map to detect straight line segments.
    4.  **Intersection Finding:** The detected lines are analyzed to find their intersection points. The `find_intersections_fast` function calculates where pairs of lines cross.
    5.  **Clustering:** The raw intersection points are often clustered close together. The DBSCAN clustering algorithm groups these points, and the centroid of each cluster is taken as a single, more accurate corner point.
-   **Output:** A list of `(x, y)` coordinates for the detected grid corners in the 2D image.

### 2. Pose Estimation (`pnp_grid_ransac.py`)

This module is responsible for taking the 2D corner points and estimating the camera's 6D pose (3D rotation and 3D translation).

-   **Input:** The list of 2D corner points from the previous step.
-   **Process:** The `ransac` function attempts to solve the Perspective-n-Point (PnP) problem.
    1.  **Define 3D Object Points:** A 3D model of a single, ideal grid patch is defined in `objectPoints`. This is our real-world reference.
    2.  **Select Image Points:** The code iterates through the detected corners, and for each one, it finds its 8 nearest neighbors in the 2D image using `get_8_closest_cyclic`. This 3x3 patch of 2D points is selected as the `imagePoints`.
    3.  **Solve for Pose:** The `cv2.solvePnP` function is the core of this step. It takes the 3D `objectPoints`, the corresponding 2D `imagePoints`, and the camera's intrinsic parameters (`K` matrix) and calculates the rotation vector (`rvec`) and translation vector (`tvec`) that describe the camera's pose relative to the selected grid patch.
-   **Output:** An `rvec` and `tvec` for the current frame.

### 3. Trajectory Reconstruction (`main.py`)

The main script orchestrates the pipeline and builds the final trajectory.

-   **Input:** The pose (`rvec`, `tvec`) for each frame.
-   **Process:**
    1.  The script reads the video frame by frame.
    2.  For each frame, it calls the corner detector and then the PnP solver.
    3.  It then converts the calculated pose into the world coordinate system to get the camera's absolute position.
    4.  This absolute position is appended to a list to form the trajectory.
-   **Output:** A complete 3D trajectory of the camera's path.

### 4. Visualization (`main.py`)

-   Once the video is processed, the `show_trajectory` function uses `matplotlib` to create a 3D plot of the reconstructed path, with a color gradient to indicate the direction of motion over time.

---

## Critical Flaw: Why It Fails During Yaw Rotation

The current implementation has a fundamental logical flaw that causes the trajectory to become chaotic (`go haywire`) the moment the camera rotates (yaws), even though it works perfectly during pure translation.

**The Problem is Unstable Feature Selection.**

-   **During Translation:** When the camera moves without rotating, the pattern of corners on the screen is rigid. The `get_8_closest_cyclic` function consistently picks the same physical patch on the grid as a reference. The input to `solvePnP` is stable, and the resulting trajectory is smooth.

-   **During Yaw:** When the camera rotates, the perspective changes dramatically. The points that were closest in the 2D image in the last frame are no longer the closest. The `get_8_closest_cyclic` function therefore picks a **completely different physical patch on the grid** to use as a reference in each frame.

This is like trying to measure your own movement while constantly changing your reference point. The result is a chaotic and meaningless series of poses. `solvePnP` is working correctly, but it is being fed inconsistent data that makes it seem as if the camera is jumping randomly across the grid.

## How to Fix the System

The solution is to provide `solvePnP` with a stable and consistent reference across frames. This is achieved by implementing **feature tracking**.

1.  **Detect, Then Track:** Instead of re-detecting and re-selecting corners in every frame, the system should be modified:
    *   **Detect Keypoints:** Run the `HoughCornerDetecter` once to establish an initial set of reliable keypoints.
    *   **Track Keypoints:** In all subsequent frames, use an optical flow algorithm, such as **`cv2.calcOpticalFlowPyrLK`**, to track the movement of those initial keypoints.

2.  **Provide Stable Input:** Feed the *tracked* 2D points into `cv2.solvePnP`. Because these points correspond to the same physical objects from frame to frame, the reference is now stable.

3.  **Use Extrinsic Guess:** To further improve stability, the `useExtrinsicGuess=True` flag should be used in `cv2.solvePnP`. This tells the solver to use the pose from the previous frame as its starting point, preventing large, incorrect jumps in the solution.

By implementing these changes, the system will correctly attribute the movement of the tracked points to the camera's rotation, resolving the failure during yaw and producing a correct and smooth trajectory.

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