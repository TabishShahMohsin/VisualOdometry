# SubSense: Underwater Visual Perception

SubSense is a Python-based project focused on underwater visual perception and later odometry, specifically for detecting and analyzing features, tracking, and finding trajectory in submerged environments. It implements various computer vision algorithms for feature detection, corner detection, and line intersection analysis, primarily geared towards tasks like tile and object recognition from underwater imagery.

## Directory Structure

The project is organized into the following directories:

```
/
├── OT_img/               # Folder for "Other" category images (e.g., side-view).
├── TDview_tiles/         # Folder for Top-Down view tile images.
├── src/
│   └── Scripts/          # Core Python scripts for each algorithm.
│       ├── SIFT.py
│       ├── SuperPoint.py
│       └── ...
├── Results/              # Output directory for detector visualizations and logs.
│   ├── HoughLines/
│   ├── SIFT/
│   ├── SuperPoint/
│   └── ...
├── tests/                # Test suite for the project.
├── .gitignore
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Dataset Setup

The project is designed to work with two main categories of images. You can use your own datasets or download the sample datasets provided.

1.  **Place your images in the correct folders:**
    -   For top-down images of tiles, place them in the **`TDview_tiles/`** directory.
    -   For all other images, place them in the **`OT_img/`** directory.

2.  **Download Sample Datasets (Optional):**
    -   **Top-Down Dataset:** [Download Link](https://drive.google.com/drive/folders/12_9TZKUDzWol0RQWsRT-4R5ubx20WPzL?usp=drive_link)
    -   **Other Dataset:** [Download Link](https://drive.google.com/drive/folders/1CdwBr9-rXeyY6qpsHfhfpY5IYHob8wCM?usp=drive_link)

    After downloading, place the images into the corresponding `TDview_tiles` and `OT_img` folders.


## Installation

We use `poetry` inside a `conda` environment in this project.

1.  **Install Poetry**: If you don't have Poetry, follow the official instructions: [Poetry Installation](https://python-poetry.org/docs/#installing-with-the-official-installer).

2.  **Create and Activate Conda Environment**:
    ```bash
    conda create -n subsense python=3.11
    conda activate subsense
    ```

3.  **Install Dependencies**: Navigate to the project root and run:
    ```bash
    poetry install
    ```

## Development

-   **Adding Packages**: To add a new dependency to the project, use:
    ```bash
    poetry add <package-name>
    ```
-   **Running Tests**: To run the test suite, you can use `pytest`:
    ```bash
    poetry run pytest
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
