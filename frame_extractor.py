import cv2
import os

def extract_frames(video_path, output_folder="frames", every_nth=1):
    """
    Extracts frames from a video and saves them as image files.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save extracted frames.
        every_nth (int): Save every nth frame (default = 1 for all).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % every_nth == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Done. Saved {saved_count} frames to '{output_folder}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <path_to_video>")
    else:
        video_path = sys.argv[1]
        extract_frames(video_path)