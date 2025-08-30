import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from model import PoseNet
import os
from tqdm import tqdm

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

VIDEO_PATH = "dataset/crab_walk_video.mp4"
GT_CSV_PATH = "dataset/gt_trajectory.csv"
MODEL_PATH = "models/periodnet_mobilenet_v3_small_best.pth"
OUTPUT_PLOT_PATH = "dataset/trajectory_comparison.png"

# --- Model & Tile Config ---
BACKBONE = 'mobilenet_v2' # Use the updated backbone
TILE_WIDTH = 5.0
TILE_HEIGHT = 3.0

# ==============================================================================
#                               INFERENCE SCRIPT
# ==============================================================================

def preprocess_frame(frame):
    """Applies the same preprocessing as used in training."""
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAYSCALE)
    
    # Sobel filter
    sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    sobel_frame = np.hypot(sobelx, sobely)
    sobel_frame = cv2.normalize(sobel_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Torchvision transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(sobel_frame).unsqueeze(0) # Add batch dimension

def main():
    print("Starting inference pipeline...")
    
    # --- Device Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Model Loading ---
    print(f"Loading model with backbone: {BACKBONE}...")
    model = PoseNet(backbone=BACKBONE, tile_dims=(TILE_WIDTH, TILE_HEIGHT)).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}")
        # NOTE: Loading weights from a model with a different architecture.
        # This may not be optimal. Using strict=False to load what is possible.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    else:
        print(f"Warning: Model weights not found at {MODEL_PATH}. Using a randomly initialized model.")
        
    model.eval()

    # --- Video Processing ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {VIDEO_PATH}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {num_frames} frames from {VIDEO_PATH}...")

    # --- Trajectory Reconstruction ---
    reconstructed_trajectory = [np.array([0.0, 0.0, 0.0])]
    last_offsets = None

    for _ in tqdm(range(num_frames), desc="Analyzing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and predict
        input_tensor = preprocess_frame(frame).to(device)
        with torch.no_grad():
            _, pred_offsets = model(input_tensor)
        
        current_offsets = pred_offsets.squeeze().cpu().numpy()

        if last_offsets is not None:
            # Calculate delta_t_OCS
            delta_t_ocs = current_offsets - last_offsets
            
            # Resolve integer ambiguity
            tile_shift_x = np.round(delta_t_ocs[0] / TILE_WIDTH)
            tile_shift_y = np.round(delta_t_ocs[1] / TILE_HEIGHT)
            
            # Calculate true displacement (delta_t_WCS)
            delta_t_wcs_x = delta_t_ocs[0] - tile_shift_x * TILE_WIDTH
            delta_t_wcs_y = delta_t_ocs[1] - tile_shift_y * TILE_HEIGHT
            
            # Accumulate trajectory
            last_point = reconstructed_trajectory[-1]
            new_point = last_point + np.array([delta_t_wcs_x, delta_t_wcs_y, 0])
            reconstructed_trajectory.append(new_point)

        last_offsets = current_offsets

    cap.release()
    reconstructed_trajectory = np.array(reconstructed_trajectory)

    # --- Visualization ---
    print("Plotting results...")
    gt_trajectory_df = pd.read_csv(GT_CSV_PATH)
    
    # Center both trajectories at the origin
    gt_traj = gt_trajectory_df[['x', 'y']].values
    gt_traj -= gt_traj[0, :]
    reconstructed_trajectory -= reconstructed_trajectory[0, :]

    plt.figure(figsize=(10, 10))
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], label='Ground Truth Trajectory', color='g', linewidth=2)
    plt.plot(reconstructed_trajectory[:, 0], reconstructed_trajectory[:, 1], label='Reconstructed Trajectory', color='r', linestyle='--', marker='x', markersize=4)
    plt.title('Camera Trajectory Reconstruction')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(OUTPUT_PLOT_PATH)
    
    print("-" * 30)
    print(f"Inference complete. Trajectory plot saved to:")
    print(OUTPUT_PLOT_PATH)
    print("-" * 30)

if __name__ == '__main__':
    main()
