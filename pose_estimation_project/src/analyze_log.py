import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

DEBUG_LOG_PATH = "dataset/debug_log.csv"
ANALYSIS_PLOT_DIR = "dataset/analysis_plots"

# ==============================================================================
#                               ANALYSIS SCRIPT
# ==============================================================================

def main():
    print(f"Analyzing debug log: {DEBUG_LOG_PATH}")
    os.makedirs(ANALYSIS_PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DEBUG_LOG_PATH)

    # --- Plot 1: PnP OCS Pose Error ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df['frame'], df['gt_t_ocs_x'], label='Ground Truth t_OCS_x')
    plt.plot(df['frame'], df['pnp_t_ocs_x'], label='PnP t_OCS_x', linestyle='--')
    plt.title('PnP OCS X-Position vs. Ground Truth')
    plt.xlabel('Frame')
    plt.ylabel('X Position (cm)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df['frame'], df['gt_t_ocs_y'], label='Ground Truth t_OCS_y')
    plt.plot(df['frame'], df['pnp_t_ocs_y'], label='PnP t_OCS_y', linestyle='--')
    plt.title('PnP OCS Y-Position vs. Ground Truth')
    plt.xlabel('Frame')
    plt.ylabel('Y Position (cm)')
    plt.legend()
    plt.grid(True)

    plot1_path = os.path.join(ANALYSIS_PLOT_DIR, "pnp_ocs_error.png")
    plt.savefig(plot1_path)
    print(f"Saved PnP error plot to: {plot1_path}")

    # --- Plot 2: Final Trajectory Error (Component-wise) ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df['gt_wcs_x'], df['gt_wcs_y'], label='Ground Truth Trajectory', color='g')
    plt.plot(df['recon_wcs_x'], df['recon_wcs_y'], label='Reconstructed Trajectory', color='b', linestyle='--')
    plt.title('Full Trajectory')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    error_x = df['recon_wcs_x'] - df['gt_wcs_x']
    error_y = df['recon_wcs_y'] - df['gt_wcs_y']
    plt.plot(df['frame'], error_x, label='X Error')
    plt.plot(df['frame'], error_y, label='Y Error')
    plt.title('Trajectory Error Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Error (cm)')
    plt.legend()
    plt.grid(True)

    plot2_path = os.path.join(ANALYSIS_PLOT_DIR, "trajectory_error.png")
    plt.savefig(plot2_path)
    print(f"Saved trajectory error plot to: {plot2_path}")

if __name__ == '__main__':
    main()
