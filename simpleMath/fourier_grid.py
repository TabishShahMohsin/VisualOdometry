import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def create_tilted_grid(width, height, theta_deg, p, q, c1_offset=0, c2_offset=0, add_noise=False):
    """
    Generates a tilted grid image with specified offsets and optional noise.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        theta_deg (float): Rotation angle of the grid in degrees.
        p (int): Spacing for the first set of parallel lines.
        q (int): Spacing for the second set of parallel lines.
        c1_offset (int): Offset for the first set of lines.
        c2_offset (int): Offset for the second set of lines.
        add_noise (bool): If True, adds Gaussian noise to the image.

    Returns:
        numpy.ndarray: A grayscale image of the tilted grid.
    """
    image = np.zeros((height, width), dtype=np.float32)
    theta_rad = np.deg2rad(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Draw the first set of parallel lines with offset c1
    for i in range(-width * 2, width * 2, p):
        offset = c1_offset + i
        pt1 = (int(width/2 - 2*width*sin_t + offset*cos_t), int(height/2 + 2*width*cos_t + offset*sin_t))
        pt2 = (int(width/2 + 2*width*sin_t + offset*cos_t), int(height/2 - 2*width*cos_t + offset*sin_t))
        cv2.line(image, pt1, pt2, (255, 255, 255), 1)

    # Draw the second set of parallel lines with offset c2
    for i in range(-height * 2, height * 2, q):
        offset = c2_offset + i
        pt1 = (int(width/2 - 2*width*cos_t - offset*sin_t), int(height/2 - 2*height*sin_t + offset*cos_t))
        pt2 = (int(width/2 + 2*width*cos_t - offset*sin_t), int(height/2 + 2*height*sin_t + offset*cos_t))
        cv2.line(image, pt1, pt2, (255, 255, 255), 1)
    
    if add_noise:
        noise = np.random.normal(0, 50, image.shape)
        image = np.clip(image + noise, 0, 255)
        
    return image.astype(np.uint8)

def extract_grid_parameters(image):
    """
    Analyzes a grid image to extract its rotation angle, spacings, and offsets.

    Args:
        image (numpy.ndarray): The input grayscale grid image.

    Returns:
        tuple: Contains (angle, spacing1, spacing2, offset1, offset2, peaks, spectrum).
    """
    height, width = image.shape

    # 1. Compute Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)

    # 2. Find peaks in the magnitude spectrum
    center_x, center_y = width // 2, height // 2
    mask = np.ones(magnitude_spectrum.shape, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius=15, color=0, thickness=-1)
    
    threshold = 0.7 * np.max(magnitude_spectrum[mask.astype(bool)])
    peaks_indices, _ = find_peaks(magnitude_spectrum.flatten(), height=threshold)
    
    peaks_coords = np.array(np.unravel_index(peaks_indices, magnitude_spectrum.shape)).T
    
    valid_peaks = [p for p in peaks_coords if np.sqrt((p[0] - center_y)**2 + (p[1] - center_x)**2) > 15]
    
    if len(valid_peaks) < 2:
        print("Could not find enough significant peaks.")
        return None, None, None, None, None, None, magnitude_spectrum

    # 3. Find the two fundamental, non-collinear peaks
    vectors = np.array(valid_peaks) - np.array([center_y, center_x])
    distances = np.linalg.norm(vectors, axis=1)
    sorted_indices = np.argsort(distances)
    
    vec1 = vectors[sorted_indices[0]]
    vec2 = None
    for i in range(1, len(sorted_indices)):
        vec_candidate = vectors[sorted_indices[i]]
        cosine_angle = np.dot(vec1, vec_candidate) / (np.linalg.norm(vec1) * np.linalg.norm(vec_candidate))
        if abs(cosine_angle) < 0.95:
            vec2 = vec_candidate
            break

    if vec2 is None:
        print("Could not find two non-collinear fundamental peaks.")
        return None, None, None, None, None, np.array(valid_peaks), magnitude_spectrum

    # 4. Calculate Angle and Spacings
    angle1_rad = np.arctan2(vec1[0], vec1[1])
    angle2_rad = np.arctan2(vec2[0], vec2[1])
    angle1_deg = (np.rad2deg(angle1_rad) + 90) % 180
    angle2_deg = (np.rad2deg(angle2_rad) + 90) % 180
    final_angle = min(angle1_deg, angle2_deg)

    spatial_freq1 = np.sqrt((vec1[1]/width)**2 + (vec1[0]/height)**2)
    spatial_freq2 = np.sqrt((vec2[1]/width)**2 + (vec2[0]/height)**2)
    spacing1 = 1 / spatial_freq1
    spacing2 = 1 / spatial_freq2

    # 5. Calculate Offsets from Phase
    # Get the complex values at the peak locations (from the un-shifted transform)
    peak1_coord = (int(vec1[0] + center_y), int(vec1[1] + center_x))
    peak2_coord = (int(vec2[0] + center_y), int(vec2[1] + center_x))
    
    # Need to handle wrapping around the edges for fft indices
    fft_peak1_coord = (peak1_coord[0] % height, peak1_coord[1] % width)
    fft_peak2_coord = (peak2_coord[0] % height, peak2_coord[1] % width)

    complex_val1 = f_transform[fft_peak1_coord]
    complex_val2 = f_transform[fft_peak2_coord]

    phase1 = np.angle(complex_val1)
    phase2 = np.angle(complex_val2)

    # Offset c = -phase / (2*pi*f), where f is spatial frequency
    # We must account for the rotation of the coordinate system
    offset1 = (-phase1 / (2 * np.pi * spatial_freq1)) % spacing1
    offset2 = (-phase2 / (2 * np.pi * spatial_freq2)) % spacing2
    
    # Ensure correct pairing of parameters
    if angle1_deg == final_angle:
        final_spacing_p, final_spacing_q = spacing1, spacing2
        final_offset_c1, final_offset_c2 = offset1, offset2
    else:
        final_spacing_p, final_spacing_q = spacing2, spacing1
        final_offset_c1, final_offset_c2 = offset2, offset1

    return final_angle, final_spacing_p, final_spacing_q, final_offset_c1, final_offset_c2, np.array(valid_peaks), magnitude_spectrum


if __name__ == '__main__':
    # --- Ground Truth Parameters ---
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    TRUE_THETA = 35.0
    TRUE_P = 25
    TRUE_Q = 40
    TRUE_C1 = 5  # Offset for the first set of lines
    TRUE_C2 = 10 # Offset for the second set of lines
    ADD_NOISE = True

    # --- Generate or Load Image ---
    grid_image = create_tilted_grid(IMG_WIDTH, IMG_HEIGHT, TRUE_THETA, TRUE_P, TRUE_Q, TRUE_C1, TRUE_C2, add_noise=ADD_NOISE)
    
    # --- Extract Parameters ---
    angle, p, q, c1, c2, peaks, spectrum = extract_grid_parameters(grid_image)

    # --- Visualization ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(grid_image, cmap='gray')
    ax1.set_title('Input Grid Image')
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(spectrum, cmap='viridis')
    ax2.set_title('Fourier Spectrum with Detected Peaks')
    if peaks is not None:
        ax2.scatter(peaks[:, 1], peaks[:, 0], c='red', marker='x', s=50)
    ax2.set_xticks([]); ax2.set_yticks([])

    ax3 = fig.add_subplot(2, 2, (3, 4))
    ax3.axis('off')
    ax3.set_title('Extraction Results', y=0.95)
    
    if angle is not None:
        s_p, s_q = sorted((p, q))
        t_p, t_q = sorted((TRUE_P, TRUE_Q))
        
        results_text = (
            f"Ground Truth:\n"
            f"  - Angle: {TRUE_THETA:.2f}°\n"
            f"  - Spacing 1: {t_p}, Spacing 2: {t_q}\n"
            f"  - Offset 1: {TRUE_C1}, Offset 2: {TRUE_C2}\n\n"
            f"Extracted Parameters:\n"
            f"  - Angle: {angle:.2f}°\n"
            f"  - Spacing 1: {s_p:.2f}, Spacing 2: {s_q:.2f}\n"
            f"  - Offset 1: {c1:.2f}, Offset 2: {c2:.2f}"
        )
    else:
        results_text = "Parameter extraction failed."

    ax3.text(0.5, 0.6, results_text, ha='center', va='center', fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.5", fc="gray", ec="white", lw=1, alpha=0.5))

    plt.tight_layout()
    plt.show()
