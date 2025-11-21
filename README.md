https://drive.google.com/drive/folders/1XlyLJL4DxY-o-H6nQwjxIqgr-QMdvxIy


import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt

# ---- Load image ----
src = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)  # Grayscale for frequency domain processing
if src is None:
    raise Exception("Image not found")

# ---- 1. Convert to float and apply FFT ----
f = np.float32(src)  # Convert to float
F = np.fft.fft2(f)  # 2D FFT
F_shift = np.fft.fftshift(F)  # Shift zero-frequency to the center

# ---- 2. Create Butterworth High-Pass Filter ----
rows, cols = src.shape
crow, ccol = rows // 2, cols // 2

# Butterworth filter parameters
D0 = 30    # Cutoff frequency (you can adjust this)
n = 2      # Filter order (you can adjust this too)

# Create distance matrix
u_coords = np.arange(rows)
v_coords = np.arange(cols)
u, v = np.meshgrid(u_coords, v_coords, indexing='ij') # Changed to use indexing='ij'
D = np.sqrt((u - crow)**2 + (v - ccol)**2)

# To avoid division by zero, add a small epsilon value
D[D == 0] = 1e-6  # Prevent division by zero at the center

# Butterworth high-pass filter formula
H = 1 / (1 + (D0 / D)**(2 * n))

# ---- 3. Apply the Butterworth filter in the frequency domain ----
G_shift = F_shift * H

# ---- 4. Inverse FFT ----
G = np.fft.ifftshift(G_shift)  # Shift back
Hp = np.abs(np.fft.ifft2(G))   # Inverse FFT to get the filtered image

# ---- 5. Normalize for display ----
Hp_display = cv2.normalize(Hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ---- Show Results ----
print("High-Pass Filter (Butterworth, Frequency Domain)")

# Display original and filtered image side by side
cv2_imshow(np.hstack((src, Hp_display)))  # Display original and filtered side by side
