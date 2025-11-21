

import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt


src = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)  # Grayscale for frequency domain processing
if src is None:
    raise Exception("Image not found")


f = np.float32(src)  # Convert to float
F = np.fft.fft2(f)  # 2D FFT
F_shift = np.fft.fftshift(F)  # Shift zero-frequency to the center


rows, cols = src.shape
crow, ccol = rows // 2, cols // 2


D0 = 30    # Cutoff frequency (you can adjust this)
n = 2      # Filter order (you can adjust this too)


u_coords = np.arange(rows)
v_coords = np.arange(cols)
u, v = np.meshgrid(u_coords, v_coords, indexing='ij') # Changed to use indexing='ij'
D = np.sqrt((u - crow)**2 + (v - ccol)**2)


D[D == 0] = 1e-6  # Prevent division by zero at the center


H = 1 / (1 + (D0 / D)**(2 * n))


G_shift = F_shift * H


G = np.fft.ifftshift(G_shift)  # Shift back
Hp = np.abs(np.fft.ifft2(G))   # Inverse FFT to get the filtered image


Hp_display = cv2.normalize(Hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


print("High-Pass Filter (Butterworth, Frequency Domain)")


cv2_imshow(np.hstack((src, Hp_display)))  # Display original and filtered side by side
