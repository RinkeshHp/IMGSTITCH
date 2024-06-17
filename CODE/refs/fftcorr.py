import cv2
import numpy as np
from matplotlib import pyplot as plt


def register_translation(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the 2D FFT of both images
    f1 = np.fft.fft2(img1_gray)
    f2 = np.fft.fft2(img2_gray)

    # Compute the cross-power spectrum
    cross_power_spectrum = (f1 * f2.conjugate()) / np.abs(f1 * f2.conjugate())

    # Compute the inverse FFT of the cross-power spectrum
    cross_correlation = np.fft.ifft2(cross_power_spectrum)

    # Find the peak in the cross-correlation
    max_idx = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )

    # Calculate the translation
    shifts = np.array(max_idx)
    shifts[shifts > np.array(cross_correlation.shape) // 2] -= np.array(
        cross_correlation.shape
    )

    return shifts


# Load the two images
img1 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/36.50_13.50_3.53.png"
)
img2 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/36.75_13.50_3.53.png"
)

# Estimate the translation parameters
translation = register_translation(img1, img2)
print(f"Translation: {translation}")

# Apply the translation to img2
rows, cols = img2.shape[:2]
M = np.float32([[1, 0, translation[1]], [0, 1, translation[0]]])
img2_translated = cv2.warpAffine(img2, M, (cols, rows))

# Display the original and aligned images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Image 1")
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Image 2")
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Image 2 Translated")
plt.imshow(cv2.cvtColor(img2_translated, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
