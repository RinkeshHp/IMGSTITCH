import numpy as np
import matplotlib.pyplot as plt

from skimage.registration import phase_cross_correlation

# from skimage.registration._phase_cross_correlation import _upsampled_dft
# from scipy.ndimage import fourier_shift
from skimage import io
import cv2
import imutils

img1 = cv2.imread(
    # "/home/rinkesh/Desktop/Stiching-calibration/sacn14092023_1/2.50/43.50_13.50_3.53.png"
    "/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/38.25_13.50_3.53.png",
    cv2.IMREAD_GRAYSCALE,
)
img2 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/38.50_13.50_3.53.png",
    cv2.IMREAD_GRAYSCALE,
)
img1 = imutils.rotate_bound(img1, 90)
img2 = imutils.rotate_bound(img2, 90)
print(img1.shape)
print(img2.shape)
shift, error, diffphase = phase_cross_correlation(img1, img2)
fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(img1, cmap="gray")
ax1.set_axis_off()
ax1.set_title("Reference img1")

ax2.imshow(img2.real, cmap="gray")
ax2.set_axis_off()
ax2.set_title("Offset img1")

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = (
    np.fft.fft2(img1) * np.fft.fft2(img2).conj()
)  # multiplication in frequency domain is convolution in spatial domain
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

plt.show()

print(f"Detected pixel offset (y, x): {shift}, error: {error},diffphase: {diffphase}")
