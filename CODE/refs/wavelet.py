import numpy as np
import pywt
import cv2
from skimage import img_as_float
from skimage.restoration import denoise_wavelet
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread(
    "/home/rinkesh/Desktop/data/IMGSTITCH-master/yonly/37.00_35.00_1.51.png"
)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = img_as_float(image)

# Perform wavelet denoising
denoised_image = denoise_wavelet(
    image,
    method="BayesShrink",
    mode="soft",
    wavelet_levels=5,
    wavelet="db1",
    rescale_sigma=True,
)

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Denoised Image")
plt.imshow(denoised_image, cmap="gray")
plt.axis("off")

plt.show()
