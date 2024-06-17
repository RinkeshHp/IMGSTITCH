import numpy as np
import pywt
import cv2
from skimage import img_as_float
from matplotlib import pyplot as plt

# Hard thresholding function
def hard_thresholding(data, threshold):
    data[np.abs(data) < threshold] = 0
    return data

# Load the image
image = cv2.imread('/home/rinkesh/Desktop/data/IMGSTITCH-master/yonly/76/86.72_25.19_2.30.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = img_as_float(image)

# Perform wavelet transform
wavelet = 'db1'
coeffs = pywt.wavedec2(image, wavelet, level=100)

# Estimate the noise standard deviation
sigma = np.median(np.abs(coeffs[-1])) / 0.6745

# Apply hard thresholding to the detail coefficients
threshold = sigma * np.sqrt(2 * np.log(image.size))
coeffs_thresh = [coeffs[0]] + [tuple(hard_thresholding(detail, threshold) for detail in level) for level in coeffs[1:]]

# Reconstruct the denoised image using the thresholded coefficients
denoised_image = pywt.waverec2(coeffs_thresh, wavelet)

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.show()
