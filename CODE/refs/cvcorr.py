import cv2
import numpy as np
from matplotlib import pyplot as plt

def register_translation_optimized(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the phase correlation to find the translation
    shift, response = cv2.phaseCorrelate(np.float32(img1_gray), np.float32(img2_gray))

    return shift

# Load the two images
img1 = cv2.imread('/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/36.50_13.50_3.53.png')
img2 = cv2.imread('/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/36.75_13.50_3.53.png')

# Estimate the translation parameters
translation = register_translation_optimized(img1, img2)
print(f'Translation: {translation}')

# Apply the translation to img2
rows, cols = img2.shape[:2]
M = np.float32([[1, 0, translation[1]], [0, 1, translation[0]]])
img2_translated = cv2.warpAffine(img2, M, (cols, rows))

# Display the original and aligned images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Image 1')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Image 2')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Image 2 Translated')
plt.imshow(cv2.cvtColor(img2_translated, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
