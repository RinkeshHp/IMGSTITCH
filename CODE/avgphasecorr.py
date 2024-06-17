import cv2
import numpy as np
from matplotlib import pyplot as plt

def phase_correlation_patch(img1_patch, img2_patch):
    # Compute the phase correlation to find the translation for a patch
    shift, response = cv2.phaseCorrelate(np.float32(img1_patch), np.float32(img2_patch))
    return shift, response

def register_translation_patches_weighted(img1, img2, patch_size=(100, 100), step_size=(50, 50)):
    rows, cols = img1.shape[:2]
    translations = []
    responses = []

    for i in range(0, rows - patch_size[0] + 1, step_size[0]):
        for j in range(0, cols - patch_size[1] + 1, step_size[1]):
            img1_patch = img1[i:i+patch_size[0], j:j+patch_size[1]]
            img2_patch = img2[i:i+patch_size[0], j:j+patch_size[1]]

            shift, response = phase_correlation_patch(img1_patch, img2_patch)
            translations.append(shift)
            responses.append(response)

    # Convert to numpy arrays for easier computation
    translations = np.array(translations)
    responses = np.array(responses)

    # Avoid division by zero by adding a small epsilon to the sum of responses
    eps = 1e-10
    weights = responses / (np.sum(responses) + eps)

    # Calculate the weighted average translation
    weighted_translation = np.sum(translations * weights[:, None], axis=0)

    return weighted_translation

# Load the two images
img1 = cv2.imread('/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/36.75_13.50_3.53.png')
img2 = cv2.imread('/home/rinkesh/Desktop/Stiching-calibration/take1/2.50/36.50_13.50_3.53.png')
# print(img1)

# Convert to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Estimate the translation parameters using patch-based weighted method
translation = register_translation_patches_weighted(img1_gray, img2_gray, patch_size=(100, 100), step_size=(50, 50))
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
