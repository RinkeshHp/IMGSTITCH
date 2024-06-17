import cv2

# import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


def split_image_into_patches(image, patch_size=(200, 200), step_size=(10, 10)):
    patches = []
    positions = []
    rows, cols = image.shape[:2]

    for i in range(0, rows - patch_size[0] + 1, step_size[0]):
        for j in range(0, cols - patch_size[1] + 1, step_size[1]):
            patch = image[i : i + patch_size[0], j : j + patch_size[1]]
            patches.append(patch)
            positions.append((i, j))

    return patches, positions


def save_patch(patch_info, save_dir):
    idx, (patch, (i, j)) = patch_info
    patch_filename = os.path.join(save_dir, f"{j}_{i}.png")
    cv2.imwrite(patch_filename, patch)


def save_patches_concurrently(patches, positions, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    patch_info = list(enumerate(zip(patches, positions)))

    with ThreadPoolExecutor() as executor:
        executor.map(lambda info: save_patch(info, save_dir), patch_info)


# Load the image
image_path = "/home/rinkesh/Desktop/Virus-EM-1024x365.jpg"
image = cv2.imread(image_path)

# Define patch size and step size
patch_size = (200, 200)
# step size = (y, x)
step_size = (10, 10)

# Split the image into patches
patches, positions = split_image_into_patches(image, patch_size, step_size)

# Save patches to a specified directory concurrently
save_dir = "patches_named_stressss"
save_patches_concurrently(patches, positions, save_dir)

print(f"Saved {len(patches)} patches in the directory: {save_dir}")
