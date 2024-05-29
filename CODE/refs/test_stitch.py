import cv2
import numpy as np
import imutils

# Load images
img1 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.70_13.50_3.53.png"
    # "2.jpg"
)
img2 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.50_13.50_3.53.png"
    # "2.jpg"
)
# img1 = img1[94 * 1 :, 0:-1]
# cv2.imshow("img1",img1)


# Example homography matrix (replace with your actual homography matrix)
H = np.array(
    [
        [1.0, 0, -95 * 1],
        [0, 1, +1],
        [0, 0, 1],
    ]
)
# img1 = imutils.rotate_bound(img1, 90)
# Get the shape of the images
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

# Get the corners of the second image
corners = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]])
transformed_corners = cv2.perspectiveTransform(np.float32([corners]), H)
# print(transformed_corners)

# Find the dimensions of the resulting stitched image
all_corners = np.vstack((corners, transformed_corners[0]))
# print(all_corners)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
# print(x_min, y_min)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
# print(x_max, y_max)
translation_dist = [-x_min, -y_min]
# print(translation_dist)

# Translate the homography to fit the result in the positive range
H_translation = np.array(
    [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
)

print(H_translation.dot(H))
# Warp the second image
# result = np.empty((y_max - y_min, x_max - x_min, 3))

# result[0:height1, 0:width1] = -img1

result = cv2.warpAffine(
    img2,
    H_translation.dot(H)[:2, :],
    (
        x_max - x_min,
        y_max - y_min,
    ),
)
cv2.imshow("intermediate_result", result)

# Place the first image in the result
result[
    translation_dist[1] : height1 + translation_dist[1],
    translation_dist[0] : width1 + translation_dist[0],
] = img1


# Function to blend images
def blend_images(image1, image2, alpha=0.5):
    return cv2.addWeighted(image1, alpha, image2, 1 / alpha, 0)


# Create a mask for blending
mask = np.zeros_like(result, dtype=np.uint8)
mask[
    translation_dist[1] : height1 + translation_dist[1],
    translation_dist[0] : width1 + translation_dist[0],
] = img1

# Blend the images
blended_result = blend_images(result, mask)

# Show the result
cv2.imshow("Stitched Image", result)
# cv2.imwrite("2.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
