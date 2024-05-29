import cv2
import numpy as np


def translation_only_stitcher(img_list, translation):
    """simplified stitcher(only translation)"""

    # Calculate dimensions of stitched image
    n = len(img_list)
    height, width = (
        img_list[0].shape[0] + int((n - 1) * translation["y"]),
        img_list[0].shape[1] + int((n - 1) * translation["x"]),
    )
    sub_height, sub_width, _ = img_list[0].shape
    stitched_image = np.empty((height, width, 3), dtype=np.uint8)
    for i in range(n):
        # Paste the images at their respective positions

        # define 4 corners for position
        yl = i * translation["y"]
        yu = yl + sub_height
        xl = i * translation["x"]
        xu = xl + sub_width
        # print(img_list[0].shape)
        stitched_image[yl:yu, xl:xu] = img_list[i]
    cv2.imshow("stitched image", stitched_image)

    return None


# Load images
img1 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.50_13.50_3.53.png"
    # "2.jpg"
)
img2 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.70_13.50_3.53.png"
    # "2.jpg"
)

homography = np.array(
    [
        [1.0, 0, -95],
        [0, 1, 0],
        [0, 0, 1],
    ]
)

# Get the size of the images
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

# Compute the corners of img2 after transformation
corners2 = np.array(
    [[0, 0], [width2, 0], [width2, height2], [0, height2]], dtype=np.float32
).reshape(-1, 1, 2)
transformed_corners2 = cv2.perspectiveTransform(corners2, homography)

# Find the bounding box of the combined image
corners1 = np.array(
    [[0, 0], [width1, 0], [width1, height1], [0, height1]], dtype=np.float32
).reshape(-1, 1, 2)
all_corners = np.vstack((corners1, transformed_corners2))

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Translation matrix to shift the image to the positive quadrant
translation_dist = [-x_min, -y_min]
# print(translation_dist)
translation_matrix = np.array(
    [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
)
# print(translation_matrix)

# Warp the second image using the homography matrix
warped_img2 = cv2.warpPerspective(
    img2, translation_matrix.dot(homography), (x_max - x_min, y_max - y_min)
)
# cv2.imshow(
#     "crop",
#     img1[0 : height1 + int(homography[1][2]), 0 : width1 + int(homography[0][2]), :],
# )

# Paste the first image into the panorama
panorama = np.empty((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
panorama[
    translation_dist[1] : height1 + translation_dist[1],
    translation_dist[0] : width1 + translation_dist[0],
] = img1
# cv2.imshow("result1", panorama)
# Blend the warped second image into the panorama
mask = (warped_img2 > 0).astype(np.uint8)
# print(mask)
panorama = cv2.addWeighted(panorama, 0.5, warped_img2, 0.5, 0)
# panorama[
#     0 : height1 + int(homography[1][2]),
#     0 :  + int(homography[0][2]),
#     :,
# ] = img1[0 : height1 + int(homography[1][2]), 0 : width1 + int(homography[0][2]), :]

# cv2.equalizeHist(panorama[:, :, 0])

# cv2.imshow("result", panorama)
# print(panorama.shape)

translation_only_stitcher(
    [
        img1,
        img2,
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.90_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/37.10_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/37.30_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/37.50_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/37.70_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/37.90_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/38.10_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/38.30_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/38.50_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/38.70_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/38.90_13.50_3.53.png"
            # "2.jpg"
        ),
        cv2.imread(
            "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/39.10_13.50_3.53.png"
            # "2.jpg"
        ),
    ],
    {"x": 95, "y": 0},
)
cv2.waitKey(0)
cv2.destroyAllWindows()
