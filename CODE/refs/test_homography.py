import cv2
import numpy as np
import imutils

# Load the images
image1 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.70_13.50_3.53.png"
    # "2.jpg"
    # "registered.jpg",
)
image2 = cv2.imread(
    "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1/36.50_13.50_3.53.png"
)
# image1 = image1[200 : image1.shape[0] - 200, 200 : image1.shape[1] - 200]
# image2 = image2[200 : image2.shape[0] - 200, 200 : image2.shape[1] - 200]
# image1 = cv2.resize(
#     image1,
#     (int(0.4 * image1.shape[1]), int(0.4 * image1.shape[0])),
#     interpolation=cv2.INTER_NEAREST,
# )

# image2 = cv2.resize(
#     image2,
#     (int(0.4 * image2.shape[1]), int(0.4 * image2.shape[0])),
#     interpolation=cv2.INTER_NEAREST,
# )
# grab the dimensions of the image and calculate the center of the
# image
# (h, w) = image1.shape[:2]
# (cX, cY) = (w // 2, h // 2)
# rotate our image by 45 degrees around the center of the image
# M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
# image1 = cv2.warpAffine(image1, M, (w, h))
# image2 = cv2.warpAffine(image2, M, (w, h))

# image1 = imutils.rotate_bound(image1, 90)
# image2 = imutils.rotate_bound(image2, 90)

# cv2.imshow("image1", image1)
# cv2.imshow("image2", image2)
# cv2.waitKey(0)


# Detect ORB key points and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Compute homography
homography_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# Use homography to warp image2 to image1
height, width, channels = image1.shape
print(homography_matrix)
registered_image = cv2.warpPerspective(image2, homography_matrix, (width, height))
# cv2.imwrite("registered.jpg", registered_image)

# Display the result
cv2.imshow("Registered Image", registered_image)
cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
