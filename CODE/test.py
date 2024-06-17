import os
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation
import time
import re


def read_in_memory(folder_path, crop_percent={"x": 0, "y": 0}):
    """read all images from folder and hold them in memory(numpy array) for further processing Optionally performs cropping as well if specified as {"x": %cropping in width, "y": %cropping in height}. Returned data structure (img_data) will be used for output as well."""
    img_data = {
        "ip": {
            "x": {},
            "y": {},
        },
        "op": {
            "x": {},
            "y": {},
        },
    }

    img_ext = (".jpg", ".jpeg", ".png", ".bmp")

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(img_ext):
            # print(filename)
            # find x,y coordinates and zoom of current image and store it accordingly in img_data
            # x, y, zoom = filename.split("_")

            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            if crop_percent:
                height, width = img.shape[:2]
                # Calculate crop amount in pixels based on percentages
                crop_amount_x = int(
                    width * crop_percent["x"] / 200
                )  # Divide by 2 for equal cropping from both sides
                crop_amount_y = int(
                    height * crop_percent["y"] / 200
                )  # Divide by 2 for equal cropping from both sides

                # Define cropping coordinates
                start_x = crop_amount_x
                end_x = width - crop_amount_x
                start_y = crop_amount_y
                end_y = height - crop_amount_y

                # Crop the image
                img = img[start_y:end_y, start_x:end_x]

            img_data["ip"][filename] = {}
            img_data["ip"][filename]["raw"] = img

    return img_data


def read_in_memory_2d(folder_path):
    """read all images from folder and hold them in memory(numpy array) for further processing . Returned data structure (img_data) will be used for output as well."""
    img_data = {
        "ip": {
            "x": {},
            "y": {},
        },
        "op": {
            "x": {},
            "y": {},
        },
    }

    img_ext = (".jpg", ".jpeg", ".png", ".bmp")

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(img_ext):
            # find x,y coordinates and zoom of current image and store it accordingly in img_data. Stitching first in X direction(group by Y) and then in Y direction
            t = re.split(r"_|\.\D", filename)
            x, y = t[:2]
            # x, y = filename.split("_")[:2]
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)

            if y not in img_data["ip"]["y"]:
                img_data["ip"]["y"][y] = {}
            img_data["ip"]["y"][y][filename] = img

    return img_data


def get_homography_ransac(img1, img2):
    # Detect ORB key points and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

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
    homography_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    homography_matrix[0][2] = -1 * homography_matrix[0][2]
    homography_matrix[1][2] = -1 * homography_matrix[1][2]
    # print(f"homo ({homography_matrix[1][2]}, {homography_matrix[0][2]})")
    return homography_matrix


def get_homography_pxc(img1, img2):
    shift, _, _ = phase_cross_correlation(img1, img2)
    # print(f"pxc {shift}")
    return np.array([[1, 0, shift[1]], [0, 1, shift[0]], [0, 0, 1]])


def compute_homography_matrix(img_data, sample_count=5):
    """Compute homography matrix based on random sampling of image pairs and using 2 methods i.e. phase cross correlation and homography estimation using RANSAC. Applicable only for batch of images in one direction"""

    homography = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    img_list = list(img_data["ip"].keys())
    img_indices = np.arange(0, len(img_list) - 1)
    sampled_imgs = [0]
    if len(img_list) > 2:
        sampled_imgs = np.random.choice(img_indices, sample_count, replace=False)
    img_pairs = []
    for img_index in sampled_imgs:
        """create image pairs"""
        img_pairs.append(
            (
                img_data["ip"][img_list[img_index]]["raw"],
                img_data["ip"][img_list[img_index + 1]]["raw"],
            )
        )
    homography = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for img_pair in img_pairs:
        h = get_homography_ransac(img_pair[0], img_pair[1])
        # print(f"ransac {h}")
        homography = homography + h

    # for img_pair in img_pairs:
    #     h = get_homography_pxc(img_pair[0], img_pair[1])
    #     # print(f"pxc {h}")
    #     homography = homography + h
    homography = homography / (sample_count)
    # print(homography)

    return homography


def stitcher(img_data, homography):
    """This will stitch the images based on the homography (translation only)"""
    # Calculate dimensions of stitched image
    translation = {
        "x": round(homography[0][2]),
        # "x": 0,
        "y": round(homography[1][2]),
        # "y": 0,
    }
    n = len(img_data["ip"])
    # calculate origin, since translations can be -ve, we need to shift the image coordinates
    ox, oy = 0, 0
    if translation["x"] < 0:
        ox = (1 - n) * (translation["x"])
    if translation["y"] < 0:
        oy = (1 - n) * (translation["y"])
    filename_1st = list(img_data["ip"].keys())[0]
    height, width = (
        img_data["ip"][filename_1st]["raw"].shape[0]
        + int((n - 1) * abs(translation["y"])),
        img_data["ip"][filename_1st]["raw"].shape[1]
        + int((n - 1) * abs(translation["x"])),
    )
    sub_height, sub_width, _ = img_data["ip"][filename_1st]["raw"].shape
    stitched_image = np.zeros((height, width, 3), dtype=np.uint8)
    i = 0
    for img in img_data["ip"].values():
        # Paste the images at their respective positions

        # define 4 corners for position of image
        yl = i * translation["y"]
        yu = yl + sub_height
        xl = i * translation["x"]
        xu = xl + sub_width
        stitched_image[oy + yl : oy + yu, ox + xl : ox + xu] = img["raw"]
        i = i + 1
    cv2.imshow("stitched image", stitched_image)
    img_data["op"] = stitched_image
    cv2.imwrite("stitched image.png", stitched_image)

    return img_data


# print(compute_homography_matrix(img_data, 2))
img_data = read_in_memory(
    # "/home/rinkesh/Desktop/FCRIT_FYP-main/2.53",
    # "/home/rinkesh/Desktop/FCRIT_FYP-main/0.05_step_5x_pt1/5.00"
    "/home/rinkesh/Desktop/data/IMGSTITCH-master/testx"
)

start = time.time()
h = compute_homography_matrix(img_data, sample_count=10)
stitcher(img_data, h)
print(f"{time.time()-start}")

cv2.waitKey(0)
cv2.destroyAllWindows()
