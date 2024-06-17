import os
import cv2
import numpy as np
import re
from operator import itemgetter
from itertools import groupby
import time


def read(folder_path):
    """
    img_data = {
        ip:[],
    }
    """

    img_data = {
        "ip": [],
    }
    img_ext = (".jpg", ".jpeg", ".png", ".bmp")

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith(img_ext):
            # find x,y coordinates of current image and store it accordingly in img_data
            t = re.split(r"_|\.\D", file_name)
            x, y = t[:2]

            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path)
            img_data["ip"].append(
                {
                    "file_name": file_name,
                    "x": float(x),
                    "y": float(y),
                    "img": img,
                }
            )
    img_data["ip"] = sorted(img_data["ip"], key=itemgetter("y", "x"))
    y_grouped = {
        key: list(group) for key, group in groupby(img_data["ip"], key=itemgetter("y"))
    }
    img_data["ip"] = sorted(img_data["ip"], key=itemgetter("x", "y"))
    x_grouped = {
        key: list(group) for key, group in groupby(img_data["ip"], key=itemgetter("x"))
    }

    return img_data, x_grouped, y_grouped


def get_homography_ransac(img1, img2, type):
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
    if type == "x":
        return homography_matrix[0][2]
    else:
        return homography_matrix[1][2]


def get_1d_translation(grouped_data, type):
    random_img_list = grouped_data[
        list(grouped_data.keys())[np.random.choice(np.arange(0, len(grouped_data)))]
    ]
    if len(random_img_list) < 25:
        sample_count = 2 * int(len(random_img_list) / 3) + 1
    else:
        sample_count = 25

    img_indices = np.arange(0, len(random_img_list) - 1)
    sampled_imgs = [0]
    if len(random_img_list) > 2:
        sampled_imgs = np.random.choice(img_indices, sample_count, replace=False)
    img_pairs = []
    for img_index in sampled_imgs:
        #     """create image pairs"""
        img_pairs.append(
            (
                random_img_list[img_index]["img"],
                random_img_list[img_index + 1]["img"],
            )
        )
    homography = []
    for img_pair in img_pairs:
        h = get_homography_ransac(img_pair[0], img_pair[1], type)
        #     # print(f"ransac {h}")
        homography.append(h)

    homography = round(np.median(homography))

    return homography


def stitcher_2d(x_grouped, y_grouped):
    # calculate translations in x and y using x_grouped and y_grouped

    x_translation, y_translation = 0, 0
    if len(y_grouped) > 1:
        y_translation = get_1d_translation(x_grouped, "y")
    if len(x_grouped) > 1:
        x_translation = get_1d_translation(y_grouped, "x")
    # print(x_translation, y_translation)

    ox, oy = 0, 0
    nx = len(x_grouped)
    ny = len(y_grouped)
    if x_translation < 0:
        ox = (1 - nx) * (x_translation)
    if y_translation < 0:
        oy = (1 - ny) * (y_translation)
    x_stitched_imgs = []
    for x_lbl in y_grouped:
        # stiching

        height, width = (
            y_grouped[x_lbl][0]["img"].shape[0],
            y_grouped[x_lbl][0]["img"].shape[1] + int((nx - 1) * abs(x_translation)),
        )
        sub_height, sub_width, _ = y_grouped[x_lbl][0]["img"].shape
        stitched_image = np.zeros((height, width, 3), dtype=np.uint8)
        i = 0

        yl, yu = 0, sub_height

        for x_imgs in y_grouped[x_lbl]:
            xl = i * x_translation
            xu = xl + sub_width
            stitched_image[yl:yu, ox + xl : ox + xu] = x_imgs["img"]
            i = i + 1
        x_stitched_imgs.append(stitched_image)

    height, width = (
        x_stitched_imgs[0].shape[0] + int((ny - 1) * abs(y_translation)),
        x_stitched_imgs[0].shape[1],
    )
    sub_height, sub_width, _ = x_stitched_imgs[0].shape
    stitched_image = np.zeros((height, width, 3), dtype=np.uint8)
    i = 0

    xl, xu = 0, sub_width
    for y_imgs in x_stitched_imgs:
        yl = i * y_translation
        yu = yl + sub_height
        stitched_image[oy + yl : oy + yu, xl:xu] = y_imgs
        i = i + 1

    # cv2.imshow("2dstitched", stitched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("2dstitched.png", stitched_image)
    return


img_data, x_grouped, y_grouped = read(
    # "/home/rinkesh/Desktop/FCRIT_FYP-main/0.05_step_5x_pt1/5.00")
    # "/home/rinkesh/Desktop/FCRIT_FYP-main/2.51"
    # "/home/rinkesh/Desktop/data/IMGSTITCH-master/biggest"
    "patches_named_stressss"
    # "/home/rinkesh/Desktop/data/IMGSTITCH-master/yonly/d"
    # "/home/rinkesh/Desktop/data/IMGSTITCH-master/biggest"
    # "/home/rinkesh/Desktop/data/IMGSTITCH-master/Test09012023_0/6.00"
    # "/home/rinkesh/Desktop/FCRIT_FYP-main/0.05_step_5x_pt1/5.00"
    # "patches"
)

start = time.time()
stitcher_2d(x_grouped, y_grouped)
print(f"{time.time()-start}secs")
