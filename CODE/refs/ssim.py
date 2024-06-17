from skimage.metrics import structural_similarity as ssim
import cv2

# Load the two input images
imageA = cv2.imread('/home/rinkesh/Desktop/data/IMGSTITCH-master/yonly/37.00_33.00_1.51.png')
imageB = cv2.imread('/home/rinkesh/Desktop/data/IMGSTITCH-master/yonly/37.00_31.00_1.51.png')

# Convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
(score, diff) = ssim(grayA, grayB, full=True)
print("SSIM: {}".format(score))
