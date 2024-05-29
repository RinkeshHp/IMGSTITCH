import cv2
import os


def stitch_images_from_folder(folder_path, output_filename="panorama.jpg"):
    """
    Stitches images from a folder using OpenCV stitcher.

    Args:
        folder_path (str): Path to the folder containing the images.
        output_filename (str, optional): Filename for the stitched panorama image. Defaults to "panorama.jpg".

    Returns:
        bool: True if stitching is successful, False otherwise.
    """

    # Get all image paths in the folder
    image_paths = []
    for filename in sorted(os.listdir(folder_path)):
        # Check for image extensions (adjust as needed)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)

    if not image_paths:
        print("No images found in the folder!")
        return False

    # Stitch the images
    panorama = stitch_images(image_paths)

    if panorama is None:
        print("Stitching failed!")
        return False

    # Save the panorama image
    cv2.imwrite(os.path.join(folder_path, output_filename), panorama)
    print(f"Panorama saved to: {os.path.join(folder_path, output_filename)}")

    return True


def stitch_images(img_paths):
    """
    Stitches multiple images using OpenCV stitcher.

    Args:
        img_paths (list): A list of paths to the images to be stitched.

    Returns:
        cv2.Mat: The stitched panorama image.
    """

    # Create a list of OpenCV image objects
    images = []
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error reading image: {path}")
            return None
        images.append(img)

    # Create a Stitcher object
    stitcher = cv2.Stitcher.create()

    # Perform image stitching
    status, pano = stitcher.stitch(images)

    # if not status:
    #     print("Stitching failed!")
    #     return None

    return pano


# Example usage:
folder_path = "/home/rinkesh/Desktop/Stiching-calibration/take3/2.50/1"  # Replace with the path to your folder of images
success = stitch_images_from_folder(folder_path)

if success:
    print("Stitching completed successfully!")
else:
    print("An error occurred during stitching.")
