import os
import cv2 as cv
import numpy as np

def Otsu_thresholding(img):
    """
    Crop the image according to the Otsu thresholding.
    input : img (numpy array) : the mammogram
    output : x, y (int) : top left coordinates, w, h (int): width and height of the crop
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cnts, _ = cv.findContours(breast_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)
    return x, y, w, h

name = "test_inb"
data_folder = os.path.join('.', 'dataset', f'{name}')  # Assuming 'Bi-rads' folder is in the current directory
output_folder = os.path.join('.', 'dataset', f'{name}_cropped')  # Output folder for cropped images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through subfolders
for subfolder in os.listdir(data_folder):
    subfolder_path = os.path.join(data_folder, subfolder)
    output_subfolder_path = os.path.join(output_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Create the corresponding output subfolder
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)
        # Iterate through images in subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(subfolder_path, filename)
                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # Load image in grayscale
                x, y, w, h = Otsu_thresholding(image)
                cropped_image = image[y:y+h, x:x+w]
                save_path = os.path.join(output_subfolder_path, filename)
                cv.imwrite(save_path, cropped_image)  # Save cropped image
                print(f"Processed and saved: {save_path}")