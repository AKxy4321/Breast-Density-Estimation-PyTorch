import cv2
import os

def apply_clahe_inplace(folder_path, clip_limit=2.0, tile_grid_size=(4, 4)):
    """
    Apply CLAHE to all images in a folder and replace them with the processed versions.

    Parameters:
        folder_path (str): Path to the folder containing images.
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (tuple): Size of the grid for the CLAHE algorithm.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping {filename}: Not a valid image.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized = clahe.apply(gray)

        # Save the processed image (replacing the original)
        cv2.imwrite(file_path, equalized)
        print(f"Processed and replaced: {file_path}")

# Example usage
folder_path = "./test_inb_cropped_split/test/Fatty"
apply_clahe_inplace(folder_path)
