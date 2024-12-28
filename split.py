import os
import shutil
import random

def split_images(input_folder, output_folder, split_ratio=(0.7, 0.3, 0.0), seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Create train, validation, and test directories
    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'validation')
    test_dir = os.path.join(output_folder, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get the list of subdirectories (benign, malignant, normal)
    subdirectories = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

    # Iterate through each subdirectory
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(input_folder, subdirectory)

        # Get the list of images in the subdirectory
        images = [f for f in os.listdir(subdirectory_path) if (f.endswith('.png') or f.endswith('.jpg'))]

        # Shuffle the list of images
        random.shuffle(images)

        # Calculate split indices
        total_images = len(images)
        train_split = int(split_ratio[0] * total_images)
        val_split = int((split_ratio[0] + split_ratio[1]) * total_images)

        # Split the images and copy to the respective directories
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        # Keep track of copied images to avoid duplicates
        copied_images = set()

        for image in train_images:
            if image not in copied_images:
                os.makedirs(os.path.join(train_dir, subdirectory), exist_ok=True)
                shutil.copy2(os.path.join(subdirectory_path, image), os.path.join(train_dir, subdirectory, image))
                copied_images.add(image)

        for image in val_images:
            if image not in copied_images:
                os.makedirs(os.path.join(val_dir, subdirectory), exist_ok=True)
                shutil.copy2(os.path.join(subdirectory_path, image), os.path.join(val_dir, subdirectory, image))
                copied_images.add(image)

        for image in test_images:
            if image not in copied_images:
                os.makedirs(os.path.join(test_dir, subdirectory), exist_ok=True)
                shutil.copy2(os.path.join(subdirectory_path, image), os.path.join(test_dir, subdirectory, image))
                copied_images.add(image)

# Example usage
name = "Dataset_2_cropped"
input_folder = os.path.join('.', 'dataset', f'{name}')
output_folder = os.path.join('.', 'dataset', f'{name}_split')
split_images(input_folder, output_folder)