import Augmentor
import os 

def create_augmentation_pipeline(input_dir, output_dir, sample_count):
    """
    Creates an augmentation pipeline and applies augmentations.

    Args:
        input_dir (str): Path to the input directory with class subfolders.
        output_dir (str): Path to save augmented images.
        sample_count (int): Number of augmented samples to generate per class.
    """
    p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
    
    p.flip_left_right(probability=0.5)                                      # Random horizontal flip
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)  # Rotate
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)         # Shear
    p.zoom(probability=0.5, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)  # Brightness adjustment
    p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)    # Contrast adjustment

    print(f"Generating {sample_count} augmented samples per class...")
    p.sample(sample_count)

    print(f"Augmented images saved to: {output_dir}")


if __name__ == "__main__":
    l = ["Dense", "Fatty"]
    for i in l:
        input_directory = os.path.join('.', i)  
        output_directory = os.path.join('.', f'{i}_augment') 
        num_samples = 2000 

        # Perform data augmentation
        create_augmentation_pipeline(input_directory, output_directory, num_samples)
