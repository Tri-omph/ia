import os
import random
import shutil
from enum import Enum
import numpy as np

from PIL import Image
from PIL import ImageEnhance

def clear_folders(folders):
    for f in folders:
        # Check if the directories exist
        if os.path.exists(f):
            try:
                # Try removing the directory (even if non-empty)
                shutil.rmtree(f)
            except OSError as e:
                print(f"Error while removing {f}: {e}")

def split_data(root_dir, train_dir, test_dir, train_size=0.7):
    # Go through each folder in the root directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        print(f"Now randomizing: {folder_path}")
        
        if os.path.isdir(folder_path):
            # Create directories for train and test if they don't exist
            os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
            os.makedirs(os.path.join(test_dir, folder), exist_ok=True)
            
            # List all images in the folder
            images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            # Separate augmented images (they will go to the train set)
            n_images = [f for f in images if '_aug' in f]
            other_images = [f for f in images if '_aug' not in f]
            
            # Shuffle the non-_n.png images
            random.shuffle(other_images)
            
            # Calculate the number of train images
            num_train = int(len(other_images) * train_size)
            train_images = other_images[:num_train]
            test_images = other_images[num_train:]
            
            # Move _n.png images to the train directory
            for image in n_images:
                shutil.copyfile(os.path.join(folder_path, image), os.path.join(train_dir, folder, image))
            
            # Move train images to the train directory
            for image in train_images:
                shutil.copyfile(os.path.join(folder_path, image), os.path.join(train_dir, folder, image))
            
            # Move test images to the test directory
            for image in test_images:
                shutil.copyfile(os.path.join(folder_path, image), os.path.join(test_dir, folder, image))


class Effect(Enum):
    LIGHT = 1
    NOISE = 2
    NONE = 3

class Change(Enum):
    ROTATION = 1
    FLIP_H = 2
    FLIP_V = 3

def add_gaussian_noise(image_np, mean=0, sigma=25):
        """
        Adds Gaussian noise to an image.
        
        Parameters:
            image_np: numpy array of the image.
            mean: Mean of the Gaussian noise.
            sigma: Standard deviation of the Gaussian noise.
            
        Returns:
            Noisy image as a numpy array.
        """
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(mean, sigma, image_np.shape)
        
        # Add the noise to the image and clip to valid range
        noisy_image = np.clip(image_np + gaussian_noise, 0, 255).astype(np.uint8)
        
        return noisy_image

def change_parameters_img(effect, change, image_path, output_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            image_np = np.array(img)
            match(effect):
                case Effect.LIGHT:
                    brightness_factor = random.uniform(0.5, 1.5)  # Increase or decrease brightness by 50%
                    enhancer = ImageEnhance.Brightness(img)
                    new_img = enhancer.enhance(brightness_factor)
                case Effect.NOISE:
                    pass
                    flipped_img = add_gaussian_noise(image_np)
                    new_img = Image.fromarray(flipped_img)
                case _:
                    new_img = img
            match(change):
                case Change.ROTATION:
                    """rotation = random.randint(1,359)
                    new_img = new_img.rotate(rotation)"""
                    new_img = img
                case Change.FLIP_H:
                    new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                case Change.FLIP_V:
                    new_img = new_img.transpose(Image.FLIP_TOP_BOTTOM)
            
            new_img.save(output_path)
    except Exception as e:
        pass
        #print(f"Error changing image {image_path}: {e}")

def augment_data(unaugmented_raw_dir, raw_dir, step_to_change = 1):
    """ Augment with a random combination all images in the subfolders of unaugmented_raw_dir 
    and output them to raw_dir, keeping the directory structure intact.

    Args:
        unaugmented_raw_dir (str): Path to the original directory with images.
        raw_dir (str): Path to the directory where flipped images will be saved.
        step_to_change (int): 
    """

    print("BEGIN : Augmentation of the dataset")
    counter = 1
    for root, dirs, files in os.walk(unaugmented_raw_dir):
        print(f"Now augmenting {root}")
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Get the relative path for the file in unaugmented_raw_dir
                relative_path = os.path.relpath(root, unaugmented_raw_dir)
                # Create the corresponding path in raw_dir
                new_dir = os.path.join(raw_dir, relative_path)
                # Create the directory if it doesn't exist
                os.makedirs(new_dir, exist_ok=True)
                # Define the full path for the original and flipped images
                image_path_old = os.path.join(root, file)

                name, ext = os.path.splitext(file)
                shutil.copyfile(os.path.join(root, file), os.path.join(new_dir, file))
                if counter == step_to_change:
                    image_path_new = os.path.join(new_dir, f"{name}_aug{ext}")

                    # Take randomly an effect and a change
                    effect = random.choice(list(Effect))
                    change = random.choice(list(Change))

                    change_parameters_img(effect, change, image_path_old, image_path_new)

            counter +=1
            if counter > step_to_change:
                counter = 1

if __name__ == "__main__":
    # Images not included to reduce bandwidth needed to import
    unaugmented_raw_dir = '../../images/Final'
    raw_dir = '../../images/final_raw'
    train_dir = '../../images/final_train'
    val_dir = '../../images/final_val'

    # Only when first generating the extra images
    #clear_folders([train_dir, val_dir, raw_dir]) # Clear folder before adding new data
    #augment_data(unaugmented_raw_dir, raw_dir)
    #split_data(raw_dir, train_dir, val_dir)

    # When just need to re-randomize the datasets 
    clear_folders([train_dir, val_dir])
    split_data(raw_dir, train_dir, val_dir)