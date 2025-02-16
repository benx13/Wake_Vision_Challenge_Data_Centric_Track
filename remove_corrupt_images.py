import os
import cv2
from tqdm import tqdm

def remove_corrupt_images(directory):
    """
    Scan through a directory and remove any images that can't be opened with OpenCV.
    
    Args:
        directory (str): Path to the directory containing images
    """
    # Keep track of statistics
    total_images = 0
    corrupt_images = 0
    
    # Walk through all files in directory and subdirectories
    print("Scanning for corrupt images...")
    for root, _, files in os.walk(directory):
        for filename in tqdm(files):
            # Check if file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                total_images += 1
                image_path = os.path.join(root, filename)
                
                try:
                    # Try to read the image with OpenCV
                    img = cv2.imread(image_path)
                    
                    # Check if image is None or empty
                    if img is None or img.size == 0:
                        print(f"\nRemoving corrupt image: {image_path}")
                        os.remove(image_path)
                        corrupt_images += 1
                        
                except Exception as e:
                    print(f"\nError processing {image_path}: {str(e)}")
                    print(f"Removing corrupt image: {image_path}")
                    os.remove(image_path)
                    corrupt_images += 1

    # Print summary
    print("\nScan complete!")
    print(f"Total images processed: {total_images}")
    print(f"Corrupt images removed: {corrupt_images}")

if __name__ == "__main__":
    # You can modify this path to point to your ImageNet format directory
    dataset_dir = "dataset"  # Change this to your dataset directory
    remove_corrupt_images(dataset_dir) 