import csv
import os
import sys

def count_images_in_folder(csv_file, image_folder):
    """
    Counts the number of images from a CSV file that are present in a folder.

    Args:
        csv_file (str): Path to the CSV file containing image filenames.
        image_folder (str): Path to the folder containing the images.

    Returns:
        int: The number of images from the CSV found in the folder.
    """
    image_count = 0
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header row
        for row in reader:
            filename = row[0]
            image_path = os.path.join(image_folder, filename)
            if os.path.exists(image_path):
                image_count += 1
    return image_count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python count2.py <csv_file> <image_folder>")
        sys.exit(1)

    csv_file = sys.argv[1]
    image_folder = sys.argv[2]

    count = count_images_in_folder(csv_file, image_folder)
    print(f"Number of images found in folder: {count}")