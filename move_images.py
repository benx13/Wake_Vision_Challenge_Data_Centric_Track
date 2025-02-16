import os
import csv
import argparse
import shutil
from tqdm import tqdm

def move_misplaced_images(csv_file, dir1, dir2):
    # Ensure both directories exist
    if not os.path.isdir(dir1):
        print(f"Error: {dir1} is not a directory or doesn't exist.")
        return
    if not os.path.isdir(dir2):
        print(f"Error: {dir2} is not a directory or doesn't exist.")
        return

    # Read the list of images from the CSV file
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        # Only process rows that contain the 'filename' key
        csv_images = [row['filename'] for row in reader if 'filename' in row]

    # List files that are already in dir1
    dir1_files = set(os.listdir(dir1))

    found_count = 0
    moved_count = 0
    not_found_count = 0

    # For each expected image, check if it's missing from dir1 and present in dir2
    for image in tqdm(csv_images, desc="Moving images"):
        if image in dir1_files:
            found_count += 1
            # print(f"Skipping '{image}': already in {dir1}.")
        else:
            source_file = os.path.join(dir2, image)
            destination_file = os.path.join(dir1, image)
            if os.path.isfile(source_file):
                # print(f"Moving '{image}' from {dir2} to {dir1}.")
                shutil.move(source_file, destination_file)
                moved_count += 1
            else:
                print(f"File '{image}' not found in {dir2}.")
                not_found_count += 1

    print(f"\nCompleted: Moved {moved_count} files, {not_found_count} files were not found in {dir2}, {found_count} files were already in {dir1}.")

def main():
    parser = argparse.ArgumentParser(
        description='Move misplaced images from dir2 to dir1 based on a CSV list of images that should be in dir1.'
    )
    parser.add_argument('csv_file', help='Path to the CSV file listing expected images (with header "filename").')
    parser.add_argument('dir1', help='Directory where images are supposed to be located.')
    parser.add_argument('dir2', help='Directory where extra images might have been placed.')
    args = parser.parse_args()

    move_misplaced_images(args.csv_file, args.dir1, args.dir2)

if __name__ == '__main__':
    main()