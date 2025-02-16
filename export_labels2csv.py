import json
import glob
import os
import csv
from tqdm import tqdm

def calculate_area(box):
    """Calculates the area of a bounding box."""
    return int(round((box[2] - box[0]) * (box[3] - box[1]), 0))

def combine_json_files_to_separate_csvs():
    # Get all json files in all shard folders
    json_pattern = os.path.join('results', 'shard_*', 'images_*-*.json')
    json_files = glob.glob(json_pattern)

    # Prepare output data - CSV files
    false_negatives_output_file = 'mcunet/false_negatives.csv'
    false_positives_output_file = 'mcunet/false_positives.csv'

    with open(false_negatives_output_file, 'w', newline='') as fn_csvfile, \
         open(false_positives_output_file, 'w', newline='') as fp_csvfile:

        fn_csv_writer = csv.writer(fn_csvfile)
        fp_csv_writer = csv.writer(fp_csvfile)

        # Write header rows
        fn_csv_writer.writerow(['image_name', 'largest_person_area'])
        fp_csv_writer.writerow(['image_name'])

        total_images_processed = 0

        # Process each JSON file
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Process each image in the json file
                for image_name, image_data in data.items():
                    original_label = image_data.get('original_label')
                    has_person_detection = False # Flag to check for person detection
                    largest_person_area = 0

                    if 'objects' in image_data:
                        for obj in image_data['objects']:
                            if obj['label'] == 'person' and obj['confidence'] > 0.5:
                                has_person_detection = True
                                area = calculate_area(obj['box'])
                                if area > largest_person_area:
                                    largest_person_area = area

                    if original_label == 0 and has_person_detection:
                        # False Negative (0 to 1 flip) - save to false_negatives.csv
                        fn_csv_writer.writerow([image_name, largest_person_area])
                    elif original_label == 1 and not has_person_detection:
                        # False Positive (1 to 0 flip) - save to false_positives.csv
                        fp_csv_writer.writerow([image_name]) # Only image_name for false positives

                    total_images_processed += 1

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue

    print(f"\nProcessing complete.")
    print(f"False Negative images (0 to 1 flip) saved to: {false_negatives_output_file}")
    print(f"False Positive images (1 to 0 flip) saved to: {false_positives_output_file}")
    print(f"Total images processed: {total_images_processed}")

if __name__ == "__main__":
    combine_json_files_to_separate_csvs()