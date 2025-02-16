import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import os
from huggingface_hub import HfApi, get_token
from datasets import config
import pandas as pd

# Configure faster download settings
def configure_fast_downloads():
    """Configure Hugging Face for faster downloads using Lambda Labs cache"""
    # Set higher download speeds and concurrent downloads
    config.MAX_SHARD_SIZE = "1000GB"

    # Use Lambda Labs' S3 cache for faster downloads
    os.environ["HF_ENDPOINT"] = "https://d3-us-east-1.lambda-gateway.net/v1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DOWNLOAD_WORKERS"] = "144"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Set cache directory to local SSD for faster access
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

def calculate_box_area_percentage(box_area, image_size):
    """Calculates the percentage of the image area that the box area occupies."""
    image_area = image_size[0] * image_size[1]
    return (box_area / image_area) * 100 if image_area > 0 else 0

def load_and_save_dataset(
    dataset_name,
    split,
    false_positive_csv=None,
    false_negative_csv=None,
    min_box_area_percentage=5.0,
):

    # Load relabeling data if needed
    false_positives = set()
    false_negatives = {}
    use_relabeling = false_positive_csv is not None or false_negative_csv is not None

    if use_relabeling:
        if false_positive_csv:
            if not os.path.exists(false_positive_csv):
                raise FileNotFoundError(f"False positive CSV file not found: {false_positive_csv}")
            fp_df = pd.read_csv(false_positive_csv)
            false_positives = set(fp_df['filename'].tolist())
            print(f"Loaded {len(false_positives)} false positives from {false_positive_csv}")

        if false_negative_csv:
            if not os.path.exists(false_negative_csv):
                raise FileNotFoundError(f"False negative CSV file not found: {false_negative_csv}")
            fn_df = pd.read_csv(false_negative_csv)
            false_negatives = dict(zip(fn_df['filename'], fn_df['largest_person_area']))
            print(f"Loaded {len(false_negatives)} false negatives from {false_negative_csv}")

    # Get dataset size (total number of images in dataset)
    total_examples = 5760428

    print(f"Downloading entire dataset with {total_examples} examples without sharding")
    print(f"Using {'relabeled' if use_relabeling else 'original'} labels")

    # Get authentication token
    token = get_token()
    if token is None:
        raise ValueError("Please login to Hugging Face using `huggingface-cli login` first")

    configure_fast_downloads()

    # Load the entire dataset at once without streaming for faster download
    print("Loading entire dataset at once (non-streaming mode)...")
    dataset = load_dataset(dataset_name, split=split, streaming=False, token=token, trust_remote_code=True, num_proc=144)

    # Create directories to save images
    base_dir = f"full_dataset_human_vs_nohuman_relabeled" if use_relabeling else "full_dataset_human_vs_nohuman_original"
    relabeled_shard_dir = base_dir # simplified directory name

    # Create subdirectories
    for label in ['human', 'no-human']:
        os.makedirs(os.path.join(relabeled_shard_dir, label), exist_ok=True)

    # Stats for reporting
    relabeled_count = 0
    total_count = 0
    flipped_labels = 0
    small_person_count = 0

    # Iterate over the dataset
    for i, item in enumerate(dataset): # iterate over entire dataset
        img = item['image']
        filename = item['filename']
        original_label = item['person']
        new_label = original_label # default label is original

        # Relabeling logic
        if use_relabeling:
            # Check if this is a false positive (human → no-human)
            if filename in false_positives and original_label == 1:
                new_label = 0
                flipped_labels += 1

            # Check if this is a false negative (no-human → human)
            elif filename in false_negatives and original_label == 0:
                area = false_negatives[filename]
                area_percentage = calculate_box_area_percentage(area, (img.width, img.height))

                if area_percentage >= min_box_area_percentage:
                    new_label = 1
                    flipped_labels += 1
                else:
                    small_person_count += 1

        # Save relabeled version only
        save_label = new_label if use_relabeling else original_label
        relabeled_path = os.path.join(relabeled_shard_dir, 'human' if save_label == 1 else 'no-human', filename)
        img.save(relabeled_path)

        total_count += 1
        if i % 1000 == 0: # increased print frequency as no shard limit anymore
            print(f"Processed {total_count} images...")

    # Print final statistics
    print(f"\nProcessing complete for entire dataset:") # updated print statement
    print(f"Total images processed: {total_count}")
    if use_relabeling:
        print(f"Images with relabeling data: {relabeled_count}")
        print(f"False positives corrected: {flipped_labels} (human → no-human)")
        print(f"False negatives corrected: {flipped_labels} (no-human → human)")
        print(f"Small detections ignored: {small_person_count}")


    # Print save locations
    print(f"\nDataset saved to:") # updated print statement
    saved_dir = relabeled_shard_dir
    print(f"Directory: {saved_dir}")
    return saved_dir


# Main function to process images and save results
def main():
    # Configure fast downloads before loading dataset
    configure_fast_downloads()

    # Get Hugging Face token
    token = get_token()
    if token is None:
        raise ValueError("Please login to Hugging Face using `huggingface-cli login` first")

    # Download and save the entire dataset in ImageNet format
    saved_dir = load_and_save_dataset( # renamed function call
        dataset_name="Harvard-Edge/Wake-Vision-Train-Large",
        split="train_large",
    )
    print(f"Dataset saved to: {saved_dir}") # print final save directory from main


if __name__ == '__main__':
    main()