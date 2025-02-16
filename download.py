import os
import pandas as pd
from datasets import load_dataset
import shutil
import numpy as np
from PIL import Image
import logging
from huggingface_hub import get_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_fast_downloads():
    """Configure Hugging Face for faster direct downloads."""
    # Set higher download speeds and concurrent downloads
    MAX_SHARD_SIZE = "1000GB"

    # Enable HF native transfer for better speed
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable HF's fast transfer
    os.environ["USE_TORCH_DOWNLOAD"] = "0"  # Disable PyTorch backend
    os.environ["HF_HUB_DOWNLOAD_WORKERS"] = "8"  # Match CPU cores
    os.environ["DATASETS_MAX_WORKER_SIZE"] = "8"  # Match CPU cores
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Enable progress bars to monitor speed
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Optimize for high bandwidth
    os.environ["HF_HUB_DOWNLOAD_CHUNK_SIZE"] = "50MB"  # Smaller chunks for better parallelization
    os.environ["HF_HUB_DOWNLOAD_RETRY_TIMES"] = "5"
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"  # Disable memory caching
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"  # Increase timeout for large chunks

    # Set cache directory to local SSD for faster access
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

def calculate_box_area_percentage(box, image_size):
    """
    Calculate the area of a bounding box as a percentage of the image area.
    
    Args:
        box: List of [x1, y1, x2, y2] coordinates
        image_size: Tuple of (width, height)
    
    Returns:
        float: Area percentage (0-100)
    """

    
    image_area = image_size[0] * image_size[1]
    return (box / image_area) * 100

def load_and_save_shard(
    dataset_name,
    split,
    target_images_per_shard,
    shard_id=0,
    false_positive_csv=None,
    false_negative_csv=None,
    min_box_area_percentage=5.0,
    dual_save=False,
    download_all=False
):
    """
    Load and save a specific shard of the Wake Vision dataset to disk with optional relabeling.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face
        split (str): Dataset split to use
        target_images_per_shard (int): Target number of images per shard
        shard_id (int): Which shard to process
        false_positive_csv (str, optional): Path to CSV file containing false positives (human → no-human)
        false_negative_csv (str, optional): Path to CSV file containing false negatives (no-human → human) with areas
        min_box_area_percentage (float): Minimum box area as percentage of image size for false negatives
        dual_save (bool): If True and CSV files provided, save both original and relabeled versions
    
    Returns:
        str or tuple: Path to saved dataset directory, or tuple of (original_dir, relabeled_dir) if dual_save=True
    """
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
    
    if download_all:
        num_shards = 1
        target_images_per_shard = total_examples
        print(f"Downloading entire dataset with {total_examples} examples without sharding")
    else:
        # Calculate the number of shards needed
        num_shards = total_examples // target_images_per_shard
        if total_examples % target_images_per_shard != 0:
            num_shards += 1
        print(f"Dataset has {total_examples} examples. Using {num_shards} shards with ~{target_images_per_shard} images per shard.")
    
    print(f"Using {'relabeled' if use_relabeling else 'original'} labels")

    # Get authentication token
    token = get_token()
    if token is None:
        raise ValueError("Please login to Hugging Face using `huggingface-cli login` first")

    configure_fast_downloads()
    
    if download_all:
        # Load the entire dataset at once without streaming for faster download
        print("Loading entire dataset at once (non-streaming mode)...")
        dataset = load_dataset(dataset_name, split=split, streaming=False, token=token)
        sharded_dataset = dataset
    else:
        # Load the dataset in streaming mode for sharded download
        print("Loading dataset in streaming mode...")
        dataset = load_dataset(dataset_name, split=split, streaming=True, token=token)
        # Shard the dataset
        sharded_dataset = dataset.shard(num_shards=num_shards, index=shard_id)

    # Create directories to save images
    base_dir = f"shard_{shard_id}_human_vs_nohuman"
    original_shard_dir = f"{base_dir}_original" if dual_save else base_dir
    relabeled_shard_dir = f"{base_dir}_relabeled" if dual_save else base_dir
    
    # Create subdirectories for both versions
    if dual_save or not use_relabeling:
        for label in ['human', 'no-human']:
            os.makedirs(os.path.join(original_shard_dir, label), exist_ok=True)
    if use_relabeling:
        for label in ['human', 'no-human']:
            os.makedirs(os.path.join(relabeled_shard_dir, label), exist_ok=True)

    # Stats for reporting
    relabeled_count = 0
    total_count = 0
    flipped_labels = 0
    small_person_count = 0

    # Iterate over the dataset
    total_items = len(sharded_dataset) if download_all else target_images_per_shard
    for i, item in enumerate(sharded_dataset):
        if not download_all and i >= target_images_per_shard:
            break

        img = item['image']
        filename = item['filename']
        original_label = item['person']
        
        # Determine label based on whether we're using relabeling
        if use_relabeling:
            # Start with original label
            new_label = original_label
            
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
        # Save original version
        if dual_save or not use_relabeling:
            original_path = os.path.join(original_shard_dir, 'human' if original_label == 1 else 'no-human', filename)
            img.save(original_path)

        # Save relabeled version
        if use_relabeling:
            relabeled_path = os.path.join(relabeled_shard_dir, 'human' if new_label == 1 else 'no-human', filename)
            img.save(relabeled_path)

        total_count += 1
        if i % 10 == 0:
            print(f"Processed {total_count} images...")

    # Print final statistics
    print(f"\nProcessing complete for shard {shard_id}:")
    print(f"Total images processed: {total_count}")
    if use_relabeling:
        print(f"Images with relabeling data: {relabeled_count}")
        print(f"Labels flipped: {flipped_labels}")
        print(f"Small person detections ignored: {small_person_count}")
    # Print final statistics
    print(f"\nProcessing complete for shard {shard_id}:")
    print(f"Total images processed: {total_count}")
    if use_relabeling:
        print(f"False positives corrected: {flipped_labels} (human → no-human)")
        print(f"False negatives corrected: {flipped_labels} (no-human → human)")
        print(f"Small detections ignored: {small_person_count}")
    
    # Print save locations
    print(f"\nShard saved to:")
    if dual_save:
        print(f"Original labels: {original_shard_dir}")
        print(f"Relabeled version: {relabeled_shard_dir}")
        return original_shard_dir, relabeled_shard_dir
    else:
        saved_dir = relabeled_shard_dir if use_relabeling else original_shard_dir
        print(f"Directory: {saved_dir}")
        return saved_dir

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Download and preprocess dataset")
    parser.add_argument("--dataset", type=str, default="Harvard-Edge/Wake-Vision-Train-Large",
                      help="Dataset name on Hugging Face")
    parser.add_argument("--split", type=str, default="train_large",
                      help="Dataset split to use")
    parser.add_argument("--images_per_shard", type=int, default=100,
                      help="Number of images per shard")
    parser.add_argument("--shard_id", type=int, default=0,
                      help="Shard ID to process")
    parser.add_argument("--false_positive_csv", type=str,
                      help="Path to CSV file containing false positives")
    parser.add_argument("--false_negative_csv", type=str,
                      help="Path to CSV file containing false negatives with areas")
    parser.add_argument("--min_box_area", type=float, default=5.0,
                      help="Minimum box area as percentage of image area")
    parser.add_argument("--dual_save", action="store_true",
                      help="Save both original and relabeled versions when using CSV files")
    parser.add_argument("--download_all", action="store_true",
                      help="Download entire dataset without sharding")
    
    args = parser.parse_args()
    
    result = load_and_save_shard(
        dataset_name=args.dataset,
        split=args.split,
        target_images_per_shard=args.images_per_shard,
        shard_id=args.shard_id,
        false_positive_csv=args.false_positive_csv,
        false_negative_csv=args.false_negative_csv,
        min_box_area_percentage=args.min_box_area,
        dual_save=args.dual_save,
        download_all=args.download_all
    )