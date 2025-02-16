from ultralytics import YOLO
import json
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import os
import threading
from queue import Queue
import time
import shutil

class ImageBuffer:
    def __init__(self, buffer_size=15000):
        self.buffer_size = buffer_size
        self.buffer_1 = Queue()
        self.buffer_2 = Queue()
        self.current_buffer = self.buffer_1
        self.next_buffer = self.buffer_2
        self.buffer_ready = threading.Event()
        self.download_complete = threading.Event()
        self.stop_signal = threading.Event()
        self.processing_needed = threading.Event()

    def swap_buffers(self):
        self.current_buffer, self.next_buffer = self.next_buffer, self.current_buffer
        while not self.next_buffer.empty():
            self.next_buffer.get()
        self.buffer_ready.clear()
        self.processing_needed.clear()

def download_worker(dataset_iterator, image_buffer):
    """Worker function to download images from sharded dataset"""
    try:
        
        for item in dataset_iterator:
            if image_buffer.stop_signal.is_set():
                break


            image_buffer.next_buffer.put({
                'image': item['image'],
                'filename': item['filename'],
                'original_label': item['person']
            })
            image_buffer.processing_needed.set()

            if image_buffer.next_buffer.qsize() >= image_buffer.buffer_size:
                image_buffer.buffer_ready.set()

        image_buffer.download_complete.set()
    except Exception as e:
        print(f"Error in download worker: {e}")
        image_buffer.stop_signal.set()
        raise

def process_images_batch(model, image_data_batch):
    results_dict = {}
    try:
        batch_images = [item['image'].convert('RGB') for item in image_data_batch]
        batch_results = model(batch_images, verbose=False, device='cuda')

        for j, results in enumerate(batch_results):
            boxes = results.boxes.xyxy.tolist()
            confidences = results.boxes.conf.tolist()
            class_ids = results.boxes.cls.tolist()

            objects = []
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                objects.append({
                    'box': box,
                    'label': model.names[int(class_id)],
                    'confidence': float(conf)
                })

            filename = image_data_batch[j]['filename']
            original_label = image_data_batch[j]['original_label']
            results_dict[filename] = {
                'objects': objects,
                'original_label': original_label
            }
    except Exception as e:
        print(f"Error processing batch: {e}")
        raise
    return results_dict

def main():
    # Sharding configuration
    SHARD_INDEX = 0    # Set this 0-5 for different machines
    NUM_SHARDS = 6     # Total number of machines
    BUFFER_SIZE = 1600
    BATCH_SIZE = 1

    # Initialize image buffer
    image_buffer = ImageBuffer(BUFFER_SIZE)

    # Load sharded dataset
    dataset = load_dataset("Harvard-Edge/Wake-Vision-Train-Large", streaming=True)
    sharded_dataset = dataset['train_large'].shard(
        num_shards=NUM_SHARDS,
        index=SHARD_INDEX
    )
    dataset_iterator = iter(sharded_dataset)

    # Start download thread
    download_thread = threading.Thread(
        target=download_worker,
        args=(dataset_iterator, image_buffer)
    )
    download_thread.start()

    # Load model
    model = YOLO('yolo11x.pt')

    # Process images
    all_results = {}
    processed_count = 0
    shard_results_dir = f"results/shard_{SHARD_INDEX}"
    os.makedirs(shard_results_dir, exist_ok=True)

    try:
        while not (image_buffer.download_complete.is_set() and image_buffer.current_buffer.empty()):
            if image_buffer.processing_needed.wait(timeout=1):
                current_data = [image_buffer.current_buffer.get() for _ in range(image_buffer.current_buffer.qsize())]
                
                for i in tqdm(range(0, len(current_data), BATCH_SIZE), desc=f"Processing shard {SHARD_INDEX}"):
                    batch_data = current_data[i:i + BATCH_SIZE]
                    results = process_images_batch(model, batch_data)
                    all_results.update(results)
                    processed_count += len(batch_data)

                    if processed_count % 5000 == 0 and processed_count > 0:
                        filename = f"{shard_results_dir}/images_{processed_count-5000}-{processed_count}.json"
                        with open(filename, 'w') as f:
                            json.dump(all_results, f, indent=4)
                        all_results = {}

                if image_buffer.buffer_ready.is_set() or image_buffer.download_complete.is_set():
                    image_buffer.swap_buffers()
            else:
                if image_buffer.download_complete.is_set():
                    image_buffer.swap_buffers()

    except KeyboardInterrupt:
        print("\nStopping gracefully...")
        image_buffer.stop_signal.set()

    finally:
        download_thread.join()

        if all_results:
            filename = f"{shard_results_dir}/images_{processed_count-len(all_results)}-{processed_count}.json"
            with open(filename, 'w') as f:
                json.dump(all_results, f, indent=4)

        print(f"\nProcessing complete on shard {SHARD_INDEX}! Processed {processed_count} images")
        print(f"Results saved to: {shard_results_dir}")

if __name__ == '__main__':
    main()