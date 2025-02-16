import numpy as np
import tensorflow as tf
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import AlbumentationsDataset
import os
import gc
import time
import psutil  # Add this for memory monitoring

def get_augmentation_pipeline(image_size=224):
    # Use exactly the same validation transform as in data_loaders.py
    augmentation_list = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(augmentation_list, p=1.0, additional_targets={'mask': 'mask'})

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def evaluate_tflite_model(interpreter, val_loader, image_size):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    is_quantized = input_details[0]['dtype'] == np.int8
    
    correct = 0
    total = 0
    start_time = time.time()
    
    # Print expected input shape
    print(f"Expected input shape: {input_details[0]['shape']}")
    
    try:
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc='Evaluating TFLite model')):
            try:
                # Process single image
                image = images.numpy()[0]  # Get first (and only) image from batch
                # Don't transpose - keep NCHW format
                
                # Prepare input
                if is_quantized:
                    image_quantized = np.round(image / input_scale + input_zero_point)
                    input_tensor = image_quantized.astype(np.int8)
                else:
                    input_tensor = image
                
                # Reshape to match expected input shape (1, 3, 144, 144)
                input_tensor = input_tensor.reshape(1, 3, image_size, image_size)
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                output = interpreter.get_tensor(output_details[0]['index'])[0]
                if is_quantized:
                    output = (output.astype(np.float32) - output_zero_point) * output_scale
                
                # Get prediction
                predicted = np.argmax(output)
                label = labels.item()
                
                correct += (predicted == label)
                total += 1
                
                # Print progress periodically
                if total % 100 == 0:
                    elapsed_time = time.time() - start_time
                    current_accuracy = 100 * correct / total
                    speed = total / elapsed_time
                    print(f"\nProcessed {total}/{len(val_loader)} images. "
                          f"Current accuracy: {current_accuracy:.2f}%. "
                          f"Speed: {speed:.2f} images/second")
                    print_memory_usage()
                
                # Force garbage collection periodically
                if total % 1000 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"\nError processing image {batch_idx}: {str(e)}")
                continue
            
    except Exception as e:
        print(f"\nError in evaluation loop: {str(e)}")
        if total > 0:
            print(f"Partial results available for {total} samples")
    
    total_time = time.time() - start_time
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\nTotal evaluation time: {total_time:.2f} seconds")
    print(f"Average speed: {total/total_time:.2f} images/second")
    
    return accuracy

def main():
    image_size = 64
    batch_size = 1
    
    print("Initial memory usage:")
    print_memory_usage()
    
    tflite_path = "tflite_export/mcunet_tiny_int8.tflite"
    
    # Create interpreter with multiple threads but smaller thread count
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path,
        num_threads=4  # Reduced thread count
    )
    print("Running inference on CPU with 4 threads")
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model Input Details:", input_details)
    print("Model Output Details:", output_details)
    
    val_transform = get_augmentation_pipeline(image_size=image_size)
    val_dataset = AlbumentationsDataset(
        '/home/server/modelcentric/mount_trainer/final_resultsx9999999999999/shard_0_human_vs_nohuman',
        transform=val_transform
    )
    
    # Use minimal number of workers and disable pin_memory
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No additional workers
        pin_memory=False
    )
    
    print(f"Validation set size: {len(val_dataset)}")
    print("Memory usage before evaluation:")
    print_memory_usage()
    
    try:
        accuracy = evaluate_tflite_model(interpreter, val_loader, image_size)
        print(f'Final accuracy on the test set: {accuracy:.2f}%')
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    finally:
        # Clean up
        del interpreter
        gc.collect()
        print("Final memory usage:")
        print_memory_usage()

if __name__ == "__main__":
    main() 