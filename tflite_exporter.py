import torch
from mcunet.model_zoo import net_id_list, build_model
import os
import tensorflow as tf
import numpy as np
import onnx
from onnx_tf.backend import prepare
from torch.utils.data import DataLoader
from dataset import AlbumentationsDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

print(net_id_list)  # the list of models in the model zoo

def get_augmentation_pipeline(image_size=224):
    augmentation_list = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(augmentation_list, p=1.0, additional_targets={'mask': 'mask'})

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-vww2", pretrained=False)  # you can replace net_id with any other option from net_id_list

print(image_size, description)

path = 'checkpoint/mcunet-vww2_best.pth'
# Load weights from checkpoint if specified
checkpoint = torch.load(path)
print("Available keys in checkpoint:", checkpoint.keys())
# print(checkpoint['val_accuracy'])
print(f'best_val_accuracy: {checkpoint["best_val_accuracy"]}')
# Remove the 'module._orig_mod.' prefix from state dict keys
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for key in state_dict:
    new_key = key.replace('module._orig_mod.', '')
    new_state_dict[new_key] = state_dict[key]

# Load the modified state dict
model.load_state_dict(new_state_dict)
model.eval()

# Create export directory if it doesn't exist
export_dir = "checkpoint/tflite_export_vww2"
os.makedirs(export_dir, exist_ok=True)

# Prepare validation dataset for calibration
val_transform = get_augmentation_pipeline(image_size=image_size)
val_dataset = AlbumentationsDataset(
    'SAMPLE_DATA_PATH',
    transform=val_transform
)
calibration_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Export to ONNX with more careful configuration
dummy_input = torch.randn(1, 3, image_size, image_size)
onnx_path = os.path.join(export_dir, "mcunet_tiny.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=True
)

print(f"Model exported to {onnx_path}")

# Convert ONNX to TensorFlow SavedModel format
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_model_path = os.path.join(export_dir, "mcunet_tiny_tf")
tf_rep.export_graph(tf_model_path)

# Convert to TFLite with improved INT8 quantization
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

# Enhanced quantization settings
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Use real data for calibration
def representative_dataset():
    num_calibration_samples = 1000  # Increased number of calibration samples
    calibration_samples = 0
    
    for images, _ in calibration_loader:
        if calibration_samples >= num_calibration_samples:
            break
            
        # Ensure correct format and normalization
        sample = images.numpy()
        yield [sample.astype(np.float32)]
        calibration_samples += 1
        
        if calibration_samples % 100 == 0:
            print(f"Calibrated with {calibration_samples} samples")

converter.representative_dataset = representative_dataset

# Additional quantization settings
converter.target_spec.supported_types = [tf.int8]
converter._experimental_disable_per_channel = False  # Enable per-channel quantization
converter.allow_custom_ops = True

print("Starting conversion with quantization...")
tflite_model = converter.convert()

# Save quantized model
tflite_path = os.path.join(export_dir, "mcunet_tiny_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"INT8 quantized model exported to {tflite_path}")

# Optional: Export FP32 model for comparison
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model_fp32 = converter.convert()
tflite_path_fp32 = os.path.join(export_dir, "mcunet_tiny_fp32.tflite")
with open(tflite_path_fp32, "wb") as f:
    f.write(tflite_model_fp32)

print(f"FP32 model exported to {tflite_path_fp32}")