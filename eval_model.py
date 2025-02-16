from mcunet.model_zoo import net_id_list, build_model
import torch
import torch
from torch.utils.data import DataLoader, DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from dataset import AlbumentationsDataset, TransformWrapper
from tqdm import tqdm

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2




print(net_id_list)  # the list of models in the model zoo

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-vww0", pretrained=False)  # you can replace net_id with any other option from net_id_list
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# download tflite file to tflite_path
path = 'final_resultsx9999999999999/vww0/XXXcheckpoints_vww0_finetune2/mcunet-vww0_best.pth'

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



def get_augmentation_pipeline(image_size=224):
    # Use exactly the same validation transform as in data_loaders.py
    augmentation_list = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(augmentation_list, p=1.0, additional_targets={'mask': 'mask'})  # Keep additional_targets to match training

def evaluate_model(model, val_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    # Match training settings
    model = model.to(memory_format=torch.channels_last)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Setup validation data loader with same settings as training
val_transform = get_augmentation_pipeline(image_size=image_size)
val_dataset = AlbumentationsDataset('/home/server/modelcentric/mount_trainer/final_resultsx9999999999999/shard_0_human_vs_nohuman', 
                                  transform=val_transform)
val_loader = DataLoader(
    val_dataset, 
    batch_size=512,  # Match training validation batch size (batch_size * 2)
    shuffle=False,
    num_workers=8,  # Match training validation workers (max(2, num_workers // 4))
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False
)

# Move model to GPU and set to eval mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Print validation set size for verification
print(f"Validation set size: {len(val_dataset)}")

# Evaluate the model
accuracy = evaluate_model(model, val_loader, device)
print(f'Accuracy on the test set: {accuracy:.2f}%')



