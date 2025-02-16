import torch
from torch.utils.data import DataLoader, DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from dataset import AlbumentationsDataset, TransformWrapper

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

def train_augmentation_pipeline(img_size: int):
    return A.Compose([
        # 1. Always crop & resize to target dimensions
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),

        # 2. Basic horizontal flip
        A.HorizontalFlip(p=0.5),

        # 3. Light color transformations (choose 1 out of the 4)
        A.SomeOf(
            transforms=[
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.ToGray(p=1.0),
                A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=1.0)
            ],
            n=1,              # Pick exactly 1 transform to apply
            p=0.4             # 40% chance to apply any color augmentation
        ),

        # 4. Apply a blur or motion blur with moderate kernel sizes
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),

        # 5. Mild geometric transformations
        A.ShiftScaleRotate(
            shift_limit=0.05,     # small shift
            scale_limit=0.1,      # up to +/-10% scale
            rotate_limit=10,      # up to +/-10 degrees
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.2
        ),

        # 6. Occasional random weather condition
        A.OneOf([
            A.RandomRain(
                brightness_coefficient=0.9,
                drop_length=8,       # smaller drop length
                drop_width=1,
                blur_value=5,
                rain_type='drizzle',
                p=1.0
            ),
            A.RandomFog(fog_limit=(10, 20), alpha_coef=0.05, p=1.0),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=1.0
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0.5,
                p=1.0
            ),
        ], p=0.15),  # 15% chance of applying any weather effect

        # 7. Occasional Coarse Dropout
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=4,
            min_height=8,
            min_width=8,
            p=0.2
        ),

        # 8. Normalize & convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])



def get_augmentation_pipeline(train=True, img_size=224):
    if train:
        augmentation_list = train_augmentation_pipeline(img_size)
        # augmentation_list = A.Compose([
        #     A.RandomResizedCrop(size=(img_size,img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
        #     A.HorizontalFlip(p=0.6),
        #     A.OneOrOther(
        #         first=A.ChannelShuffle(p=0.5),
        #         second=A.SelectiveChannelTransform(
        #             transforms=[
        #                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        #                 A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1.0)
        #             gi],
        #             channels=[0, 1, 2],
        #             p=0.5
        #         ),
        #         p=0.3
        #     ),
        #     A.OneOf([
        #         A.GaussianBlur(blur_limit=(3, 25), p=1.0),
        #         A.MotionBlur(blur_limit=(101, 491), p=1.0),
        #         A.GlassBlur(sigma=0.7, max_delta=50, iterations=2, p=1.0),
        #     ], p=0.5),
        #     A.Sequential([
        #         A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=50, p=0.5),
        #         A.Affine(scale=(0.95, 1.05), translate_percent=(-0.02, 0.02), rotate=(-5, 5), shear=(-2, 2),p=0.5)
        #     ], p=0.3),
        #     A.OneOf([
        #         A.ElasticTransform(alpha=1, sigma=25, alpha_affine=25,interpolation=cv2.INTER_CUBIC,border_mode=cv2.BORDER_REFLECT_101,p=1.0),
        #         A.GridDistortion(num_steps=5, distort_limit=(-0.1, 0.1),interpolation=cv2.INTER_CUBIC,border_mode=cv2.BORDER_REFLECT_101,p=1.0),
        #         A.OpticalDistortion(distort_limit=(0.8, 0.9), shift_limit=(0.4, 0.6),border_mode=0,p=1.0),
        #     ], p=0.3),
        #     A.SomeOf(
        #         transforms=[
        #             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        #             A.InvertImg(p=1.0),
        #             A.ToGray(p=1.0),
        #             A.Posterize(num_bits=3, p=1.0),
        #             A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=1.0)],
        #         n=2,
        #         replace=False,
        #         p=0.5
        #     ),
        #     A.OneOf([
        #         A.RandomRain(brightness_coefficient=0.9, drop_length=10, drop_width=1,blur_value=7, rain_type='drizzle', p=1.0),
        #         A.RandomFog(fog_limit=(10, 30), alpha_coef=0.08, p=1.0),
        #         A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=1.0),
        #         A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1.0),
        #         A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1.0),
        #     ], p=0.3),
        #     A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
        #     A.CoarseDropout(max_holes=8, max_height=16, max_width=16,min_holes=4, min_height=8, min_width=8,p=0.2),
        #     A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #     ToTensorV2()
        # ])


    else:
        augmentation_list = [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    return A.Compose(augmentation_list, p=1.0, additional_targets={'mask': 'mask'})

def create_data_loaders(
    data_dir,
    input_shape, 
    batch_size,
    val_split=0.1,
    test_split=0.1,
    num_workers=32,
    seed=42,
    val_dir=None,
    test_dir=None,
    distributed=False,      # New parameter to indicate distributed mode
    local_rank=0           # Local rank (for reproducibility in the sampler)
):
    """
    Creates data loaders for training, validation, and testing using data from disk in ImageNet format.
    If val_dir and test_dir are provided, uses those directories for validation and test sets.
    Otherwise, splits data_dir into train/val/test sets using the specified splits.
    Uses Albumentations for data augmentation.

    Args:
        data_dir (str): Directory containing the training dataset in ImageNet format
        input_shape (tuple): The desired input shape for images (height, width, channels)
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data to use for validation when splitting (default: 0.1)
        test_split (float): Fraction of data to use for testing when splitting (default: 0.1)
        num_workers (int, optional): Number of workers for data loading. If None, uses CPU count
        seed (int): Random seed for reproducibility
        val_dir (str, optional): Directory containing validation dataset. If provided, val_split is ignored
        test_dir (str, optional): Directory containing test dataset. If provided, test_split is ignored
        distributed (bool): Whether to use distributed training mode
        local_rank (int): Local rank for distributed training

    Returns:
        tuple: (train_loader, val_loader, test_loader) - PyTorch DataLoaders for each split
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")
    if val_dir and not os.path.exists(val_dir):
        raise ValueError(f"Validation directory {val_dir} does not exist")
    if test_dir and not os.path.exists(test_dir):
        raise ValueError(f"Test directory {test_dir} does not exist")

    # Get augmentation pipelines
    train_transform = get_augmentation_pipeline(train=True, img_size=input_shape[0])
    val_transform = get_augmentation_pipeline(train=False, img_size=input_shape[0])

    # Case 1: Using separate directories for validation and test
    if val_dir and test_dir:
        train_dataset = AlbumentationsDataset(data_dir, transform=train_transform)
        val_dataset = AlbumentationsDataset(val_dir, transform=val_transform)
        test_dataset = AlbumentationsDataset(test_dir, transform=val_transform)
        
        print(f"\nUsing separate directories for validation and test sets:")
        print(f"Training: {len(train_dataset)} images from {data_dir}")
        print(f"Validation: {len(val_dataset)} images from {val_dir}")
        print(f"Test: {len(test_dataset)} images from {test_dir}")
    
    # Case 2: Split single directory into train/val/test
    else:
        # Load the full dataset
        full_dataset = AlbumentationsDataset(data_dir, transform=val_transform)
        
        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(val_split * total_size)
        test_size = int(test_split * total_size)
        train_size = total_size - val_size - test_size

        # Split the dataset
        torch.manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size, test_size]
        )

        # Apply transforms to each split
        train_dataset = TransformWrapper(train_dataset, train_transform)
        val_dataset = TransformWrapper(val_dataset, val_transform)
        test_dataset = TransformWrapper(test_dataset, val_transform)
        
        print(f"\nSplit single directory into train/val/test:")
        print(f"Total images: {total_size} from {data_dir}")
        print(f"Training: {train_size} images")
        print(f"Validation: {val_size} images")
        print(f"Test: {test_size} images")

    # Verify class consistency across splits
    if val_dir and test_dir:
        train_classes = set(train_dataset.classes)
        val_classes = set(val_dataset.classes)
        test_classes = set(test_dataset.classes)
    else:
        train_classes = set(full_dataset.classes)
        val_classes = train_classes
        test_classes = train_classes
    
    if not (train_classes == val_classes == test_classes):
        raise ValueError("Classes are not consistent across train, validation, and test sets")

    print(f"Classes: {sorted(train_classes)}")
    print(f"Process started with local_rank: {local_rank}")


    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=local_rank,
            shuffle=True,
            seed=seed
        )
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=shuffle_flag, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        sampler=train_sampler,
        drop_last=True,
        multiprocessing_context='fork'
    )
    
    # For validation and testing, you can either use a DistributedSampler as well (if you want distributed evaluation)
    # or simply set shuffle=False. Often, you may run evaluation only on rank 0.
    # Here we use a sampler for consistency:
    if distributed:
        val_sampler = DistributedSampler(val_dataset, num_replicas=torch.distributed.get_world_size(), rank=local_rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=torch.distributed.get_world_size(), rank=local_rank, shuffle=False)
    else:
        val_sampler = None
        test_sampler = None

    val_workers = max(2, num_workers // 4)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,
        shuffle=False, 
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        sampler=val_sampler,
        drop_last=False,
        multiprocessing_context='fork'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size * 2,
        shuffle=False, 
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        sampler=test_sampler,
        drop_last=False
    )
    # For the training dataset, use DistributedSampler if distributed training is enabled
    if local_rank == 0:
        print(f"Train dataset length: {len(train_dataset)}, Sampler length: {len(train_sampler)}")
    if local_rank == 1:
        print(f"Train dataset length: {len(train_dataset)}, Sampler length: {len(train_sampler)}")

    return train_loader, val_loader, test_loader