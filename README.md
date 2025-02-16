# Edge AI Foundation Wake Vision Challenge - Data Centric Track Submission

This is the official submission for the data centric track for the Edge AI Foundation Wake Vision Challenge.

## Overview

This repository contains our submission focusing on data-centric improvements to enhance wake word detection performance on microcontrollers. The main idea behind the submission is detecting problimatic images in train-large split of VWW2 dataset and relabeling them, combining them strong data augmentation and good training practices we were able to achieve 92.43% accuracy on the test set approximately 6.8% above the baseline.

The submission includes:

- Data processing and labeling pipeline
- Model training configuration
- TFLite model export for microcontroller deployment

## Envirenment Setup
we recomend using the official torch from docker hub to avoid any issues with the dependencies.

```bash
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

```bash
# run the container with the current directory mounted to /workspace
docker run -it --gpus all pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel -v $(pwd):/workspace
```

```bash
# install the dependencies
pip install ultralytics albumentations albumentations-pytorch wandb datasets
```


## Label Generation

For label generation we used YOLOv11x model pretrained on COCO dataset to generate the labels for the train-large split of VWW2 dataset. 
```bash
yolo export model=yolo11x.pt format=engine imgsz=640 workspace=15 half=True
```
This will export the model into trt engine format to maximize inference speed.

```bash
# run the script to generate the labels
python generate_labels.py
```
this script shards the dataset into 6 shards and runs the YOLOv11x model to generate the labels for each shard using a dual buffer to speed up the process without the need to download the data where we launch two processes one for downloading and the other for infence the labels. the inference process is done on the fly and the results are saved in the results folder. the script processes 1 shard in approximately 53 minutes using TensorRT 8.6 runtime on a RTX 3070 GPU. The speed is approximately 125 images per second matching the interned speed to maximize efficiency and gpu up time.


```bash
# run the script to generate the labels
python export_labels2csv.py
```
Finally we run the script to export the labels to a csv file. this will save a minimized version of the problimatic images as false positives and false negatives. 

False positives: are images that should have been labeled as no-person but were labeled as person. 


False negatives: are images that should have been labeled as person but were labeled as no-person.

to reduce strorage we memory usage we export into a very compact format where false_positives.csv contains only filenames and false_negatives.csv contains only filenames and area of the larget person bounding box in pixels (we save the area in pixels instead of percentage of the image because we forgot to save the image size at the start of the process).

This will export the directory of jsons from approximately 9GB into 20mb. 

## Training Data Downloader

After exporting the data we download the data from huggingface and save it in image net format.

```bash
# run the script to download the data
python download.py --dataset Harvard-Edge/Wake-Vision-Train-Large --split train_large --images_per_shard 5760428 --shard_id 0 --false_positive_csv false_positives.csv --false_negative_csv false_negatives.csv --dual_save
```
This will download the data and relabel it on the fly in steaming mode takes about 3 hours (@ 2Gbits internet speed) to finish. this is slow but uses only 200GB of storage.

> **Warning**: Using the `--download_all` flag will download the entire dataset without sharding and extracting. This is **not recommended** unless you have:
> - 5+ Gbps internet speed
> - 256+ threads available for extraction
> - 3TB of available storage for the ImageNet format export
>
> Even with these requirements, the process will still take approximately 45 minutes to complete.




## Repository initialization.

we use the original weights from of mcunet-vww2 from the hanlab repository. for that we fork the repository and the build the rest of the process on top of it.

to load the pretrained weights on vww dataset we use:

```python
from mcunet.model_zoo import net_id_list, build_model, download_tflite
print(net_id_list)  # the list of models in the model zoo      
model, image_size, description = build_model(net_id="mcunet-vww2", pretrained=True)  # you can replace net_id with any other option from net_id_list
```

## Data Augmetation 

We use the albumentations library to augment the data. We use a strong augmentation pipeline including various standard image ops as well as weather operations as well.

```python
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
```

This pipeline was crafted through trail and error on a small subset of the data to find the best balance between data augmentation and model performance. MixUP based ops were discarded as they causing training instability.


## Label smoothing loss

We use the label smoothing loss to improve the performance of the model.

The label smoothing loss is defined as:

L = (1 - α) * CE(p, q) + α * KL(u || q)

where:
- CE(p,q) is the standard cross-entropy loss
- KL(u || q) is the KL divergence between uniform distribution u and predicted distribution q
- α is the smoothing parameter (typically 0.1)
- p are the hard targets (0/1)
- q are the predicted probabilities

This smoothing helps prevent the model from becoming overconfident in its predictions and improves generalization.

Trail and error yielded the best results with α = 0.05 on a small subset of the data. proving the effectiveness of the relabeling process as going higher only decreased accuracy.


## Training Loop

to train the model we use the train script which is a very optimized for speed to handle the 5.7 million images in the train-large split.

```bash
# run the script to train the model

CUDA_VISIBLE_DEVICES=0,1  torchrun --nproc_per_node=2 train.py   --batch_size 256 --learning_rate 3e-4 --data_dir full_dataset_human_vs_nohuman_relabeled/ --test_dir shard_0_human_vs_nohuman --val_dir shard_0_human_vs_nohuman --num_workers 256 --net_id mcunet-vww2 --use_sce false  --label_smoothing 0.05 --wandb-project mcunet_vww2
```


This will train the model on 2 GPUs in parallel and use 256 workers for data loading. it takes around 18 mins per epoch on two RTX 4090s. 


To train the model on a single GPU please refere to the stable_training_singGPU branch from this repo: https://github.com/benx13/mount_trainer.git


## Results

We achieved 92.43% accuracy on the test set. after ~120 epochs of training. Fine tuning didn't yield any improvment. The training stagnated at around 89% accuracy. After we included images with person area < 5% we seen a jump in accuracy to 92%. After manually checking the test set we found many images we really small ppl in them. 

<a href="https://ibb.co/dw27jsJw"><img src="https://i.ibb.co/cSFxkKXS/Screenshot-2025-02-12-at-6-13-28-AM.png" alt="Screenshot-2025-02-12-at-6-13-28-AM" border="0"></a>






## export into tflite 

```bash
# run the script to export the model into tflite
python tflite_exporter.py
```

This will export the model into tflite format. please note this works only with the following versions of the modules in requirements_tflite.txt
