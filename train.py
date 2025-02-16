import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import torch.distributed as dist
from data_loaders import create_data_loaders
from mcunet.model_zoo import build_model
from trainer import train_model
from lebel_smooth import LabelSmoothingLoss
from losses import SCELoss
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import sys 

def main(args):
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        print("Not using distributed training")
        print("Not using distributed training")
        print("Not using distributed training")
        print("Not using distributed training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Only the main process logs to wandb
    if (not hasattr(args, 'local_rank')) or (args.local_rank == 0):
        wandb.init(
            project=args.wandb_project,
            config={
                "net_id": args.net_id,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "val_split": args.val_split,
                "test_split": args.test_split,
                "seed": args.seed,
                "label_smoothing": args.label_smoothing,
                "loss_type": "SCE" if args.use_sce_loss else "LabelSmoothing",
                "sce_alpha": args.sce_alpha if args.use_sce_loss else None,
                "sce_beta": args.sce_beta if args.use_sce_loss else None,
            }
        )
        wandb.config.update({"device": str(device)})
    else:
        # In non-main processes you can choose to disable wandb logging or use a dummy run.
        wandb.init(mode="disabled")
    
    print(f"Using device: {device}")
    
    # Enable cuDNN benchmarking and deterministic mode
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set higher priority for GPU operations
    if torch.cuda.is_available():
        # Create separate streams for data transfer and computation
        data_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        torch.cuda.set_device(args.local_rank)
        
        # Set default stream
        torch.cuda.set_stream(compute_stream)

    # Adjust batch size more conservatively
    if args.local_rank is not None:
        world_size = dist.get_world_size()
        args.batch_size = args.batch_size * (world_size // 2)  # More conservative scaling
        args.num_workers = min(args.num_workers * (world_size // 2), os.cpu_count())

    # Build the model
    model, image_size, description = build_model(
        net_id=args.net_id,
        pretrained=True,
    )
    print(f"Loaded model: {args.net_id}")
    print(f"Image size: {image_size}")
    print(f"Description: {description}")
    


    # Move model to device and set channels_last format
    model = model.to(device, memory_format=torch.channels_last)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        model = torch.compile(model)  # PyTorch 2.0+ optimization
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Wrap model with DistributedDataParallel if in distributed mode
    if args.local_rank is not None:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)


    # Load weights from checkpoint if specified
    if args.load_from:
        if os.path.exists(args.load_from):
            print(f"Loading model weights from checkpoint: {args.load_from}")
            checkpoint = torch.load(args.load_from)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: Checkpoint file not found: {args.load_from}")

    print(args.resume_from)
    print(args.resume_from)
    print(args.resume_from)
    print(args.resume_from)
    print(args.resume_from)
    print(args.resume_from)
    print(args.resume_from)
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    # Create data loaders with an extra flag for distributed training
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        input_shape=(image_size, image_size, 3),  # Model's expected input size
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        distributed=(args.local_rank is not None),   # Pass distributed flag
        local_rank=args.local_rank                     # Pass local_rank for sampler seeding if needed
    )

    # Define loss function, optimizer, scheduler, and GradScaler
    if args.use_sce_loss:
        criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, 
                          num_classes=2, smoothing=args.label_smoothing).to(device)
        print("Using SCE Loss with alpha={}, beta={}, smoothing={}".format(
            args.sce_alpha, args.sce_beta, args.label_smoothing))
    else:
        criterion = LabelSmoothingLoss(smoothing=args.label_smoothing).to(device)
        print("Using Label Smoothing CE Loss with smoothing={}".format(args.label_smoothing))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    scaler = GradScaler('cuda')

    # Train the model
    best_val_accuracy, test_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=args.epochs,
        output_dir=args.resume_from if args.resume_from is not None else args.checkpoint_dir,
        model_name=args.net_id,
        net_id=args.net_id,
        num_classes=2,
        rank=args.local_rank,
        resume_training=(args.resume_from is not None),
        load_from_checkpoint=(args.load_from is not None)
    )

    # Only rank 0 prints the final results
    if (not hasattr(args, 'local_rank')) or (args.local_rank == 0):
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")

    # Cleanup distributed process group
    if args.local_rank is not None:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCUNet model on Wake Vision dataset")
    print("sys.argv:", sys.argv)
    import os
    print("LOCAL_RANK from env:", os.environ.get("LOCAL_RANK"))

    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default="mcunet-training",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity (username or team name)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Weights & Biases run name")
    
    # Model arguments
    parser.add_argument("--net_id", type=str, default="mcunet-vww2",
                      help="Model ID from MCUNet model zoo")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing the training dataset in ImageNet format (human/no-human subdirs)")
    parser.add_argument("--val_dir", type=str,
                      help="Optional directory containing validation dataset. If provided, val_split is ignored")
    parser.add_argument("--test_dir", type=str,
                      help="Optional directory containing test dataset. If provided, test_split is ignored")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data to use for validation when val_dir is not provided")
    parser.add_argument("--test_split", type=float, default=0.1,
                      help="Fraction of data to use for testing when test_dir is not provided")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=16,
                      help="Number of workers for training")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                      help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str,
                      help="Path to checkpoint to resume training from")
    parser.add_argument("--load_from", type=str,
                      help="Path to checkpoint to load model weights from (starts fresh training)")
    
    # Distributed training argument
    parser.add_argument("--local_rank", type=int, default=None,
                      help="Local rank for distributed training")

    # Add label smoothing argument
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                      help="Label smoothing factor (default: 0.05)")

    # Add SCE loss arguments
    parser.add_argument("--use_sce_loss", action='store_true',
                      help="Use Symmetric Cross Entropy Loss instead of standard CE")
    parser.add_argument("--sce_alpha", type=float, default=1.0,
                      help="Alpha parameter for SCE loss (default: 1.0)")
    parser.add_argument("--sce_beta", type=float, default=1.0,
                      help="Beta parameter for SCE loss (default: 1.0)")

    args = parser.parse_args()
    
    if args.local_rank is None:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            args.local_rank = int(local_rank_env)
        else:
            args.local_rank = 0  # default to 0 if nothing is found
    print(f"Process started with local_rank: {args.local_rank}")

    # Validate directory arguments
    if (args.val_dir and not args.test_dir) or (args.test_dir and not args.val_dir):
        parser.error("If providing separate validation/test directories, both --val_dir and --test_dir must be specified")
    
    if args.val_dir and args.test_dir:
        print("\nUsing separate directories for validation and test sets")
        for dir_path in [args.val_dir, args.test_dir]:
            if not os.path.exists(dir_path):
                parser.error(f"Directory does not exist: {dir_path}")
    else:
        print("\nUsing splits from training directory for validation and test sets")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    main(args)
