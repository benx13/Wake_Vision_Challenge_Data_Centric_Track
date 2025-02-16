import torch
import torch.nn as nn
from tqdm import tqdm
import os
import wandb
from torch.amp import GradScaler, autocast # Import GradScaler and autocast
import time
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def save_checkpoint(state, is_best, output_dir, model_name):
    """Save model checkpoint and optionally log to wandb."""
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    
    # Save latest checkpoint
    latest_path = os.path.join(output_dir, f"{model_name}_latest.pth")
    torch.save(state, latest_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(output_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)
        
        # Log best model to wandb
        wandb.save(best_path)
        
    return latest_path if not is_best else best_path
    """
    Save model checkpoint.
    
    Args:
        state (dict): State dictionary containing model state and metadata
        is_best (bool): Whether this is the best model so far
        output_dir (str): Directory to save checkpoints
        model_name (str): Base name for the checkpoint files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save latest checkpoint
    latest_path = os.path.join(output_dir, f"{model_name}_latest.pth")
    torch.save(state, latest_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(output_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        scaler: Optional GradScaler to load state
    
    Returns:
        tuple: (start_epoch, best_val_accuracy)
    """
    if not os.path.exists(checkpoint_path):
        return 0, 0.0
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_accuracy', 0.0)

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epochs,
    output_dir,
    model_name,
    net_id,
    num_classes,
    rank=0,
    resume_training=False,
    load_from_checkpoint=False
):
    """
    Train a model with the given parameters and data loaders.
    
    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train
        output_dir: Directory to save checkpoints
        model_name: Name of the model for saving checkpoints
        net_id: ID of the network architecture
        num_classes: Number of output classes
        resume_training: Whether to resume from latest checkpoint with same training state
        load_from_checkpoint: Whether model was loaded from checkpoint (but starting fresh training)
        rank: Process rank in distributed training (default: 0)
    
    Returns:
        float: Best validation accuracy achieved
        float: Test accuracy of the best model
    """
    # Initialize best accuracy and start epoch
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    print(output_dir)
    latest_checkpoint = os.path.join(output_dir, f"{model_name}_latest.pth")
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    print(latest_checkpoint)
    if resume_training and os.path.exists(latest_checkpoint):
        start_epoch, best_val_accuracy = load_checkpoint(
            latest_checkpoint,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )
        if rank == 0:
            print(f"Resuming training from epoch {start_epoch} with best validation accuracy: {best_val_accuracy:.4f}")
    else:
        # When loading from checkpoint but starting fresh, or starting completely new
        start_epoch = 0
        best_val_accuracy = 0.0
        if load_from_checkpoint and rank == 0:
            print("Starting fresh training with loaded model weights")

    model = model.to(memory_format=torch.channels_last)
    
    # Enable gradient synchronization only when needed
    if isinstance(model, DistributedDataParallel):
        model.require_backward_grad_sync = False
    
    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        running_loss = torch.zeros(1, device=device)
        correct_predictions = torch.zeros(1, device=device)
        total_samples = 0
        
        train_iter = tqdm(train_loader, leave=False) if rank == 0 else train_loader
        for step, (images, labels) in enumerate(train_iter):
            # Enable gradient sync every step
            if isinstance(model, DistributedDataParallel):
                model.require_backward_grad_sync = True
            
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum()

            if rank == 0:
                train_iter.set_description(f"Epoch [{epoch+1}/{epochs}]")
                train_iter.set_postfix(
                    train_loss=running_loss.item() / (step + 1),
                    train_acc=correct_predictions.item() / total_samples
                )

        # Synchronize metrics across GPUs if in distributed mode
        if isinstance(model, DistributedDataParallel):
            dist.all_reduce(running_loss)
            dist.all_reduce(correct_predictions)
            total_samples = total_samples * dist.get_world_size()
        train_accuracy = correct_predictions.item() / total_samples
        train_loss = running_loss.item() / len(train_loader)
        
        # Log training metrics to wandb only on rank 0
        if rank == 0:
            wandb.log({
                'train/loss': train_loss,
                'train/accuracy': train_accuracy,
                'train/epoch': epoch + 1
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        val_iter = tqdm(val_loader, leave=False) if rank == 0 else val_loader
        
        with torch.no_grad():
            for images, labels in val_iter:
                # Move to GPU with channels_last and non_blocking
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                
                if rank == 0:
                    val_iter.set_description(f"Epoch [{epoch+1}/{epochs}]")
                    val_iter.set_postfix(
                        val_loss=val_loss / (val_iter.n + 1e-5),
                        val_acc=val_correct_predictions / val_total_samples
                    )

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss = val_loss / len(val_loader)

        # Log validation metrics to wandb only on rank 0
        if rank == 0:
            wandb.log({
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
                'val/epoch': epoch + 1,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}, "
                  f"LR: {current_lr:.2e}")

        # Save checkpoint only on rank 0
        is_best = val_accuracy > best_val_accuracy
        best_val_accuracy = max(val_accuracy, best_val_accuracy)
        
        if rank == 0:
            # Log best metrics to wandb
            if is_best:
                wandb.run.summary['best_val_accuracy'] = best_val_accuracy
            
            # Prepare checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            save_checkpoint(checkpoint, is_best, output_dir, model_name)

    if rank == 0:
        print(f"\nBest validation accuracy achieved: {best_val_accuracy:.4f}")
        best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")
        print(f"Best model saved to: {best_model_path}")

        # Testing phase - use the best model for testing
        from mcunet.model_zoo import build_model
        best_model = build_model(net_id=net_id, pretrained=False)[0]
        checkpoint = torch.load(best_model_path)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model.to(device)
        best_model.eval()

        test_correct = 0
        total = 0
        test_loop = tqdm(test_loader, leave=False)
        
        with torch.no_grad():
            for images, labels in test_loop:
                # Move to GPU with channels_last and non_blocking
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = best_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = test_correct / total
        
        # Log final test metrics to wandb
        wandb.run.summary.update({
            'test/accuracy': test_accuracy,
            'test/correct': test_correct,
            'test/total': total,
            'test/wrong': total - test_correct
        })
        
        print(f"\nTest accuracy of the best model: {test_accuracy:.4f}\n")

        # Finish the wandb run
        wandb.finish()

        return best_val_accuracy, test_accuracy
    
    return best_val_accuracy, 0.0  # Return 0.0 as test accuracy for non-rank-0 processes
