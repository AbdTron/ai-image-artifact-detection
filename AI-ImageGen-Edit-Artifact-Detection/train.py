"""
Training Script for AI Image Detection
Features:
- Mixed Precision Training (AMP)
- TensorBoard logging
- Checkpointing
- Early stopping
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import create_model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast(enabled=config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Stats
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{running_loss/(batch_idx+1):.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
        
        # Log to TensorBoard
        if batch_idx % config.LOG_INTERVAL == 0:
            step = epoch * len(loader) + batch_idx
            writer.add_scalar("Train/Loss", loss.item(), step)
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        
        with autocast(enabled=config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, _ = get_dataloaders()
    
    # Model
    print("\nCreating model...")
    model = create_model().to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
        if ckpt_path.exists():
            print(f"\nResuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_acc = checkpoint["val_acc"]
            print(f"Resumed from epoch {start_epoch} with val_acc={best_acc:.2f}%")
        else:
            print("No checkpoint found, starting fresh.")
    
    # TensorBoard
    writer = SummaryWriter(config.LOG_DIR)
    
    # Clear GPU cache before training
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Training loop
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs (from epoch {start_epoch})...")
    print(f"Batch size: {config.BATCH_SIZE}, AMP: {config.USE_AMP}")
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,  # Save next epoch to resume from
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, config.CHECKPOINT_DIR / "best_model.pth")
            print(f"  Saved best model (acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # GPU check mode
        if args.check_gpu:
            print("\nGPU Check Mode - stopping after 1 epoch")
            break
    
    writer.close()
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {config.CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--check-gpu", action="store_true", help="Run 1 epoch to check GPU utilization")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()
    main(args)
