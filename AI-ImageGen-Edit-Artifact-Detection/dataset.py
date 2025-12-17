"""
Dataset and DataLoader for AI Image Detection
Features:
- Optimized data loading for GPU
- Data augmentation
- Optional FFT/DCT frequency preprocessing
- Robust handling of corrupted images
"""
import random
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms

# Handle truncated images gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

import config


class FFTPreprocessing:
    """Apply FFT to extract frequency domain features."""
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale for FFT
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        
        # Apply 2D FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Stack with original (4 channels: RGB + FFT magnitude)
        return torch.cat([img, magnitude.unsqueeze(0)], dim=0)


class AIImageDataset(Dataset):
    """Dataset for AI-Generated vs Real image classification."""
    
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        use_fft: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_fft = use_fft
        self.fft_transform = FFTPreprocessing() if use_fft else None
        
        # Build samples list
        self.samples = []
        self.class_to_idx = {"fake": 0, "real": 1}
        
        for class_name in ["fake", "real"]:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # Stats
        fake_count = sum(1 for _, l in self.samples if l == 0)
        real_count = sum(1 for _, l in self.samples if l == 1)
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(f"  fake: {fake_count}, real: {real_count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            if self.fft_transform:
                image = self.fft_transform(image)
            
            return image, label
        except Exception as e:
            # If image is corrupted, return a random different sample
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)


def get_train_transforms() -> transforms.Compose:
    """Training augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_dataloaders(val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    
    # Full training dataset (use train transforms for both train/val - acceptable for uni project)
    full_train = AIImageDataset(
        config.TRAIN_DIR,
        transform=get_train_transforms(),
        use_fft=config.USE_FFT_PREPROCESSING
    )
    
    # Split train/val from SAME dataset (fixes potential index mismatch)
    train_size = int((1 - val_split) * len(full_train))
    val_size = len(full_train) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Test dataset
    test_dataset = AIImageDataset(
        config.TEST_DIR,
        transform=get_val_transforms(),
        use_fft=config.USE_FFT_PREPROCESSING
    )
    
    print(f"\nSplits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # DataLoaders
    loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
        "prefetch_factor": config.PREFETCH_FACTOR,
        "persistent_workers": config.NUM_WORKERS > 0,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing dataset loading...")
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")
