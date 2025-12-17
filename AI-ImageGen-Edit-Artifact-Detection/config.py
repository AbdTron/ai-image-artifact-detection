"""
Configuration for AI Image Detection Training
Optimized for RTX 3090 (24GB VRAM)
"""
import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataSet"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CHECKPOINT_DIR = BASE_DIR / "Model-Graphs"
LOG_DIR = BASE_DIR / "runs"

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL
# ============================================================================
MODEL_NAME = "efficientnet_b4"
NUM_CLASSES = 2  # real vs fake
IMAGE_SIZE = 380  # EfficientNet-B4 optimal size
PRETRAINED = True
USE_SRM = False  # Set True for noise residual preprocessing

# ============================================================================
# TRAINING (RTX 3090 Optimized)
# ============================================================================
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True
PREFETCH_FACTOR = 2

EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Mixed Precision (FP16) - 2x faster, 50% less VRAM
USE_AMP = True

# Early Stopping
PATIENCE = 5

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
USE_FFT_PREPROCESSING = False  # Set True for frequency domain features

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# CUDA
# ============================================================================
CUDNN_BENCHMARK = True

# ============================================================================
# LOGGING
# ============================================================================
LOG_INTERVAL = 50
SAVE_BEST_ONLY = True

# Class names for display (model terminology)
CLASS_NAMES = ["AI-Generated/AI-Edited", "Real/Authentic"]
