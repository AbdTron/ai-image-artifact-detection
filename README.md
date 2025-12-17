# 🔍 AI Image Artifact Detection

A deep learning project that detects whether images are AI-generated/AI-edited or real/authentic using PyTorch and EfficientNet-B4.

## 📋 Overview

This project implements a binary image classifier capable of distinguishing between:
- **AI-Generated/AI-Edited** images
- **Real/Authentic** images

## ✨ Features

- 🎯 **High Accuracy Detection** - Optimized for image classification
- 🖼️ **User-Friendly GUI** - Tkinter-based interface for easy image testing
- 📊 **Comprehensive Evaluation** - Detailed metrics including precision, recall, F1-score, ROC curves, and confusion matrices
- ⚡ **GPU Optimized** - Configured for NVIDIA RTX 3090 (24GB VRAM) with mixed precision training
- 🎨 **Optional SRM Layer** - Noise residual extraction for improved artifact detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 2.0+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AbdTron/ai-image-artifact-detection.git
cd ai-image-artifact-detection
```

2. Install dependencies:
```bash
pip install torch torchvision timm pillow numpy matplotlib scikit-learn
```

3. Prepare your dataset:
```
dataSet/
├── train/
│   ├── 0/  # AI-Generated/AI-Edited images
│   └── 1/  # Real/Authentic images
└── test/
    ├── 0/
    └── 1/
```

## 🎓 Training

Train the model with:

```bash
python train.py
```

### Training Configuration

Key parameters in `config.py`:
- **Model**: EfficientNet-B4
- **Image Size**: 380x380
- **Batch Size**: 64
- **Epochs**: 20
- **Learning Rate**: 1e-4
- **Mixed Precision**: Enabled (FP16)

## 📊 Evaluation

Evaluate model performance:

```bash
python evaluate.py
```

This generates:
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Detailed metrics report

## 🖥️ GUI Application

Launch the graphical interface for single image testing:

```bash
python gui.py
```

Features:
- Load and test individual images
- Real-time prediction with confidence scores
- Visual feedback with color-coded results

## 🧠 Model Architecture

```
Input Image (380x380x3)
    ↓
[Optional SRM Layer] → Noise Residual Extraction
    ↓
EfficientNet-B4 Backbone
    ↓
Feature Extraction (1792 features)
    ↓
Classifier Head:
  - Dropout (0.3)
  - Linear (1792 → 512)
  - ReLU
  - Dropout (0.2)
  - Linear (512 → 2)
    ↓
Output: [AI-Generated/Edited, Real/Authentic]
```

## 📁 Project Structure

```
ai-image-artifact-detection/
├── config.py           # Configuration settings
├── model.py            # Model architecture (EfficientNet + SRM)
├── dataset.py          # Data loading and augmentation
├── train.py            # Training script
├── evaluate.py         # Evaluation and metrics
├── inference.py        # Single image inference
├── gui.py              # Tkinter GUI application
├── dataSet/
│   ├── train/          # Training data
│   └── test/           # Testing data
└── Model-Graphs/       # Saved models and evaluation plots
```

## 🔧 Configuration Options

### Enable SRM Preprocessing
In `config.py`, set:
```python
USE_SRM = True  # Enables noise residual extraction
```

### Adjust Batch Size
For different GPU memory:
```python
BATCH_SIZE = 32  # Reduce if out of memory
BATCH_SIZE = 128  # Increase for larger GPUs
```

## 📈 Performance Metrics

The model is evaluated using:
- **Accuracy** - Overall classification accuracy
- **Precision** - True positive rate
- **Recall** - Sensitivity
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve
- **Confusion Matrix** - Visual representation of predictions

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **timm** - Pre-trained model library
- **PIL/Pillow** - Image processing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **scikit-learn** - Metrics and evaluation
- **Tkinter** - GUI framework

## 💡 Use Cases

- Academic research on AI-generated content detection
- Content verification for journalism and media
- Social media integrity and misinformation prevention
- Digital forensics and authentication

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- SRM layer implementation for deepfake detection
- PyTorch and timm libraries

## 👤 Author

**AbdTron**
- GitHub: [@AbdTron](https://github.com/AbdTron)
- Repository: [ai-image-artifact-detection](https://github.com/AbdTron/ai-image-artifact-detection)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/AbdTron/ai-image-artifact-detection/issues).

---

⭐ If you find this project useful, please consider giving it a star!
