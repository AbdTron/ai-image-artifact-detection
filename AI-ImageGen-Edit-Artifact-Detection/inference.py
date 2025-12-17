"""
Inference Script for AI Image Detection
Features:
- Single image or batch prediction
- Confidence scores
- Grad-CAM visualization
"""
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast

import config
from model import create_model


class GradCAM:
    """Grad-CAM visualization for model interpretability."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = torch.nn.functional.interpolate(
            cam, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode='bilinear', align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()


def load_image(image_path):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    return image, image_tensor


def visualize_gradcam(image, heatmap, prediction, confidence, save_path=None):
    """Visualize Grad-CAM overlay."""
    # Resize image to match heatmap
    image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[2].set_title(f"Prediction: {prediction} ({confidence:.1f}%)")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


@torch.no_grad()
def predict(model, image_tensor, device):
    """Make prediction."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with autocast(enabled=config.USE_AMP):
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
    
    pred_class = output.argmax(dim=1).item()
    confidence = probs[0, pred_class].item() * 100
    
    return config.CLASS_NAMES[pred_class], confidence, probs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = create_model().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load image
    print(f"Loading image: {args.image}")
    image, image_tensor = load_image(args.image)
    
    # Predict
    prediction, confidence, probs = predict(model, image_tensor, device)
    
    print("\n" + "="*40)
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities: fake={probs[0,0].item()*100:.1f}%, real={probs[0,1].item()*100:.1f}%")
    print("="*40)
    
    # Grad-CAM
    if args.gradcam:
        print("\nGenerating Grad-CAM visualization...")
        
        # Get last conv layer from EfficientNet backbone
        # For EfficientNet, it's the last block before pooling
        target_layer = model.backbone.conv_head
        
        gradcam = GradCAM(model, target_layer)
        
        # Need gradients for Grad-CAM
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad = True
        
        heatmap = gradcam.generate(image_tensor)
        
        # Save path
        save_path = Path(args.image).stem + "_gradcam.png"
        if args.output:
            save_path = args.output
        
        visualize_gradcam(image, heatmap, prediction, confidence, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualization")
    parser.add_argument("--output", type=str, help="Output path for Grad-CAM image")
    args = parser.parse_args()
    main(args)
