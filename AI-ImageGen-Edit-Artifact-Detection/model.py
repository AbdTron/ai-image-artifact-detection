"""
Model Architecture for AI Image Detection
- EfficientNet-B4 backbone
- Optional SRM layer for noise residual extraction
"""
import torch
import torch.nn as nn
import timm

import config


class SRMLayer(nn.Module):
    """
    Steganalysis Rich Model (SRM) layer.
    Extracts noise residuals to detect AI generation artifacts.
    """
    
    def __init__(self):
        super().__init__()
        self.filters = nn.Conv2d(3, 9, kernel_size=5, padding=2, bias=False)
        
        # Initialize with high-pass filters
        weights = torch.zeros(9, 3, 5, 5)
        
        # Horizontal edge
        weights[0, :, 2, 1] = -1
        weights[0, :, 2, 2] = 2
        weights[0, :, 2, 3] = -1
        
        # Vertical edge
        weights[1, :, 1, 2] = -1
        weights[1, :, 2, 2] = 2
        weights[1, :, 3, 2] = -1
        
        # Laplacian
        weights[2, :, 1, 2] = -1
        weights[2, :, 2, 1] = -1
        weights[2, :, 2, 2] = 4
        weights[2, :, 2, 3] = -1
        weights[2, :, 3, 2] = -1
        
        # Diagonal filters
        for i in range(3, 9):
            weights[i, :, 2, 2] = 4
            weights[i, :, 0, 0] = -1
            weights[i, :, 0, 4] = -1
            weights[i, :, 4, 0] = -1
            weights[i, :, 4, 4] = -1
        
        self.filters.weight = nn.Parameter(weights, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.filters(x)
        return torch.cat([x, noise], dim=1)  # 3 + 9 = 12 channels


class AIImageDetector(nn.Module):
    """EfficientNet-B4 based detector with optional SRM preprocessing."""
    
    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        use_srm: bool = config.USE_SRM
    ):
        super().__init__()
        self.use_srm = use_srm
        
        if use_srm:
            self.srm = SRMLayer()
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=12)
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
        print(f"Model: {model_name}, Features: {self.feature_dim}, SRM: {use_srm}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_srm:
            x = self.srm(x)
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for Grad-CAM."""
        if self.use_srm:
            x = self.srm(x)
        return self.backbone(x)


def create_model(**kwargs) -> AIImageDetector:
    """Factory function to create model."""
    return AIImageDetector(**kwargs)


if __name__ == "__main__":
    print("Testing model...")
    model = create_model()
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
