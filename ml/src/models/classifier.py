"""
Model Architectures for Blood Group Classification

This module provides CNN architectures using transfer learning for
fingerprint-based blood group classification.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from loguru import logger


class FingerprintClassifier(nn.Module):
    """
    CNN Classifier using transfer learning from pretrained models.
    
    Supports multiple backbones:
    - EfficientNet (B0, B1, B2, etc.)
    - ResNet (18, 34, 50, etc.)
    - MobileNetV3
    - ConvNeXt
    """
    
    SUPPORTED_MODELS = [
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "resnet18", "resnet34", "resnet50",
        "mobilenetv3_large_100", "mobilenetv3_small_100",
        "convnext_tiny", "convnext_small"
    ]
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the backbone model
            num_classes: Number of output classes (blood groups)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate before final classification
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        logger.info(f"Building {model_name} classifier with {num_classes} classes")
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=""  # Remove global pooling
        )
        
        # Get feature dimensions
        self.feature_dim = self._get_feature_dim()
        logger.info(f"Backbone feature dimension: {self.feature_dim}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
            
        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
    def _get_feature_dim(self) -> int:
        """Determine the feature dimension from backbone."""
        # Use a dummy input to get output shape
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        return features.shape[1]
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
        
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
        
    def _init_classifier(self) -> None:
        """Initialize classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (for visualization/analysis).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings
        """
        features = self.backbone(x)
        pooled = self.global_pool(features)
        return pooled.flatten(1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers for improved accuracy.
    """
    
    def __init__(
        self,
        model_configs: list,
        num_classes: int = 8,
        weights: Optional[list] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            model_configs: List of model configuration dicts
            num_classes: Number of output classes
            weights: Optional weights for each model
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.models = nn.ModuleList()
        
        for config in model_configs:
            model = FingerprintClassifier(
                model_name=config.get("name", "efficientnet_b0"),
                num_classes=num_classes,
                pretrained=config.get("pretrained", True),
                dropout_rate=config.get("dropout_rate", 0.3)
            )
            self.models.append(model)
        
        # Normalize weights
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        logger.info(f"Created ensemble with {len(self.models)} models")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of model outputs."""
        outputs = []
        
        for model, weight in zip(self.models, self.weights):
            logits = model(x)
            outputs.append(F.softmax(logits, dim=1) * weight)
            
        # Weighted average of probabilities
        ensemble_probs = torch.stack(outputs).sum(dim=0)
        
        # Convert back to logits for loss computation
        return torch.log(ensemble_probs + 1e-10)


def create_model(config: Dict) -> FingerprintClassifier:
    """
    Factory function to create a model from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured model
    """
    model_config = config.get("model", {})
    
    return FingerprintClassifier(
        model_name=model_config.get("name", "efficientnet_b0"),
        num_classes=model_config.get("num_classes", 8),
        pretrained=model_config.get("pretrained", True),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        freeze_backbone=model_config.get("freeze_backbone", False)
    )


def load_model(checkpoint_path: str, config: Dict, device: str = "cpu") -> FingerprintClassifier:
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
