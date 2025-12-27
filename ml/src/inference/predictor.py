"""
Production Inference Module

Provides optimized inference capabilities:
- Single image prediction
- Batch prediction
- ONNX runtime inference
- Preprocessing utilities
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loguru import logger

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")


CLASS_NAMES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


class Preprocessor:
    """
    Image preprocessor for inference.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
    def __call__(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: Image path, numpy array, or PIL Image
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Ensure RGB format
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
            
        # Apply transforms
        transformed = self.transform(image=image)
        return transformed["image"]
    
    def batch_preprocess(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            Batch tensor of shape (N, C, H, W)
        """
        tensors = [self(img) for img in images]
        return torch.stack(tensors)


class PyTorchPredictor:
    """
    PyTorch-based predictor for fingerprint blood group classification.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        image_size: int = 224
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use (auto, cuda, cpu, mps)
            image_size: Input image size
        """
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(image_size=image_size)
        
        self.class_names = CLASS_NAMES
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        from ..models import FingerprintClassifier
        
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extract config
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        
        # Create model
        model = FingerprintClassifier(
            model_name=model_config.get("name", "efficientnet_b0"),
            num_classes=model_config.get("num_classes", 8),
            pretrained=False,
            dropout_rate=model_config.get("dropout_rate", 0.3)
        )
        
        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        logger.info(f"Model loaded from {model_path}")
        return model
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict:
        """
        Predict blood group from a single image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        pred_prob = probs[pred_idx].item()
        
        return {
            "predicted_class": self.class_names[pred_idx],
            "predicted_index": pred_idx,
            "confidence": pred_prob,
            "probabilities": {
                name: probs[i].item()
                for i, name in enumerate(self.class_names)
            }
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Dict]:
        """
        Predict blood groups for a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess batch
        batch = self.preprocessor.batch_preprocess(images).to(self.device)
        
        # Forward pass
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        
        # Process results
        results = []
        for i in range(len(images)):
            pred_idx = torch.argmax(probs[i]).item()
            pred_prob = probs[i][pred_idx].item()
            
            results.append({
                "predicted_class": self.class_names[pred_idx],
                "predicted_index": pred_idx,
                "confidence": pred_prob,
                "probabilities": {
                    name: probs[i][j].item()
                    for j, name in enumerate(self.class_names)
                }
            })
            
        return results


class ONNXPredictor:
    """
    ONNX Runtime-based predictor for optimized inference.
    """
    
    def __init__(
        self,
        model_path: str,
        image_size: int = 224,
        use_gpu: bool = True
    ):
        """
        Initialize ONNX predictor.
        
        Args:
            model_path: Path to ONNX model
            image_size: Input image size
            use_gpu: Whether to use GPU acceleration
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
            
        # Set up execution providers
        providers = []
        if use_gpu:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in ort.get_available_providers():
                providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        logger.info(f"Using ONNX providers: {providers}")
        
        # Create session
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(image_size=image_size)
        
        self.class_names = CLASS_NAMES
        
        logger.info(f"ONNX model loaded from {model_path}")
        
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict:
        """
        Predict blood group from a single image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        tensor = self.preprocessor(image).unsqueeze(0).numpy()
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_prob = probs[pred_idx]
        
        return {
            "predicted_class": self.class_names[pred_idx],
            "predicted_index": int(pred_idx),
            "confidence": float(pred_prob),
            "probabilities": {
                name: float(probs[i])
                for i, name in enumerate(self.class_names)
            }
        }
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Dict]:
        """
        Predict blood groups for a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess batch
        batch = self.preprocessor.batch_preprocess(images).numpy()
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        logits = outputs[0]
        
        # Softmax for each sample
        results = []
        for i in range(len(images)):
            exp_logits = np.exp(logits[i] - np.max(logits[i]))
            probs = exp_logits / exp_logits.sum()
            
            pred_idx = np.argmax(probs)
            pred_prob = probs[pred_idx]
            
            results.append({
                "predicted_class": self.class_names[pred_idx],
                "predicted_index": int(pred_idx),
                "confidence": float(pred_prob),
                "probabilities": {
                    name: float(probs[j])
                    for j, name in enumerate(self.class_names)
                }
            })
            
        return results


def create_predictor(
    model_path: str,
    backend: str = "pytorch",
    device: str = "auto",
    image_size: int = 224
) -> Union[PyTorchPredictor, ONNXPredictor]:
    """
    Factory function to create appropriate predictor.
    
    Args:
        model_path: Path to model file
        backend: "pytorch" or "onnx"
        device: Device to use
        image_size: Input image size
        
    Returns:
        Predictor instance
    """
    if backend.lower() == "onnx":
        return ONNXPredictor(model_path, image_size=image_size)
    else:
        return PyTorchPredictor(model_path, device=device, image_size=image_size)
