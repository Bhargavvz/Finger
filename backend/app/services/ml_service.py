"""
ML Service for Blood Group Prediction

Handles model loading and inference.
"""

import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from loguru import logger

from app.core.config import get_settings, BLOOD_GROUPS, IDX_TO_BLOOD_GROUP

settings = get_settings()


class Preprocessor:
    """Image preprocessing for inference."""
    
    def __init__(
        self,
        img_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to numpy
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize
        img_array = (img_array - self.mean) / self.std
        
        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        return img_array


class MLService:
    """Service for blood group prediction from fingerprint images."""
    
    _instance: Optional["MLService"] = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.model_type = None
        self.model_version = None
        self.preprocessor = Preprocessor()
        self._initialized = True
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the ML model."""
        model_path = model_path or settings.MODEL_PATH
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}")
            return False
        
        try:
            if model_path.suffix == ".onnx":
                self._load_onnx_model(model_path)
            else:
                self._load_pytorch_model(model_path)
            
            self.model_version = model_path.stem
            logger.info(f"Model loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_onnx_model(self, model_path: Path) -> None:
        """Load ONNX model."""
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if settings.MODEL_DEVICE == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.model = ort.InferenceSession(str(model_path), providers=providers)
        self.model_type = "onnx"
        
        # Get input name
        self._input_name = self.model.get_inputs()[0].name
    
    def _load_pytorch_model(self, model_path: Path) -> None:
        """Load PyTorch model."""
        import torch
        
        device = torch.device(settings.MODEL_DEVICE)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Import model architecture
        from ml.src.models import FingerprintClassifier
        
        self.model = FingerprintClassifier(num_classes=8)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        self.model_type = "pytorch"
        self._device = device
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        return_probabilities: bool = True
    ) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image: PIL Image, numpy array, or path to image
            return_probabilities: Whether to return all class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Preprocess
        input_tensor = self.preprocessor(image)
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        # Run inference
        if self.model_type == "onnx":
            outputs = self.model.run(None, {self._input_name: input_batch})
            logits = outputs[0][0]
        else:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(input_batch).to(self._device)
                outputs = self.model(tensor)
                logits = outputs[0].cpu().numpy()
        
        # Softmax
        probs = self._softmax(logits)
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        blood_group = IDX_TO_BLOOD_GROUP[pred_idx]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        result = {
            "blood_group": blood_group,
            "confidence": confidence,
            "inference_time_ms": inference_time,
        }
        
        if return_probabilities:
            result["probabilities"] = {
                BLOOD_GROUPS[i]: float(probs[i])
                for i in range(len(BLOOD_GROUPS))
            }
        
        return result
    
    def predict_batch(
        self,
        images: list,
        return_probabilities: bool = True
    ) -> list:
        """Make predictions on multiple images."""
        results = []
        for image in images:
            result = self.predict(image, return_probabilities)
            results.append(result)
        return results
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    @staticmethod
    def compute_image_hash(image_bytes: bytes) -> str:
        """Compute SHA256 hash of image."""
        return hashlib.sha256(image_bytes).hexdigest()


# Global instance
ml_service = MLService()


def get_ml_service() -> MLService:
    """Get the ML service instance."""
    return ml_service
