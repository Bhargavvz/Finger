"""
Data Loading and Preprocessing Module
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from loguru import logger


class FingerprintDataset(Dataset):
    """
    Custom Dataset for Fingerprint Blood Group Classification
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to fingerprint images
            labels: List of corresponding labels (0-7)
            transform: Albumentations transform pipeline
            class_names: List of class names for reference
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or [
            "A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"
        ]
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index."""
        return self.class_names[label]


class DataModule:
    """
    Data Module for managing data loading and preprocessing
    """
    
    # Mapping from folder names to class indices
    CLASS_MAPPING = {
        "A+": 0, "A-": 1, "AB+": 2, "AB-": 3,
        "B+": 4, "B-": 5, "O+": 6, "O-": 7
    }
    
    CLASS_NAMES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    
    def __init__(self, config: Dict):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config["data"]["data_dir"])
        self.image_size = config["data"]["image_size"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.val_split = config["data"]["val_split"]
        self.test_split = config["data"]["test_split"]
        self.pin_memory = config["data"]["pin_memory"]
        
        # Augmentation config
        self.aug_config = config.get("augmentation", {})
        
        # Data containers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        
    def prepare_data(self) -> None:
        """
        Scan data directory and prepare file paths and labels.
        """
        logger.info(f"Scanning data directory: {self.data_dir}")
        
        image_paths = []
        labels = []
        
        for class_name, class_idx in self.CLASS_MAPPING.items():
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            # Get all image files
            extensions = ["*.bmp", "*.BMP", "*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
            class_images = []
            
            for ext in extensions:
                class_images.extend(list(class_dir.glob(ext)))
            
            logger.info(f"Found {len(class_images)} images for class {class_name}")
            
            for img_path in class_images:
                image_paths.append(str(img_path))
                labels.append(class_idx)
        
        self.image_paths = image_paths
        self.labels = labels
        
        logger.info(f"Total images found: {len(image_paths)}")
        
        # Compute class weights for handling imbalance
        self._compute_class_weights()
        
    def _compute_class_weights(self) -> None:
        """
        Compute class weights for handling class imbalance.
        """
        label_counts = np.bincount(self.labels, minlength=len(self.CLASS_NAMES))
        total_samples = len(self.labels)
        
        # Inverse frequency weighting
        self.class_weights = total_samples / (len(self.CLASS_NAMES) * label_counts + 1e-6)
        self.class_weights = torch.FloatTensor(self.class_weights)
        
        logger.info(f"Class distribution: {dict(zip(self.CLASS_NAMES, label_counts))}")
        logger.info(f"Class weights: {dict(zip(self.CLASS_NAMES, self.class_weights.numpy().round(2)))}")
        
    def setup(self, seed: int = 42) -> None:
        """
        Split data into train/val/test sets and create datasets.
        
        Args:
            seed: Random seed for reproducibility
        """
        logger.info("Setting up data splits...")
        
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.image_paths,
            self.labels,
            test_size=self.test_split,
            stratify=self.labels,
            random_state=seed
        )
        
        # Second split: separate validation from training
        val_size = self.val_split / (1 - self.test_split)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=seed
        )
        
        logger.info(f"Train samples: {len(train_paths)}")
        logger.info(f"Validation samples: {len(val_paths)}")
        logger.info(f"Test samples: {len(test_paths)}")
        
        # Create datasets
        self.train_dataset = FingerprintDataset(
            train_paths,
            train_labels,
            transform=self._get_train_transforms(),
            class_names=self.CLASS_NAMES
        )
        
        self.val_dataset = FingerprintDataset(
            val_paths,
            val_labels,
            transform=self._get_val_transforms(),
            class_names=self.CLASS_NAMES
        )
        
        self.test_dataset = FingerprintDataset(
            test_paths,
            test_labels,
            transform=self._get_val_transforms(),
            class_names=self.CLASS_NAMES
        )
        
        # Store for weighted sampler
        self.train_labels = train_labels
        
    def _get_train_transforms(self) -> A.Compose:
        """
        Get training augmentation pipeline.
        
        Returns:
            Albumentations Compose object
        """
        aug = self.aug_config.get("train", {})
        norm = self.aug_config.get("normalize", {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        })
        
        transforms_list = [
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
            A.VerticalFlip(p=aug.get("vertical_flip", 0.3)),
            A.Rotate(limit=aug.get("rotation_limit", 30), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness_limit", 0.2),
                contrast_limit=aug.get("contrast_limit", 0.2),
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=aug.get("gaussian_blur", 0.1)),
            A.GaussNoise(var_limit=(10.0, 50.0), p=aug.get("gaussian_noise", 0.1)),
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=aug.get("elastic_transform", 0.2)
            ),
            A.GridDistortion(p=aug.get("grid_distortion", 0.2)),
            A.CoarseDropout(
                max_holes=aug.get("coarse_dropout", {}).get("max_holes", 8),
                max_height=aug.get("coarse_dropout", {}).get("max_height", 16),
                max_width=aug.get("coarse_dropout", {}).get("max_width", 16),
                fill_value=0,
                p=aug.get("coarse_dropout", {}).get("prob", 0.3)
            ),
            A.Normalize(mean=norm["mean"], std=norm["std"]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms_list)
    
    def _get_val_transforms(self) -> A.Compose:
        """
        Get validation/test transforms (no augmentation).
        
        Returns:
            Albumentations Compose object
        """
        norm = self.aug_config.get("normalize", {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        })
        
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=norm["mean"], std=norm["std"]),
            ToTensorV2()
        ])
    
    def get_train_dataloader(self, use_weighted_sampler: bool = True) -> DataLoader:
        """
        Get training DataLoader with optional weighted sampling.
        
        Args:
            use_weighted_sampler: Whether to use weighted random sampling
            
        Returns:
            Training DataLoader
        """
        sampler = None
        shuffle = True
        
        if use_weighted_sampler:
            # Create sample weights based on class weights
            sample_weights = [self.class_weights[label].item() for label in self.train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


def create_data_module(config: Dict, seed: int = 42) -> DataModule:
    """
    Factory function to create and setup a DataModule.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        Configured DataModule
    """
    data_module = DataModule(config)
    data_module.prepare_data()
    data_module.setup(seed=seed)
    return data_module
