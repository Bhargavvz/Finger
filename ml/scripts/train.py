"""
Training Script for Fingerprint Blood Group Classification

Usage:
    python train.py --config configs/efficientnet_config.yaml
    python train.py --config configs/efficientnet_config.yaml --resume checkpoints/best_model.pth
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_data_module
from src.models import create_model, count_parameters
from src.training import Trainer
from src.evaluation import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    generate_classification_report
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For full reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str) -> None:
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_dir / "training_{time}.log",
        rotation="100 MB",
        level="INFO"
    )


def main():
    parser = argparse.ArgumentParser(description="Train Fingerprint Blood Group Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/efficientnet_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set seed
    seed = args.seed or config.get("hardware", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Setup logging
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    setup_logging(log_dir)
    
    # Create data module
    logger.info("Preparing data...")
    data_module = create_data_module(config, seed=seed)
    
    # Get data loaders
    train_loader = data_module.get_train_dataloader(use_weighted_sampler=True)
    val_loader = data_module.get_val_dataloader()
    test_loader = data_module.get_test_dataloader()
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Log model info
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    device = config.get("hardware", {}).get("device", "auto")
    trainer = Trainer(model, config, device=device)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Train
    logger.info("Starting training...")
    results = trainer.fit(
        train_loader,
        val_loader,
        class_weights=data_module.class_weights
    )
    
    logger.info(f"Training completed!")
    logger.info(f"Best metrics: {results['best_metrics']}")
    logger.info(f"Best checkpoint: {results['best_checkpoint']}")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    
    # Load best model
    best_checkpoint = results["best_checkpoint"]
    if best_checkpoint:
        checkpoint = torch.load(best_checkpoint, map_location=trainer.device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Generate evaluation artifacts
    logger.info("Generating evaluation artifacts...")
    
    # Collect all predictions
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(trainer.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Save confusion matrix
    save_dir = Path(config.get("checkpoint", {}).get("save_dir", "./checkpoints"))
    
    plot_confusion_matrix(
        all_labels,
        all_preds,
        normalize=True,
        save_path=str(save_dir / "confusion_matrix.png")
    )
    
    # Save per-class metrics
    plot_per_class_metrics(
        test_metrics,
        save_path=str(save_dir / "per_class_metrics.png")
    )
    
    # Save classification report
    report = generate_classification_report(all_labels, all_preds)
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save final test metrics
    import json
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"All artifacts saved to {save_dir}")
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
