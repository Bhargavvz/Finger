"""
Evaluation Script

Standalone evaluation script for trained models.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data ../data
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_data_module
from src.models import create_model
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_per_class_metrics,
    generate_classification_report
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fingerprint Blood Group Classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/efficientnet_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data directory (overrides config)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "all"],
        default="test",
        help="Data split to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override data path if provided
    if args.data:
        config["data"]["data_dir"] = args.data
    
    config["data"]["batch_size"] = args.batch_size
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data module
    logger.info("Loading data...")
    data_module = create_data_module(config)
    
    # Get appropriate data loader
    if args.split == "val":
        data_loader = data_module.get_val_dataloader()
    elif args.split == "test":
        data_loader = data_module.get_test_dataloader()
    else:  # all
        # Combine all splits
        from torch.utils.data import ConcatDataset, DataLoader
        all_data = ConcatDataset([
            data_module.train_dataset,
            data_module.val_dataset,
            data_module.test_dataset
        ])
        data_loader = DataLoader(
            all_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.get("hardware", {}).get("num_workers", 4),
            pin_memory=True
        )
    
    logger.info(f"Evaluating on {args.split} split: {len(data_loader.dataset)} samples")
    
    # Create model
    logger.info("Loading model...")
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Collect predictions
    logger.info("Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(all_labels, all_preds)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    logger.info(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    logger.info("="*60)
    
    # Generate and save classification report
    class_names = data_module.class_names
    report = generate_classification_report(all_labels, all_preds, class_names)
    
    logger.info(f"\nClassification Report:\n{report}")
    
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Samples: {len(all_labels)}\n")
        f.write("="*60 + "\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names=class_names,
        normalize=True,
        save_path=str(cm_path),
        title=f"Confusion Matrix ({args.split} split)"
    )
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Save per-class metrics
    metrics_path = output_dir / "per_class_metrics.png"
    plot_per_class_metrics(
        metrics,
        class_names=class_names,
        save_path=str(metrics_path)
    )
    logger.info(f"Per-class metrics saved to {metrics_path}")
    
    # Save raw metrics
    import json
    metrics_json_path = output_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_json_path}")
    
    # Save predictions
    preds_path = output_dir / "predictions.npz"
    np.savez(
        preds_path,
        predictions=all_preds,
        labels=all_labels,
        probabilities=all_probs,
        class_names=class_names
    )
    logger.info(f"Predictions saved to {preds_path}")
    
    # Per-class accuracy analysis
    logger.info("\n" + "="*60)
    logger.info("PER-CLASS ANALYSIS")
    logger.info("="*60)
    
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            logger.info(f"{class_name}: {mask.sum()} samples, {class_acc:.4f} accuracy")
    
    # Most confused pairs
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    logger.info("\n" + "="*60)
    logger.info("MOST CONFUSED PAIRS")
    logger.info("="*60)
    
    # Get off-diagonal elements
    n_classes = len(class_names)
    confusions = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((i, j, cm[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, count in confusions[:10]:
        logger.info(f"{class_names[i]} -> {class_names[j]}: {count} samples")
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
