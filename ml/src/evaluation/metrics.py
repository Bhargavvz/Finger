"""
Evaluation Metrics and Analysis Module

Provides comprehensive metrics for model evaluation:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class metrics
- ROC-AUC curves
- Calibration analysis
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


CLASS_NAMES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: List[str] = CLASS_NAMES
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Macro averages (treat all classes equally)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        if i < len(per_class_precision):
            metrics[f"precision_{class_name}"] = per_class_precision[i]
            metrics[f"recall_{class_name}"] = per_class_recall[i]
            metrics[f"f1_{class_name}"] = per_class_f1[i]
    
    # ROC-AUC if probabilities provided
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
            metrics["roc_auc_weighted"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
    return cm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    cm = compute_confusion_matrix(y_true, y_pred, class_names, normalize)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
        
    return fig


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str] = CLASS_NAMES,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        metrics: Metrics dictionary
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    precision_vals = [metrics.get(f"precision_{c}", 0) for c in class_names]
    recall_vals = [metrics.get(f"recall_{c}", 0) for c in class_names]
    f1_vals = [metrics.get(f"f1_{c}", 0) for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width, precision_vals, width, label="Precision", color="#2ecc71")
    bars2 = ax.bar(x, recall_vals, width, label="Recall", color="#3498db")
    bars3 = ax.bar(x + width, f1_vals, width, label="F1-Score", color="#e74c3c")
    
    ax.set_xlabel("Blood Group")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8
            )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Per-class metrics plot saved to {save_path}")
        
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with lists of metric values
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", color="#3498db")
        axes[0].plot(history["val_loss"], label="Val Loss", color="#e74c3c")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy/F1 plot
    if "train_accuracy" in history and "val_accuracy" in history:
        axes[1].plot(history["train_accuracy"], label="Train Accuracy", color="#3498db")
        axes[1].plot(history["val_accuracy"], label="Val Accuracy", color="#e74c3c")
        
    if "val_f1" in history:
        axes[1].plot(history["val_f1"], label="Val F1", color="#2ecc71", linestyle="--")
        
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Training Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")
        
    return fig


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report as string
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )


class MetricsTracker:
    """
    Tracks metrics during training for history plotting.
    """
    
    def __init__(self):
        self.history = {}
        
    def update(self, metrics: Dict, prefix: str = "") -> None:
        """
        Update history with new metrics.
        
        Args:
            metrics: Dictionary of metric values
            prefix: Prefix for metric names (e.g., "train_", "val_")
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                key = f"{prefix}{name}" if prefix else name
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
                
    def get_history(self) -> Dict[str, List[float]]:
        """Get the full history."""
        return self.history
    
    def get_best(self, metric: str, mode: str = "max") -> Tuple[int, float]:
        """
        Get the best value and epoch for a metric.
        
        Args:
            metric: Metric name
            mode: "max" or "min"
            
        Returns:
            Tuple of (best_epoch, best_value)
        """
        if metric not in self.history:
            return -1, 0.0
            
        values = self.history[metric]
        if mode == "max":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
            
        return best_idx, values[best_idx]
