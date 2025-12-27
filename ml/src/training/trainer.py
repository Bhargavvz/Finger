"""
Training Module for Fingerprint Blood Group Classifier

Implements:
- Training loop with validation
- Learning rate scheduling
- Early stopping
- Mixed precision training
- Gradient clipping
- Checkpointing
- Logging (TensorBoard/WandB)
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from loguru import logger

from ..models import FingerprintClassifier
from ..evaluation import compute_metrics, MetricsTracker


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Optimizer
    optimizer_name: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Scheduler
    scheduler_name: str = "cosine_annealing_warm_restarts"
    scheduler_params: Dict = field(default_factory=lambda: {"T_0": 10, "T_mult": 2, "eta_min": 1e-6})
    
    # Loss
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    
    # Training
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Checkpoint
    save_dir: str = "./checkpoints"
    save_top_k: int = 3
    monitor: str = "val_f1"
    mode: str = "max"  # "max" or "min"
    
    # Logging
    log_dir: str = "./logs"
    log_interval: int = 10


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""
    
    def __init__(
        self,
        save_dir: str,
        save_top_k: int = 3,
        monitor: str = "val_f1",
        mode: str = "max"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.checkpoints: List[Tuple[float, str]] = []
        
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object],
        epoch: int,
        metrics: Dict,
        config: Dict
    ) -> Optional[str]:
        """
        Save checkpoint if it's in top-k.
        
        Returns:
            Path to saved checkpoint or None
        """
        score = metrics.get(self.monitor, 0)
        
        # Check if this checkpoint should be saved
        should_save = len(self.checkpoints) < self.save_top_k
        if not should_save and self.checkpoints:
            worst_score = self.checkpoints[0][0]
            if self.mode == "max":
                should_save = score > worst_score
            else:
                should_save = score < worst_score
                
        if not should_save:
            return None
            
        # Save checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch}_{self.monitor}_{score:.4f}.pth"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update checkpoint list
        self.checkpoints.append((score, str(checkpoint_path)))
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        
        # Remove excess checkpoints
        while len(self.checkpoints) > self.save_top_k:
            _, path_to_remove = self.checkpoints.pop(0)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
                logger.info(f"Removed old checkpoint: {path_to_remove}")
                
        return str(checkpoint_path)
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1][1] if self.mode == "max" else self.checkpoints[0][1]


class Trainer:
    """
    Main trainer class for fingerprint classification model.
    """
    
    def __init__(
        self,
        model: FingerprintClassifier,
        config: Dict,
        device: str = "auto"
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            config: Full configuration dictionary
            device: Device to use (auto, cuda, cpu, mps)
        """
        self.config = config
        self.train_config = self._parse_training_config(config)
        
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
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = None  # Will be set with class weights
        
        # Mixed precision
        self.scaler = GradScaler() if self.train_config.mixed_precision and self.device.type == "cuda" else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.train_config.early_stopping_patience,
            mode=self.train_config.mode
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.train_config.save_dir,
            save_top_k=self.train_config.save_top_k,
            monitor=self.train_config.monitor,
            mode=self.train_config.mode
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.train_config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def _parse_training_config(self, config: Dict) -> TrainingConfig:
        """Parse training configuration."""
        tc = config.get("training", {})
        checkpoint_config = config.get("checkpoint", {})
        logging_config = config.get("logging", {})
        
        return TrainingConfig(
            epochs=tc.get("epochs", 100),
            early_stopping_patience=tc.get("early_stopping_patience", 15),
            optimizer_name=tc.get("optimizer", {}).get("name", "adamw"),
            learning_rate=tc.get("optimizer", {}).get("lr", 0.001),
            weight_decay=tc.get("optimizer", {}).get("weight_decay", 0.01),
            scheduler_name=tc.get("scheduler", {}).get("name", "cosine_annealing_warm_restarts"),
            scheduler_params={
                "T_0": tc.get("scheduler", {}).get("T_0", 10),
                "T_mult": tc.get("scheduler", {}).get("T_mult", 2),
                "eta_min": tc.get("scheduler", {}).get("eta_min", 1e-6)
            },
            label_smoothing=tc.get("loss", {}).get("label_smoothing", 0.1),
            use_class_weights=tc.get("loss", {}).get("use_class_weights", True),
            mixed_precision=tc.get("mixed_precision", True),
            gradient_clip_val=tc.get("gradient_clip_val", 1.0),
            accumulate_grad_batches=tc.get("accumulate_grad_batches", 1),
            save_dir=checkpoint_config.get("save_dir", "./checkpoints"),
            save_top_k=checkpoint_config.get("save_top_k", 3),
            monitor=checkpoint_config.get("monitor", "val_f1"),
            mode=checkpoint_config.get("mode", "max"),
            log_dir=logging_config.get("log_dir", "./logs")
        )
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        name = self.train_config.optimizer_name.lower()
        params = self.model.parameters()
        
        if name == "adam":
            return optim.Adam(
                params,
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay
            )
        elif name == "adamw":
            return optim.AdamW(
                params,
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay
            )
        elif name == "sgd":
            return optim.SGD(
                params,
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")
            
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler."""
        name = self.train_config.scheduler_name.lower()
        params = self.train_config.scheduler_params
        
        if name == "cosine_annealing_warm_restarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=params.get("T_0", 10),
                T_mult=params.get("T_mult", 2),
                eta_min=params.get("eta_min", 1e-6)
            )
        elif name == "cosine_annealing":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.epochs,
                eta_min=params.get("eta_min", 1e-6)
            )
        elif name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 30),
                gamma=params.get("gamma", 0.1)
            )
        elif name == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.train_config.mode,
                patience=params.get("patience", 5),
                factor=params.get("factor", 0.5)
            )
        else:
            logger.warning(f"Unknown scheduler: {name}, using None")
            return None
            
    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        """Set class weights for loss function."""
        if self.train_config.use_class_weights:
            class_weights = class_weights.to(self.device)
        else:
            class_weights = None
            
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.train_config.label_smoothing
        )
        
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                # Scale loss for accumulation
                loss = loss / self.train_config.accumulate_grad_batches
                
                self.scaler.scale(loss).backward()
                
                # Update weights after accumulation
                if (batch_idx + 1) % self.train_config.accumulate_grad_batches == 0:
                    # Gradient clipping
                    if self.train_config.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_config.gradient_clip_val
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.train_config.accumulate_grad_batches
                loss.backward()
                
                if (batch_idx + 1) % self.train_config.accumulate_grad_batches == 0:
                    if self.train_config.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_config.gradient_clip_val
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.train_config.accumulate_grad_batches
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item() * self.train_config.accumulate_grad_batches})
            
            # Log to TensorBoard
            if self.global_step % self.train_config.log_interval == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                
            self.global_step += 1
            
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        metrics["loss"] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        metrics["loss"] = avg_loss
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Optional class weights for loss
            
        Returns:
            Dictionary with best metrics and checkpoint path
        """
        # Set up loss function
        self.set_class_weights(class_weights if class_weights is not None else torch.ones(8))
        
        # Check if we need to unfreeze backbone
        freeze_epochs = self.config.get("model", {}).get("freeze_epochs", 0)
        backbone_frozen = self.config.get("model", {}).get("freeze_backbone", False)
        
        best_metrics = {}
        
        logger.info(f"Starting training for {self.train_config.epochs} epochs")
        
        for epoch in range(self.train_config.epochs):
            self.current_epoch = epoch
            
            # Unfreeze backbone after initial epochs
            if backbone_frozen and epoch == freeze_epochs:
                self.model.unfreeze_backbone()
                logger.info(f"Unfreezing backbone at epoch {epoch}")
                
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.train_config.monitor, val_metrics["loss"]))
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            prefixed_val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            checkpoint_path = self.checkpoint_manager.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                prefixed_val_metrics,
                self.config
            )
            
            # Update best metrics
            monitor_value = val_metrics.get(self.train_config.monitor.replace("val_", ""), 0)
            if not best_metrics or (
                (self.train_config.mode == "max" and monitor_value > best_metrics.get(self.train_config.monitor, 0)) or
                (self.train_config.mode == "min" and monitor_value < best_metrics.get(self.train_config.monitor, float("inf")))
            ):
                best_metrics = prefixed_val_metrics.copy()
                best_metrics["epoch"] = epoch
                
            # Early stopping
            if self.early_stopping(monitor_value):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Save final model
        self._save_final_model()
        
        self.writer.close()
        
        return {
            "best_metrics": best_metrics,
            "best_checkpoint": self.checkpoint_manager.get_best_checkpoint()
        }
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int) -> None:
        """Log metrics to console and TensorBoard."""
        # Console logging
        logger.info(
            f"Epoch {epoch + 1}/{self.train_config.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )
        
        # TensorBoard logging
        for name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{name}", value, epoch)
                
        for name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{name}", value, epoch)
                
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
        
    def _save_final_model(self) -> None:
        """Save the final model state."""
        final_path = Path(self.train_config.save_dir) / "final_model.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config
        }, final_path)
        logger.info(f"Saved final model to {final_path}")
        
        # Also save best model separately
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            import shutil
            best_path = Path(self.train_config.save_dir) / "best_model.pth"
            shutil.copy(best_checkpoint, best_path)
            logger.info(f"Copied best model to {best_path}")
