"""
Training engine with:
- Periodic detailed evaluation at configurable intervals
- Checkpoint saving
- Early stopping
- Optional multi-GPU (DataParallel)
- TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from src.utils.config import Config
from src.utils.logger import MetricsTracker


class Trainer:
    """Training and evaluation engine."""
    
    def __init__(self, model: nn.Module, model_name: str, cfg: Config, 
                 device: torch.device, logger, class_names: list, 
                 class_weights: np.ndarray = None):
        self.model_name = model_name
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.class_names = class_names
        
        # Multi-GPU
        if cfg.distributed.enabled and torch.cuda.device_count() > 1:
            gpu_ids = cfg.distributed.gpu_ids
            logger.info(f"  Using DataParallel on GPUs: {gpu_ids}")
            self.model = nn.DataParallel(model, device_ids=gpu_ids).to(device)
        else:
            self.model = model.to(device)
        
        # Loss (with optional class weights for imbalanced data)
        if class_weights is not None:
            weight_tensor = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            logger.info(f"  Using weighted loss: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        opt_cfg = cfg.training.optimizer
        if opt_cfg.name == 'adamw':
            self.optimizer = optim.AdamW(trainable, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name == 'adam':
            self.optimizer = optim.Adam(trainable, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
        else:
            self.optimizer = optim.SGD(trainable, lr=opt_cfg.lr, momentum=0.9, weight_decay=opt_cfg.weight_decay)
        
        # Scheduler
        sched_cfg = cfg.training.scheduler
        if sched_cfg.name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.training.num_epochs
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
            )
        
        # Output
        self.model_dir = Path(cfg.project.output_dir) / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(Path(cfg.project.log_dir) / model_name))
        
        # Metrics tracker
        self.tracker = MetricsTracker(model_name, cfg.project.output_dir)
        
        # Early stopping
        self.patience_counter = 0
    
    def _run_epoch(self, dataloader: DataLoader, train: bool = True):
        """Run one epoch (train or eval)."""
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        ctx = torch.no_grad() if not train else _nullcontext()
        with ctx:
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                if train:
                    loss.backward()
                    self.optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        n = len(dataloader.dataset)
        metrics = {
            'loss': running_loss / n,
            'acc': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }
        
        return metrics, all_preds, all_labels
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Full training loop."""
        cfg = self.cfg.training
        num_epochs = cfg.num_epochs
        eval_interval = cfg.eval_interval
        save_every = cfg.save_every
        patience = cfg.early_stopping_patience
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training: {self.model_name}")
        self.logger.info(f"{'='*60}")
        
        for epoch in range(num_epochs):
            self.tracker.start_epoch()
            
            # Train
            train_metrics, _, _ = self._run_epoch(train_loader, train=True)
            
            # Validate
            val_metrics, val_preds, val_labels = self._run_epoch(val_loader, train=False)
            
            # LR step
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            # Track
            is_best = self.tracker.end_epoch(epoch, train_metrics, val_metrics, lr)
            
            # Log to tensorboard
            self.writer.add_scalars('Loss', {'train': train_metrics['loss'], 'val': val_metrics['loss']}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_metrics['acc'], 'val': val_metrics['acc']}, epoch)
            self.writer.add_scalars('F1', {'train': train_metrics['f1'], 'val': val_metrics['f1']}, epoch)
            self.writer.add_scalar('LR', lr, epoch)
            
            # Print summary
            summary = self.tracker.get_summary_str(epoch, num_epochs)
            self.logger.info(summary)
            
            # Detailed per-class report at intervals
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                all_label_ids = list(range(len(self.class_names)))
                report = classification_report(
                    val_labels, val_preds,
                    labels=all_label_ids,
                    target_names=self.class_names,
                    digits=4, output_dict=True, zero_division=0
                )
                detail = self.tracker.get_detailed_report(epoch, report)
                self.logger.info(detail)
            
            # Save best model
            if is_best:
                self._save_checkpoint(epoch, val_metrics, 'best_model.pth')
                self.logger.info(f"  ✓ Best model saved (F1: {val_metrics['f1']:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Periodic checkpoint
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, val_metrics, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if self.patience_counter >= patience:
                self.logger.info(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        self.tracker.save()
        self.writer.close()
        
        summary = self.tracker.summary()
        self.logger.info(f"\nTraining complete: {summary['total_time_min']:.1f} min, "
                         f"best F1: {summary['best_val_f1']:.4f} @ epoch {summary['best_epoch']+1}")
        
        return self.tracker.history
    
    def test(self, test_loader: DataLoader) -> dict:
        """Evaluate on test set using best model."""
        # Load best
        ckpt_path = self.model_dir / 'best_model.pth'
        checkpoint = torch.load(str(ckpt_path), map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        metrics, preds, labels = self._run_epoch(test_loader, train=False)
        
        all_label_ids = list(range(len(self.class_names)))
        report_str = classification_report(
            labels, preds, labels=all_label_ids,
            target_names=self.class_names, digits=4, zero_division=0
        )
        report_dict = classification_report(
            labels, preds, labels=all_label_ids,
            target_names=self.class_names, digits=4, 
            output_dict=True, zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=all_label_ids)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TEST RESULTS: {self.model_name}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"\n{report_str}")
        
        results = {
            'model': self.model_name,
            'test_accuracy': metrics['acc'],
            'test_f1_weighted': metrics['f1'],
            'classification_report': report_dict,
            'confusion_matrix': cm.tolist(),
            'best_epoch': checkpoint['epoch'],
        }
        
        return results, cm, np.array(preds), np.array(labels)
    
    def _save_checkpoint(self, epoch: int, metrics: dict, filename: str):
        """Save model checkpoint."""
        state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) \
                else self.model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1': metrics['f1'],
            'val_acc': metrics['acc'],
        }, str(self.model_dir / filename))


class _nullcontext:
    """Dummy context manager for Python < 3.7 compat."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
