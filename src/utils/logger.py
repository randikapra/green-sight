"""
Logging and metrics tracking utilities.
"""

import logging
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np


def get_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_fmt = logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}_{timestamp}.log'))
    fh.setLevel(level)
    fh_fmt = logging.Formatter('%(asctime)s | %(levelname)-5s | %(name)s | %(message)s')
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)
    
    return logger


class MetricsTracker:
    """
    Track and log training/validation metrics.
    Provides periodic summaries and saves history to JSON.
    """
    
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'lr': [],
            'epoch_time': [],
        }
        
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.start_time = None
    
    def start_epoch(self):
        """Call at the start of each epoch."""
        self.start_time = time.time()
    
    def end_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        """
        Log metrics at end of epoch.
        
        Args:
            epoch: current epoch number
            train_metrics: dict with 'loss', 'acc', 'f1'
            val_metrics: dict with 'loss', 'acc', 'f1'
            lr: current learning rate
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_acc'].append(train_metrics['acc'])
        self.history['val_acc'].append(val_metrics['acc'])
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['lr'].append(lr)
        self.history['epoch_time'].append(elapsed)
        
        is_best = val_metrics['f1'] > self.best_val_f1
        if is_best:
            self.best_val_f1 = val_metrics['f1']
            self.best_epoch = epoch
        
        return is_best
    
    def get_summary_str(self, epoch: int, total_epochs: int) -> str:
        """Format a one-line summary for the current epoch."""
        h = self.history
        i = -1  # last entry
        return (
            f"Epoch [{epoch+1}/{total_epochs}] "
            f"Loss: {h['train_loss'][i]:.4f}/{h['val_loss'][i]:.4f} "
            f"Acc: {h['train_acc'][i]:.4f}/{h['val_acc'][i]:.4f} "
            f"F1: {h['train_f1'][i]:.4f}/{h['val_f1'][i]:.4f} "
            f"LR: {h['lr'][i]:.6f} "
            f"({h['epoch_time'][i]:.1f}s)"
        )
    
    def get_detailed_report(self, epoch: int, val_report: dict) -> str:
        """Format a detailed per-class report (used at eval_interval)."""
        lines = [
            f"\n{'─'*50}",
            f"  Detailed Metrics @ Epoch {epoch+1}",
            f"{'─'*50}",
        ]
        for cls_name, metrics in val_report.items():
            if cls_name in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            lines.append(
                f"  {cls_name:<16} P: {metrics['precision']:.4f}  "
                f"R: {metrics['recall']:.4f}  F1: {metrics['f1-score']:.4f}  "
                f"N: {metrics['support']}"
            )
        if 'weighted avg' in val_report:
            w = val_report['weighted avg']
            lines.append(f"{'─'*50}")
            lines.append(
                f"  {'Weighted Avg':<16} P: {w['precision']:.4f}  "
                f"R: {w['recall']:.4f}  F1: {w['f1-score']:.4f}"
            )
        lines.append(f"{'─'*50}")
        return '\n'.join(lines)
    
    def save(self):
        """Save metrics history to JSON."""
        filepath = self.output_dir / 'metrics_history.json'
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def summary(self) -> dict:
        """Return summary stats."""
        return {
            'model': self.model_name,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.history['train_loss']),
            'total_time_min': sum(self.history['epoch_time']) / 60,
        }
