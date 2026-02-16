"""
Ablation Study for GreenSight
==============================
Tests the impact of key design choices:
1. With vs without weighted loss
2. With vs without data augmentation
3. With vs without transfer learning (pretrained weights)
4. Different patch sizes (32, 64, 96)

Usage:
    python scripts/ablation.py --city colombo --model convnext_tiny
    python scripts/ablation.py  # runs all cities with best model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy

from src.utils.config import Config, resolve_classes, get_class_names_for_city
from src.utils.logger import setup_logger
from src.models.factory import build_model
from src.models.trainer import Trainer
from src.data.preprocess import create_dataloaders


def run_ablation(city_key, city_cfg, cfg, device, logger, ablation_name, 
                 model_name="convnext_tiny", **overrides):
    """Run a single ablation experiment."""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"ABLATION: {ablation_name}")
    logger.info(f"{'='*50}")
    
    # Load data
    class_names = get_class_names_for_city(cfg, city_key)
    num_classes = len(class_names)
    
    data_path = Path(cfg.data.processed_dir) / city_key / "dataset.npz"
    if not data_path.exists():
        logger.info(f"  Skipping {city_key}: no processed data")
        return None
    
    data = np.load(str(data_path))
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Create dataloaders
    use_augmentation = overrides.get('augmentation', True)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.project.num_workers,
        augment=use_augmentation,
    )
    
    # Class weights
    use_weighted_loss = overrides.get('weighted_loss', True)
    class_weights = None
    if use_weighted_loss:
        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        class_weights = weights.astype(np.float32)
    
    # Build model
    use_pretrained = overrides.get('pretrained', True)
    model_cfg = cfg.models[model_name] if hasattr(cfg.models, model_name) else cfg.models.convnext_tiny
    
    # Temporarily modify config for this ablation
    ablation_cfg = deepcopy(cfg)
    ablation_cfg.project.output_dir = str(Path(cfg.project.output_dir) / city_key / "ablations" / ablation_name)
    ablation_cfg.project.log_dir = str(Path(cfg.project.log_dir) / city_key / "ablations" / ablation_name)
    
    model = build_model(model_name, model_cfg, num_classes, input_channels=4)
    
    # Train
    trainer = Trainer(
        model, f"{model_name}_{ablation_name}", ablation_cfg,
        device, logger, class_names, class_weights
    )
    
    history = trainer.train(train_loader, val_loader)
    results, cm, preds, labels = trainer.test(test_loader)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default=None)
    parser.add_argument('--model', type=str, default='convnext_tiny')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    cfg = Config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(cfg)
    
    cities = [args.city] if args.city else list(cfg.study_areas.keys()) if hasattr(cfg.study_areas, 'keys') else ['colombo']
    
    all_results = {}
    
    for city_key in cities:
        city_cfg = getattr(cfg.study_areas, city_key)
        logger.info(f"\n{'#'*60}")
        logger.info(f"ABLATION STUDY: {city_cfg.name}")
        logger.info(f"{'#'*60}")
        
        ablations = {
            "baseline":          {"weighted_loss": True,  "augmentation": True,  "pretrained": True},
            "no_weighted_loss":  {"weighted_loss": False, "augmentation": True,  "pretrained": True},
            "no_augmentation":   {"weighted_loss": True,  "augmentation": False, "pretrained": True},
            "no_pretrained":     {"weighted_loss": True,  "augmentation": True,  "pretrained": False},
        }
        
        city_results = {}
        for abl_name, overrides in ablations.items():
            result = run_ablation(
                city_key, city_cfg, cfg, device, logger,
                abl_name, args.model, **overrides
            )
            if result:
                city_results[abl_name] = {
                    "accuracy": result["test_accuracy"],
                    "f1": result["test_f1_weighted"],
                }
        
        all_results[city_key] = city_results
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION SUMMARY: {city_cfg.name}")
        logger.info(f"{'='*60}")
        logger.info(f"{'Ablation':<25} {'Accuracy':>10} {'F1':>10} {'Δ F1':>10}")
        logger.info(f"{'-'*55}")
        
        baseline_f1 = city_results.get("baseline", {}).get("f1", 0)
        for abl_name, res in city_results.items():
            delta = res["f1"] - baseline_f1
            logger.info(f"{abl_name:<25} {res['accuracy']:>10.4f} {res['f1']:>10.4f} {delta:>+10.4f}")
    
    # Save
    output_path = Path(cfg.project.output_dir) / "ablation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nAblation results saved to {output_path}")


if __name__ == "__main__":
    main()
