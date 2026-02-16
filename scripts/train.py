#!/usr/bin/env python3
"""Train all enabled models for each study area."""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, '.')

from src.utils.config import (
    load_config, setup_environment, get_enabled_models,
    get_study_areas, Config
)
from src.utils.logger import get_logger
from src.data.dataset import SatelliteDataset
from src.models.factory import build_model, get_model_info
from src.models.trainer import Trainer
from src.analysis.visualize import (
    plot_training_curves, plot_confusion_matrix, plot_model_comparison
)


def train_city(city_key: str, city_cfg: dict, cfg: Config, device, logger):
    """Train all models for a single city."""
    city_name = city_cfg['name']
    classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}
    class_names = [classes[i]['name'] for i in sorted(classes.keys())]
    num_classes = len(class_names)

    logger.info(f"\n{'#'*60}")
    logger.info(f"CITY: {city_name} ({num_classes} classes: {class_names})")
    logger.info(f"{'#'*60}")

    # Load dataset
    dataset_path = Path(cfg.data.processed_dir) / city_key / 'dataset.npz'
    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}. Run preprocess first. Skipping.")
        return []

    data = np.load(str(dataset_path))
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    logger.info(f"Data: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # Class weights
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    logger.info(f"Class weights: {dict(zip(class_names, np.round(class_weights, 3)))}")

    # Augmentation
    aug_cfg = cfg.training.augmentation
    if isinstance(aug_cfg, Config):
        aug_cfg = aug_cfg.to_dict()

    # DataLoaders
    bs = cfg.training.batch_size
    nw = cfg.project.num_workers
    train_ds = SatelliteDataset(X_train, y_train, augment=True, aug_cfg=aug_cfg)
    val_ds = SatelliteDataset(X_val, y_val, augment=False)
    test_ds = SatelliteDataset(X_test, y_test, augment=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True,
                                                num_workers=nw, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False,
                                              num_workers=nw, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False,
                                               num_workers=nw, pin_memory=True)

    # Output dir for this city
    city_output = Path(cfg.project.output_dir) / city_key
    city_output.mkdir(parents=True, exist_ok=True)

    enabled = get_enabled_models(cfg)
    all_histories = {}
    all_results = []

    for model_name, model_cfg in enabled.items():
        logger.info(f"\nBuilding: {model_name} for {city_name}")

        model = build_model(model_name, model_cfg, num_classes, input_channels=4)
        info = get_model_info(model)
        logger.info(f"  Params: {info['total_params_M']} total, {info['trainable_params_M']} trainable")

        # Override output dir for city-specific results
        city_cfg_override = Config({
            'project': cfg.project.to_dict(),
            'training': cfg.training.to_dict(),
            'distributed': cfg.distributed.to_dict(),
        })
        city_cfg_override.project.output_dir = str(city_output)
        city_cfg_override.project.log_dir = str(Path(cfg.project.log_dir) / city_key)

        trainer = Trainer(model, model_name, city_cfg_override, device, logger,
                         class_names, class_weights)

        history = trainer.train(train_loader, val_loader)
        all_histories[model_name] = history

        results, cm, preds, labels = trainer.test(test_loader)
        all_results.append(results)

        # Save
        result_path = city_output / model_name / 'test_results.json'
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)

        plot_confusion_matrix(cm, class_names, model_name, str(city_output))

        del model, trainer
        torch.cuda.empty_cache()

    # Comparison plots
    if len(all_histories) > 1:
        plot_training_curves(all_histories, str(city_output))
        plot_model_comparison(all_results, str(city_output))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS — {city_name}")
    logger.info(f"{'='*50}")
    logger.info(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
    logger.info("-" * 42)
    for r in all_results:
        logger.info(f"{r['model']:<20} {r['test_accuracy']:>10.4f} {r['test_f1_weighted']:>10.4f}")

    if all_results:
        best = max(all_results, key=lambda x: x['test_f1_weighted'])
        logger.info(f"Best: {best['model']} (F1: {best['test_f1_weighted']:.4f})")

    # Save city summary
    summary = {'city': city_name, 'results': all_results,
               'best_model': best['model'] if all_results else None}
    with open(city_output / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--city', default=None, help='Train for specific city only (e.g., colombo)')
    parser.add_argument('--model', default=None, help='Train specific model only (e.g., resnet50)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = setup_environment(cfg)
    logger = get_logger('train', cfg.project.log_dir)

    study_areas = get_study_areas(cfg)

    if args.city:
        if args.city not in study_areas:
            logger.error(f"City '{args.city}' not found. Available: {list(study_areas.keys())}")
            sys.exit(1)
        study_areas = {args.city: study_areas[args.city]}

    # If --model specified, temporarily disable all others
    if args.model:
        models = cfg.models.to_dict()
        if args.model not in models:
            logger.error(f"Model '{args.model}' not found.")
            sys.exit(1)
        for m in models:
            models[m]['enabled'] = (m == args.model)

    all_city_results = {}
    for city_key, city_cfg in study_areas.items():
        results = train_city(city_key, city_cfg, cfg, device, logger)
        all_city_results[city_key] = results

    # Cross-city comparison
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-CITY COMPARISON")
    logger.info(f"{'='*60}")
    for city_key, results in all_city_results.items():
        if results:
            best = max(results, key=lambda x: x['test_f1_weighted'])
            logger.info(f"  {city_key:<15} Best: {best['model']:<18} F1: {best['test_f1_weighted']:.4f}")


if __name__ == '__main__':
    main()
