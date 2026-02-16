#!/usr/bin/env python3
"""Visualize processed patches for all study areas."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

from src.utils.config import load_config, setup_environment, get_study_areas


def normalize_patch_rgb(patch, percentile_clip=2):
    rgb = np.stack([patch[2], patch[1], patch[0]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[:, :, i]
        valid = band[band > 0]
        if len(valid) == 0:
            continue
        lo = np.percentile(valid, percentile_clip)
        hi = np.percentile(valid, 100 - percentile_clip)
        out[:, :, i] = np.clip((band - lo) / (hi - lo + 1e-10), 0, 1)
    return out


def plot_samples(X, y, classes, n_per_class, save_path):
    class_ids = sorted(classes.keys())
    n_classes = len(class_ids)

    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class*2.5, n_classes*2.5))
    fig.suptitle('Sample Patches by Class', fontsize=16, fontweight='bold', y=1.02)

    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for row, cls_id in enumerate(class_ids):
        cls_name = classes[cls_id]['name']
        idx = np.where(y == cls_id)[0]

        if len(idx) == 0:
            for col in range(n_per_class):
                axes[row, col].text(0.5, 0.5, 'No samples', ha='center', va='center')
                axes[row, col].axis('off')
        else:
            chosen = np.random.choice(idx, min(n_per_class, len(idx)), replace=False)
            for col in range(n_per_class):
                ax = axes[row, col]
                if col < len(chosen):
                    ax.imshow(normalize_patch_rgb(X[chosen[col]]))
                ax.axis('off')

        axes[row, 0].set_ylabel(f'{cls_name}\n(n={len(idx)})',
                                 fontsize=10, fontweight='bold', rotation=0,
                                 labelpad=65, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_distribution(y_train, y_val, y_test, classes, save_path):
    class_ids = sorted(classes.keys())
    class_names = [classes[c]['name'] for c in class_ids]
    class_colors = [classes[c]['color'] for c in class_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dataset Distribution', fontsize=14, fontweight='bold')

    x = np.arange(len(class_names))
    width = 0.25
    for i, (name, y_split) in enumerate([('Train', y_train), ('Val', y_val), ('Test', y_test)]):
        counts = [np.sum(y_split == c) for c in class_ids]
        bars = ax1.bar(x + i*width, counts, width, label=name, alpha=0.8)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        str(cnt), ha='center', fontsize=8)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(class_names, rotation=15)
    ax1.legend()
    ax1.set_ylabel('Patches')
    ax1.grid(True, alpha=0.3, axis='y')

    total = [np.sum(np.concatenate([y_train, y_val, y_test]) == c) for c in class_ids]
    ax2.pie(total, labels=class_names, colors=class_colors,
            autopct=lambda p: f'{p:.1f}%\n({int(p*sum(total)/100)})', startangle=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--city', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_environment(cfg)

    study_areas = get_study_areas(cfg)
    if args.city:
        study_areas = {args.city: study_areas[args.city]}

    for city_key, city_cfg in study_areas.items():
        city_name = city_cfg['name']
        classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}

        print(f"\n{'='*40}")
        print(f"City: {city_name}")

        dataset_path = Path(cfg.data.processed_dir) / city_key / 'dataset.npz'
        if not dataset_path.exists():
            print(f"  Dataset not found: {dataset_path}")
            continue

        data = np.load(str(dataset_path))
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']

        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])

        print(f"  Patches: {len(X_all)} (Train={len(X_train)} Val={len(X_val)} Test={len(X_test)})")

        save_dir = Path(cfg.project.output_dir) / city_key / 'visualizations'
        save_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(cfg.project.seed)
        plot_samples(X_all, y_all, classes, n_per_class=6, save_path=save_dir / 'patches.png')
        plot_distribution(y_train, y_val, y_test, classes, save_path=save_dir / 'distribution.png')


if __name__ == '__main__':
    main()
