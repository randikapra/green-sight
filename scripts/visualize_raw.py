#!/usr/bin/env python3
"""Visualize raw satellite data for all study areas."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
import rasterio

sys.path.insert(0, '.')

from src.utils.config import load_config, setup_environment, resolve_periods, get_study_areas


def normalize_rgb(image, percentile_clip=2):
    out = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
        band = image[:, :, i]
        valid = band[~np.isnan(band)]
        if len(valid) == 0:
            continue
        lo = np.percentile(valid, percentile_clip)
        hi = np.percentile(valid, 100 - percentile_clip)
        out[:, :, i] = np.clip((band - lo) / (hi - lo + 1e-10), 0, 1)
    return out


def plot_raw_overview(data, city_name, period_label, classes, save_dir):
    b2, b3, b4, b8 = data[0], data[1], data[2], data[3]
    ndvi = data[4] if data.shape[0] == 5 else (b8 - b4) / (b8 + b4 + 1e-10)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Raw Satellite Data — {city_name} ({period_label})',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # True Color
    ax1 = fig.add_subplot(gs[0, 0])
    rgb = normalize_rgb(np.stack([b4, b3, b2], axis=-1))
    ax1.imshow(rgb)
    ax1.set_title('True Color (R-G-B)')
    ax1.axis('off')

    # False Color
    ax2 = fig.add_subplot(gs[0, 1])
    fc = normalize_rgb(np.stack([b8, b4, b3], axis=-1))
    ax2.imshow(fc)
    ax2.set_title('False Color NIR (vegetation = red)')
    ax2.axis('off')

    # NDVI
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(np.ma.masked_where(np.isnan(ndvi), ndvi), cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    ax3.set_title('NDVI')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Classification preview
    ax4 = fig.add_subplot(gs[1, 0])
    class_map = np.full(ndvi.shape, -1, dtype=np.int32)
    valid = ~np.isnan(ndvi)
    colors_list = []
    for cls_id in sorted(classes.keys()):
        cls = classes[cls_id]
        lo = cls.get('ndvi_min', -float('inf'))
        hi = cls.get('ndvi_max', float('inf'))
        class_map[valid & (ndvi >= lo) & (ndvi < hi)] = cls_id
        colors_list.append(cls['color'])

    cmap = ListedColormap(colors_list)
    ax4.imshow(np.ma.masked_where(class_map < 0, class_map), cmap=cmap,
               vmin=0, vmax=len(classes)-1)
    ax4.set_title('NDVI Classification Preview')
    ax4.axis('off')
    patches = [mpatches.Patch(color=classes[i]['color'], label=classes[i]['name'])
               for i in sorted(classes.keys())]
    ax4.legend(handles=patches, loc='lower right', fontsize=9)

    # Band histograms
    ax5 = fig.add_subplot(gs[1, 1])
    for band, name, color in zip([b2, b3, b4, b8],
                                  ['B2', 'B3', 'B4', 'B8'],
                                  ['#2196F3', '#4CAF50', '#F44336', '#9C27B0']):
        v = band[~np.isnan(band)].flatten()
        if len(v) > 0:
            ax5.hist(v, bins=100, alpha=0.5, label=name, color=color, density=True)
    ax5.set_xlabel('Reflectance')
    ax5.set_title('Band Histograms')
    ax5.legend()
    ax5.set_xlim(0, 0.5)
    ax5.grid(True, alpha=0.3)

    # NDVI histogram
    ax6 = fig.add_subplot(gs[1, 2])
    nv = ndvi[~np.isnan(ndvi)].flatten()
    if len(nv) > 0:
        ax6.hist(nv, bins=150, color='#4CAF50', alpha=0.7, density=True)

    # Threshold lines from this city's classes
    drawn = set()
    for cls_id in sorted(classes.keys()):
        cls = classes[cls_id]
        for key in ['ndvi_min', 'ndvi_max']:
            if key in cls and cls[key] not in drawn:
                ax6.axvline(x=cls[key], color='red', linestyle='--', linewidth=1.5)
                drawn.add(cls[key])

    ax6.set_xlabel('NDVI')
    ax6.set_title('NDVI Distribution + Thresholds')
    ax6.grid(True, alpha=0.3)

    save_path = Path(save_dir) / f'raw_overview_{period_label}.png'
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
    periods = resolve_periods(cfg)

    if args.city:
        study_areas = {args.city: study_areas[args.city]}

    for city_key, city_cfg in study_areas.items():
        city_name = city_cfg['name']
        classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}

        print(f"\n{'='*40}")
        print(f"City: {city_name}")
        print(f"{'='*40}")

        save_dir = Path(cfg.project.output_dir) / city_key / 'visualizations'
        save_dir.mkdir(parents=True, exist_ok=True)

        for period in periods:
            label = period['label']
            filepath = Path(cfg.data.raw_dir) / city_key / f"{city_key}_sentinel2_{label}.tif"

            if not filepath.exists():
                print(f"  Skipping {label}: {filepath} not found")
                continue

            print(f"  Processing: {label}")
            with rasterio.open(str(filepath)) as src:
                data = src.read()
            print(f"    Shape: {data.shape}")
            plot_raw_overview(data, city_name, period_label=label,
                             classes=classes, save_dir=save_dir)


if __name__ == '__main__':
    main()
