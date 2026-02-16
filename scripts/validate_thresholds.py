#!/usr/bin/env python3
"""
Validate NDVI thresholds against actual satellite data.
Generates NDVI histograms per city to verify threshold choices.
"""

import argparse
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '.')

from src.utils.config import load_config, get_study_areas, resolve_periods
from src.utils.logger import get_logger


def plot_ndvi_histogram(city_key, city_cfg, cfg, logger):
    """Plot NDVI distribution and mark configured thresholds."""
    city_name = city_cfg['name']
    classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"NDVI Distribution: {city_name}")
    logger.info(f"{'='*60}")
    
    periods = resolve_periods(cfg)
    
    fig, axes = plt.subplots(len(periods), 1, figsize=(12, 4*len(periods)))
    if len(periods) == 1:
        axes = [axes]
    
    all_ndvi = []
    
    for idx, period in enumerate(periods):
        label = period['label']
        img_path = Path(cfg.data.raw_dir) / city_key / f"{city_key}_sentinel2_{label}.tif"
        
        if not img_path.exists():
            logger.warning(f"  Image not found: {img_path}")
            continue
        
        logger.info(f"\n  Period: {label}")
        with rasterio.open(str(img_path)) as src:
            data = src.read()
        
        # Compute NDVI
        if data.shape[0] == 5:
            ndvi = data[4]
        else:
            nir, red = data[3], data[2]
            ndvi = (nir - red) / (nir + red + 1e-10)
        
        # Filter valid NDVI
        valid_ndvi = ndvi[~np.isnan(ndvi)]
        valid_ndvi = valid_ndvi[(valid_ndvi >= -0.5) & (valid_ndvi <= 1.0)]
        all_ndvi.append(valid_ndvi)
        
        # Statistics
        logger.info(f"    NDVI range: [{valid_ndvi.min():.3f}, {valid_ndvi.max():.3f}]")
        logger.info(f"    NDVI median: {np.median(valid_ndvi):.3f}")
        logger.info(f"    NDVI percentiles:")
        for p in [10, 25, 50, 75, 90]:
            logger.info(f"      {p}th: {np.percentile(valid_ndvi, p):.3f}")
        
        # Plot histogram
        ax = axes[idx]
        ax.hist(valid_ndvi, bins=100, range=(-0.2, 1.0), alpha=0.7, 
                color='green', edgecolor='black', density=True)
        ax.set_xlabel('NDVI', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{city_name} - {label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark configured thresholds
        for cls_id in sorted(classes.keys()):
            cls = classes[cls_id]
            
            # Lower bound
            if 'ndvi_min' in cls:
                t_min = cls['ndvi_min']
                ax.axvline(t_min, color=cls['color'], linestyle='--', linewidth=2, 
                          label=f"{cls['name']} min: {t_min:.2f}")
            
            # Upper bound
            if 'ndvi_max' in cls and cls['ndvi_max'] < 1.0:
                t_max = cls['ndvi_max']
                ax.axvline(t_max, color=cls['color'], linestyle='--', linewidth=2,
                          label=f"{cls['name']} max: {t_max:.2f}")
        
        ax.legend(loc='upper left', fontsize=9)
        
        # Highlight problematic regions
        ylim = ax.get_ylim()
        
        # Region with NDVI > 0.95 (should be minimal)
        high_ndvi_count = np.sum(valid_ndvi > 0.95)
        high_ndvi_pct = high_ndvi_count / len(valid_ndvi) * 100
        if high_ndvi_pct > 1.0:
            ax.axvspan(0.95, 1.0, alpha=0.2, color='red', 
                      label=f'NDVI > 0.95: {high_ndvi_pct:.1f}% (likely clouds!)')
            logger.warning(f"    ⚠️  {high_ndvi_pct:.1f}% of pixels have NDVI > 0.95 (cloud contamination?)")
    
    plt.tight_layout()
    
    out_dir = Path(cfg.project.output_dir) / city_key / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = out_dir / 'ndvi_histogram.png'
    fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    logger.info(f"\n  Histogram saved: {fig_path}")
    plt.close(fig)
    
    # Combined histogram for all periods
    if len(all_ndvi) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        combined = np.concatenate(all_ndvi)
        ax.hist(combined, bins=100, range=(-0.2, 1.0), alpha=0.7,
               color='green', edgecolor='black', density=True)
        ax.set_xlabel('NDVI', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'{city_name} - Combined NDVI Distribution (All Periods)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark thresholds
        for cls_id in [0, 1, 2, 3]:  # Water, Built-up, Barren, Green
            cls = classes[cls_id]
            if 'ndvi_min' in cls:
                t = cls['ndvi_min']
                ax.axvline(t, color=cls['color'], linestyle='--', linewidth=2.5,
                          label=f"{cls['name']}: {t:.2f}")
        
        # Mark green upper bound
        green_max = classes[3].get('ndvi_max', 1.0)
        if green_max < 1.0:
            ax.axvline(green_max, color=classes[3]['color'], linestyle='--', linewidth=2.5,
                      label=f"Green max: {green_max:.2f}")
        
        ax.legend(loc='upper left', fontsize=11)
        
        plt.tight_layout()
        combined_path = out_dir / 'ndvi_histogram_combined.png'
        fig.savefig(str(combined_path), dpi=200, bbox_inches='tight')
        logger.info(f"  Combined histogram saved: {combined_path}")
        plt.close(fig)
    
    # Recommendations
    logger.info(f"\n  THRESHOLD VALIDATION:")
    combined = np.concatenate(all_ndvi) if len(all_ndvi) > 1 else all_ndvi[0]
    
    # Check if green threshold captures reasonable amount
    green_min = classes[3]['ndvi_min']
    green_pct = np.sum(combined >= green_min) / len(combined) * 100
    
    if green_pct < 5:
        logger.warning(f"    ⚠️  Only {green_pct:.1f}% classified as green - threshold may be TOO HIGH")
        logger.warning(f"    → Consider lowering from {green_min:.2f} to {np.percentile(combined, 75):.2f}")
    elif green_pct > 50:
        logger.warning(f"    ⚠️  {green_pct:.1f}% classified as green - threshold may be TOO LOW")
        logger.warning(f"    → Consider raising from {green_min:.2f} to {np.percentile(combined, 60):.2f}")
    else:
        logger.info(f"    ✓ {green_pct:.1f}% classified as green - threshold looks reasonable")
    
    # Check upper bound effectiveness
    green_max = classes[3].get('ndvi_max', 1.0)
    if green_max < 1.0:
        excluded = np.sum(combined > green_max) / len(combined) * 100
        logger.info(f"    ✓ Upper bound {green_max:.2f} excludes {excluded:.2f}% of pixels")
        if excluded > 5:
            logger.warning(f"    ⚠️  Excluding {excluded:.1f}% might be too aggressive")


def main():
    parser = argparse.ArgumentParser(
        description='Validate NDVI thresholds against actual data'
    )
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--city', default=None, help='Analyze specific city only')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    logger = get_logger('threshold_validation', cfg.project.log_dir)
    
    study_areas = get_study_areas(cfg)
    
    if args.city:
        study_areas = {args.city: study_areas[args.city]}
    
    for city_key, city_cfg in study_areas.items():
        plot_ndvi_histogram(city_key, city_cfg, cfg, logger)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Threshold validation complete!")
    logger.info("  Review histograms in results/*/validation/ndvi_histogram.png")
    logger.info("  Adjust thresholds in configs/default.yaml if needed")


if __name__ == '__main__':
    main()