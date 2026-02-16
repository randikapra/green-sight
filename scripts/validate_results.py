#!/usr/bin/env python3
"""
Validation visualization script: Generate side-by-side visualizations for manual verification.
Creates RGB, NDVI, predictions, and confidence maps for visual quality assessment.
"""

import argparse
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.colors import ListedColormap

sys.path.insert(0, '.')

from src.utils.config import load_config, get_study_areas, resolve_periods
from src.utils.logger import get_logger


def normalize_rgb(rgb_bands):
    """Normalize RGB for visualization (2%, 98% stretch)."""
    rgb = np.stack([rgb_bands[2], rgb_bands[1], rgb_bands[0]], axis=-1)  # B4, B3, B2 -> RGB
    rgb = rgb.astype(np.float32)
    
    # Percentile normalization per band
    for i in range(3):
        p2, p98 = np.percentile(rgb[:,:,i][~np.isnan(rgb[:,:,i])], (2, 98))
        rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2 + 1e-10), 0, 1)
    
    rgb = np.nan_to_num(rgb, nan=0)
    return rgb


def plot_validation_grid(city_key, city_cfg, cfg, logger):
    """Create comprehensive validation visualization for one city."""
    city_name = city_cfg['name']
    classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}
    class_names = [classes[i]['name'] for i in sorted(classes.keys())]
    class_colors = [classes[i]['color'] for i in sorted(classes.keys())]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Validation Visualization: {city_name}")
    logger.info(f"{'='*60}")
    
    periods = resolve_periods(cfg)
    out_dir = Path(cfg.project.output_dir) / city_key / 'temporal_analysis'
    
    if not out_dir.exists():
        logger.warning(f"  No temporal analysis results found for {city_name}")
        return
    
    # Load predictions
    pred_file = out_dir / 'prediction_maps.npz'
    if not pred_file.exists():
        logger.warning(f"  No prediction maps found: {pred_file}")
        return
    
    pred_data = np.load(str(pred_file))
    
    # Create figure
    n_periods = len(periods)
    fig, axes = plt.subplots(n_periods, 5, figsize=(24, 5*n_periods))
    if n_periods == 1:
        axes = axes.reshape(1, -1)
    
    for idx, period in enumerate(periods):
        label = period['label']
        
        # Load satellite image
        img_path = Path(cfg.data.raw_dir) / city_key / f"{city_key}_sentinel2_{label}.tif"
        if not img_path.exists():
            logger.warning(f"  Image not found: {img_path}")
            continue
        
        with rasterio.open(str(img_path)) as src:
            data = src.read()
        
        # Get predictions and confidence
        pred_map = pred_data[f'pred_{label}']
        conf_map = pred_data.get(f'conf_{label}', np.ones_like(pred_map) * 0.5)
        
        # 1. RGB
        rgb = normalize_rgb(data)
        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f'{label} - True Color RGB', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # 2. NDVI
        if data.shape[0] == 5:
            ndvi = data[4]
        else:
            nir, red = data[3], data[2]
            ndvi = (nir - red) / (nir + red + 1e-10)
        
        ndvi_display = np.clip(ndvi, -0.5, 1.0)
        im_ndvi = axes[idx, 1].imshow(ndvi_display, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[idx, 1].set_title(f'{label} - NDVI', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        plt.colorbar(im_ndvi, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        
        # 3. Prediction map
        cmap = ListedColormap(class_colors)
        pred_display = np.ma.masked_where(pred_map < 0, pred_map)
        im_pred = axes[idx, 2].imshow(pred_display, cmap=cmap, vmin=0, vmax=len(class_names)-1)
        axes[idx, 2].set_title(f'{label} - Predictions', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Legend
        patches = [mpatches.Patch(color=class_colors[i], label=class_names[i]) 
                   for i in range(len(class_names))]
        axes[idx, 2].legend(handles=patches, loc='upper right', fontsize=9, 
                           framealpha=0.9, fancybox=True)
        
        # 4. Confidence map
        conf_display = np.ma.masked_where(pred_map < 0, conf_map)
        im_conf = axes[idx, 3].imshow(conf_display, cmap='plasma', vmin=0, vmax=1)
        axes[idx, 3].set_title(f'{label} - Confidence', fontsize=12, fontweight='bold')
        axes[idx, 3].axis('off')
        plt.colorbar(im_conf, ax=axes[idx, 3], fraction=0.046, pad=0.04)
        
        # 5. RGB + Prediction overlay
        axes[idx, 4].imshow(rgb)
        pred_overlay = np.ma.masked_where(pred_map < 0, pred_map)
        axes[idx, 4].imshow(pred_overlay, cmap=cmap, alpha=0.5, vmin=0, vmax=len(class_names)-1)
        axes[idx, 4].set_title(f'{label} - RGB + Predictions', fontsize=12, fontweight='bold')
        axes[idx, 4].axis('off')
        
        # Stats
        valid = pred_map >= 0
        coverage = valid.sum() / pred_map.size
        avg_conf = conf_map[valid].mean() if valid.any() else 0
        logger.info(f"  {label}: Coverage {coverage:.1%}, Avg Confidence {avg_conf:.3f}")
    
    # Change map (if available)
    if 'change_map' in pred_data and len(periods) >= 2:
        change_map = pred_data['change_map']
        if change_map.size > 0:
            fig_change, ax_change = plt.subplots(1, 2, figsize=(16, 8))
            
            # RGB from last period
            t2_data = data  # Already loaded from last period
            rgb_t2 = normalize_rgb(t2_data)
            ax_change[0].imshow(rgb_t2)
            ax_change[0].set_title(f'{periods[-1]["label"]} - RGB', fontsize=14, fontweight='bold')
            ax_change[0].axis('off')
            
            # Change map
            change_colors = ['#CCCCCC', '#4CAF50', '#F44336', '#FF9800']  # No change, Gain, Loss, Other
            change_cmap = ListedColormap(change_colors)
            change_display = np.ma.masked_where(change_map < 0, change_map)
            ax_change[1].imshow(change_display, cmap=change_cmap, vmin=0, vmax=3)
            ax_change[1].set_title(f'Green Space Change: {periods[0]["label"]} → {periods[-1]["label"]}', 
                                  fontsize=14, fontweight='bold')
            ax_change[1].axis('off')
            
            # Legend
            change_labels = ['No Change', 'Green Gain', 'Green Loss', 'Other Transition']
            patches_change = [mpatches.Patch(color=change_colors[i], label=change_labels[i]) 
                             for i in range(4)]
            ax_change[1].legend(handles=patches_change, loc='upper right', fontsize=11, 
                              framealpha=0.9, fancybox=True)
            
            plt.tight_layout()
            change_fig_path = out_dir / 'validation_change_map.png'
            fig_change.savefig(str(change_fig_path), dpi=200, bbox_inches='tight')
            logger.info(f"  Change map saved: {change_fig_path}")
            plt.close(fig_change)
    
    plt.tight_layout()
    fig_path = out_dir / 'validation_grid.png'
    fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    logger.info(f"  Validation grid saved: {fig_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate validation visualizations for manual quality assessment'
    )
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--city', default=None, help='Visualize specific city only')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    logger = get_logger('validation_viz', cfg.project.log_dir)
    
    study_areas = get_study_areas(cfg)
    
    if args.city:
        study_areas = {args.city: study_areas[args.city]}
    
    for city_key, city_cfg in study_areas.items():
        plot_validation_grid(city_key, city_cfg, cfg, logger)
    
    logger.info("\n✓ Validation visualizations complete!")
    logger.info("  Review the validation_grid.png files to verify:")
    logger.info("    1. RGB images match NDVI patterns")
    logger.info("    2. Predictions align with visible vegetation")
    logger.info("    3. Confidence is high in homogeneous areas")
    logger.info("    4. Changes are spatially coherent (not random noise)")


if __name__ == '__main__':
    main()
