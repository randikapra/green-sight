"""
Temporal analysis: predict land cover for each period and compute change detection.
Supports per-city analysis with different class schemes.
IMPROVEMENTS:
- Confidence-based prediction filtering
- Sanity checks for unrealistic changes
- Statistical validation
- Quality metrics reporting
"""

import torch
import torch.nn as nn
import numpy as np
import rasterio
import json
from pathlib import Path

from src.models.factory import build_model
from src.utils.config import Config, get_enabled_models, resolve_periods


@torch.no_grad()
def predict_full_image(model, image_data, patch_size, stride, batch_size, device, 
                       min_confidence, logger):
    """
    Sliding-window prediction over full satellite image with confidence filtering.
    
    Args:
        min_confidence: Minimum confidence threshold (0-1) to accept predictions
    
    Returns:
        prediction_map: Class predictions (-1 for low confidence)
        confidence_map: Prediction confidence scores
    """
    model.eval()
    spectral = image_data[:4]
    _, height, width = spectral.shape

    prediction_map = np.full((height, width), -1, dtype=np.int32)
    confidence_map = np.zeros((height, width), dtype=np.float32)

    n_rows = (height - patch_size) // stride + 1
    n_cols = (width - patch_size) // stride + 1
    logger.info(f"    Predicting: {n_rows}x{n_cols} = {n_rows*n_cols} windows")

    batch_patches, batch_positions = [], []
    total_predictions = 0
    high_confidence_predictions = 0

    for i in range(n_rows):
        for j in range(n_cols):
            r, c = i * stride, j * stride
            patch = np.nan_to_num(
                spectral[:, r:r+patch_size, c:c+patch_size], nan=0.0
            ).astype(np.float32)
            batch_patches.append(patch)
            batch_positions.append((r, c))

            if len(batch_patches) >= batch_size or (i == n_rows-1 and j == n_cols-1):
                tensor = torch.FloatTensor(np.array(batch_patches)).to(device)
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

                for k, (rk, ck) in enumerate(batch_positions):
                    pred = preds[k].cpu().item()
                    conf = confs[k].cpu().item()
                    total_predictions += 1
                    
                    # Only update if confidence exceeds threshold AND higher than existing
                    if conf >= min_confidence:
                        high_confidence_predictions += 1
                        region_conf = confidence_map[rk:rk+patch_size, ck:ck+patch_size]
                        mask = conf > region_conf
                        prediction_map[rk:rk+patch_size, ck:ck+patch_size][mask] = pred
                        region_conf[mask] = conf

                batch_patches, batch_positions = [], []

    coverage = (prediction_map >= 0).sum() / prediction_map.size
    conf_ratio = high_confidence_predictions / total_predictions if total_predictions > 0 else 0
    logger.info(f"    Coverage: {coverage:.1%}")
    logger.info(f"    High-confidence predictions: {conf_ratio:.1%} (threshold: {min_confidence:.2f})")
    
    return prediction_map, confidence_map


def compute_stats(prediction_map, class_names, pixel_size_m):
    """Compute area statistics per class."""
    pixel_area_ha = (pixel_size_m ** 2) / 10000
    valid = prediction_map[prediction_map >= 0]
    total = len(valid)
    stats = {}
    for idx, name in enumerate(class_names):
        count = int(np.sum(valid == idx))
        stats[name] = {
            'pixel_count': count,
            'area_hectares': round(count * pixel_area_ha, 2),
            'percentage': round(count / total * 100, 2) if total > 0 else 0,
        }
    return stats


def compute_change(pred_t1, pred_t2, conf_t1, conf_t2, num_classes, green_idx, 
                   min_confidence_for_change, logger):
    """
    Compute transition matrix and change map with confidence filtering.
    
    Args:
        min_confidence_for_change: Only compute changes where BOTH time periods 
                                   have high confidence predictions
    """
    # Only consider pixels with high confidence in BOTH periods
    valid = (pred_t1 >= 0) & (pred_t2 >= 0) & \
            (conf_t1 >= min_confidence_for_change) & \
            (conf_t2 >= min_confidence_for_change)
    
    n_valid = valid.sum()
    n_total = (pred_t1 >= 0).sum()
    
    logger.info(f"    Using {n_valid:,} pixels for change detection " +
                f"({n_valid/n_total*100:.1f}% of predicted area)")
    
    p1, p2 = pred_t1[valid], pred_t2[valid]

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = int(np.sum((p1 == i) & (p2 == j)))

    change_map = np.full(pred_t1.shape, -1, dtype=np.int32)
    no_change = pred_t1 == pred_t2
    gained = (pred_t1 != green_idx) & (pred_t2 == green_idx)
    lost = (pred_t1 == green_idx) & (pred_t2 != green_idx)
    other = valid & ~no_change & ~gained & ~lost

    change_map[valid & no_change] = 0  # No change
    change_map[valid & gained] = 1      # Green space gain
    change_map[valid & lost] = 2        # Green space loss
    change_map[other] = 3                # Other transitions

    return matrix, change_map


def validate_change(change_pct, city_name, flag_threshold, reject_threshold, logger):
    """
    Validate detected change against sanity thresholds.
    
    Returns:
        status: 'valid', 'flagged', or 'rejected'
        message: Explanation
    """
    abs_change = abs(change_pct)
    
    if abs_change > reject_threshold:
        msg = f"⚠️  REJECTED: {abs_change:.1f}% change exceeds plausibility threshold " + \
              f"({reject_threshold}%). This likely indicates a data or model error."
        logger.error(f"  {msg}")
        return 'rejected', msg
    
    elif abs_change > flag_threshold:
        msg = f"⚠️  FLAGGED: {abs_change:.1f}% change exceeds expected threshold " + \
              f"({flag_threshold}%). Manual verification recommended."
        logger.warning(f"  {msg}")
        return 'flagged', msg
    
    else:
        msg = f"✓ Valid: {change_pct:+.1f}% change within expected range"
        logger.info(f"  {msg}")
        return 'valid', msg


def run_temporal_analysis_city(cfg, city_key, city_cfg, device, logger):
    """Run temporal analysis for one city with enhanced validation."""
    city_name = city_cfg['name']
    classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}
    class_names = [classes[i]['name'] for i in sorted(classes.keys())]
    num_classes = len(class_names)

    # Find green space class index
    green_idx = None
    for cid, cinfo in classes.items():
        if 'green' in cinfo['name'].lower():
            green_idx = cid
            break
    if green_idx is None:
        green_idx = num_classes - 1

    out_dir = Path(cfg.project.output_dir) / city_key / 'temporal_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find best model for this city
    city_results_dir = Path(cfg.project.output_dir) / city_key
    best_model_name = cfg.temporal.best_model
    if best_model_name == "auto":
        enabled = get_enabled_models(cfg)
        best_f1 = -1
        for mname in enabled:
            result_path = city_results_dir / mname / 'test_results.json'
            if result_path.exists():
                with open(result_path) as f:
                    res = json.load(f)
                if res['test_f1_weighted'] > best_f1:
                    best_f1 = res['test_f1_weighted']
                    best_model_name = mname
        if best_f1 < 0:
            logger.warning(f"  No trained models found for {city_name}. Skipping.")
            return None, None, None
        logger.info(f"  Best model: {best_model_name} (F1: {best_f1:.4f})")

    # Build and load model
    enabled_models = get_enabled_models(cfg)
    model_cfg = enabled_models[best_model_name]
    model = build_model(best_model_name, model_cfg, num_classes, input_channels=4)

    ckpt_path = city_results_dir / best_model_name / 'best_model.pth'
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Process each period
    periods = resolve_periods(cfg)
    patch_size = cfg.preprocessing.patch_size
    stride = cfg.temporal.prediction_stride
    batch_size = cfg.temporal.prediction_batch_size
    min_conf = cfg.temporal.min_prediction_confidence

    predictions = {}
    confidences = {}
    all_stats = {}

    for period in periods:
        label = period['label']
        filepath = Path(cfg.data.raw_dir) / city_key / f"{city_key}_sentinel2_{label}.tif"

        if not filepath.exists():
            logger.warning(f"  Image not found: {filepath}")
            continue

        logger.info(f"\n  Period: {label}")
        with rasterio.open(str(filepath)) as src:
            data = src.read()
        logger.info(f"    Image shape: {data.shape}")

        pred_map, conf_map = predict_full_image(
            model, data, patch_size, stride, batch_size, device, min_conf, logger
        )
        predictions[label] = pred_map
        confidences[label] = conf_map

        stats = compute_stats(pred_map, class_names, cfg.temporal.pixel_size_m)
        all_stats[label] = stats
        for cls, s in stats.items():
            logger.info(f"    {cls}: {s['area_hectares']} ha ({s['percentage']}%)")

    # Change detection with validation
    labels_sorted = sorted(predictions.keys())
    matrix, change_map, change_pct = None, None, None
    validation_status = None
    validation_msg = None

    if len(labels_sorted) >= 2:
        t1, t2 = labels_sorted[0], labels_sorted[-1]
        logger.info(f"\n  Change detection: {t1} -> {t2}")
        
        min_conf_change = cfg.temporal.change_detection.min_confidence_for_change
        matrix, change_map = compute_change(
            predictions[t1], predictions[t2], 
            confidences[t1], confidences[t2],
            num_classes, green_idx, min_conf_change, logger
        )

        green_t1 = all_stats[t1].get('Green Space', {}).get('area_hectares', 0)
        green_t2 = all_stats[t2].get('Green Space', {}).get('area_hectares', 0)
        
        if green_t1 > 0:
            change_pct = ((green_t2 - green_t1) / green_t1 * 100)
        else:
            change_pct = 0
            logger.warning(f"  No green space detected in {t1}, cannot compute change %")
        
        logger.info(f"  Green Space: {green_t1:.0f} ha -> {green_t2:.0f} ha ({change_pct:+.1f}%)")
        
        # Validate the change
        flag_thresh = cfg.temporal.change_detection.flag_threshold_pct
        reject_thresh = cfg.temporal.change_detection.reject_threshold_pct
        validation_status, validation_msg = validate_change(
            change_pct, city_name, flag_thresh, reject_thresh, logger
        )

    # Save
    results = {
        'city': city_name,
        'model_used': best_model_name,
        'stats': all_stats,
        'green_space_change_pct': round(change_pct, 2) if change_pct is not None else None,
        'validation_status': validation_status,
        'validation_message': validation_msg,
        'transition_matrix': matrix.tolist() if matrix is not None else None,
        'config': {
            'min_prediction_confidence': min_conf,
            'min_confidence_for_change': min_conf_change if len(labels_sorted) >= 2 else None,
            'flag_threshold_pct': flag_thresh if len(labels_sorted) >= 2 else None,
            'reject_threshold_pct': reject_thresh if len(labels_sorted) >= 2 else None,
        }
    }
    with open(out_dir / 'temporal_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        str(out_dir / 'prediction_maps.npz'),
        **{f'pred_{k}': v for k, v in predictions.items()},
        **{f'conf_{k}': v for k, v in confidences.items()},
        change_map=change_map if change_map is not None else np.array([]),
    )

    return results, predictions, change_map
