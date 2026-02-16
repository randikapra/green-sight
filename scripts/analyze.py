#!/usr/bin/env python3
"""Run temporal analysis and change detection for all cities."""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

from src.utils.config import (
    load_config, setup_environment, get_study_areas, resolve_periods
)
from src.utils.logger import get_logger
from src.analysis.temporal import run_temporal_analysis_city
from src.analysis.visualize import (
    plot_land_cover_maps, plot_change_map,
    plot_transition_matrix, plot_area_comparison
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--city', default=None, help='Analyze specific city only')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = setup_environment(cfg)
    logger = get_logger('temporal', cfg.project.log_dir)

    study_areas = get_study_areas(cfg)

    if args.city:
        study_areas = {args.city: study_areas[args.city]}

    all_city_results = {}

    for city_key, city_cfg in study_areas.items():
        city_name = city_cfg['name']
        classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}
        class_names = [classes[i]['name'] for i in sorted(classes.keys())]
        class_colors = [classes[i]['color'] for i in sorted(classes.keys())]

        logger.info(f"\n{'='*50}")
        logger.info(f"Temporal Analysis: {city_name}")
        logger.info(f"{'='*50}")

        out_dir = Path(cfg.project.output_dir) / city_key / 'temporal_analysis'
        out_dir.mkdir(parents=True, exist_ok=True)

        results, predictions, change_map = run_temporal_analysis_city(
            cfg, city_key, city_cfg, device, logger
        )

        if results is None:
            logger.warning(f"Skipping visualizations for {city_name}")
            continue

        all_city_results[city_key] = results

        # Generate visualizations
        logger.info(f"\n  Generating visualizations for {city_name}...")
        plot_land_cover_maps(predictions, class_names, class_colors, str(out_dir))

        if change_map is not None and change_map.size > 0:
            plot_change_map(change_map, str(out_dir))

        if results.get('transition_matrix'):
            matrix = np.array(results['transition_matrix'])
            plot_transition_matrix(matrix, class_names, str(out_dir))

        if results.get('stats'):
            plot_area_comparison(results['stats'], class_names, class_colors, str(out_dir))

    # Cross-city green space summary
    logger.info(f"\n{'='*60}")
    logger.info("GREEN SPACE CHANGE SUMMARY — ALL CITIES")
    logger.info(f"{'='*60}")
    for city_key, results in all_city_results.items():
        change = results.get('green_space_change_pct')
        if change is not None:
            logger.info(f"  {city_key:<15} Green Space Change: {change:+.1f}%")

    logger.info("\nTemporal analysis complete.")


if __name__ == '__main__':
    main()
