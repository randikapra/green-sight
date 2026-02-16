#!/usr/bin/env python3
"""
Enhanced temporal analysis with comprehensive validation and policy recommendations.

This script integrates:
1. Data quality validation
2. Uncertainty-aware temporal analysis
3. Statistical significance testing
4. Policy recommendation generation
5. Interactive visualizations
"""

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
from src.data.validation import validate_dataset, create_quality_report
from src.analysis.temporal_enhanced import run_temporal_analysis_city
from src.analysis.policy import run_policy_analysis
from src.analysis.visualize import (
    plot_land_cover_maps, plot_change_map,
    plot_transition_matrix, plot_area_comparison
)


def main():
    parser = argparse.ArgumentParser(
        description='Run enhanced temporal analysis with validation and policy recommendations'
    )
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--city', default=None, help='Analyze specific city only')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='Skip data validation (not recommended)')
    parser.add_argument('--skip-policy', action='store_true',
                       help='Skip policy recommendation generation')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = setup_environment(cfg)
    logger = get_logger('temporal_enhanced', cfg.project.log_dir)

    logger.info("="*80)
    logger.info("ENHANCED TEMPORAL ANALYSIS WITH UNCERTAINTY QUANTIFICATION")
    logger.info("="*80)

    study_areas = get_study_areas(cfg)
    periods = resolve_periods(cfg)

    if args.city:
        study_areas = {args.city: study_areas[args.city]}

    all_city_results = {}
    
    for city_key, city_cfg in study_areas.items():
        city_name = city_cfg['name']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"CITY: {city_name} ({city_cfg.get('climate_zone', '')})")
        logger.info(f"{'='*80}")

        # Phase 1: Data Validation
        if not args.skip_validation:
            logger.info("\n[PHASE 1] DATA QUALITY VALIDATION")
            logger.info("-" * 60)
            
            data_dir = Path(cfg.data.raw_dir) / city_key
            validation_results = validate_dataset(data_dir, periods, logger)
            
            # Save validation report
            out_dir = Path(cfg.project.output_dir) / city_key / 'temporal_analysis'
            out_dir.mkdir(parents=True, exist_ok=True)
            create_quality_report(validation_results, out_dir / 'quality_report.json')
            
            # Check if data quality is acceptable
            acceptable = True
            for period, metrics in validation_results.items():
                if hasattr(metrics, 'is_acceptable') and not metrics.is_acceptable():
                    logger.error(f"  Data quality for {period} is below threshold!")
                    logger.error(f"  Quality score: {metrics.quality_score:.1f}/100")
                    acceptable = False
            
            if not acceptable:
                logger.warning(f"  Proceeding despite quality issues - results may be unreliable!")
                logger.warning(f"  Consider re-collecting data or adjusting time periods.")
        
        # Phase 2: Temporal Analysis with Uncertainty
        logger.info("\n[PHASE 2] TEMPORAL ANALYSIS WITH UNCERTAINTY QUANTIFICATION")
        logger.info("-" * 60)
        
        classes = {int(k): v for k, v in city_cfg.get('classes', {}).items()}
        class_names = [classes[i]['name'] for i in sorted(classes.keys())]
        class_colors = [classes[i]['color'] for i in sorted(classes.keys())]
        
        out_dir = Path(cfg.project.output_dir) / city_key / 'temporal_analysis'
        
        results, predictions, change_map = run_temporal_analysis_city(
            cfg, city_key, city_cfg, device, logger
        )
        
        if results is None:
            logger.warning(f"Skipping {city_name} - no trained models found")
            continue
        
        all_city_results[city_key] = results
        
        # Phase 3: Policy Recommendations
        if not args.skip_policy and change_map is not None:
            logger.info("\n[PHASE 3] POLICY RECOMMENDATION GENERATION")
            logger.info("-" * 60)
            
            # Load prediction maps
            pred_data = np.load(out_dir / 'prediction_maps.npz')
            labels_sorted = sorted([k.replace('pred_', '') for k in pred_data.keys() 
                                   if k.startswith('pred_')])
            
            if len(labels_sorted) >= 2:
                t1, t2 = labels_sorted[0], labels_sorted[-1]
                pred_t1 = pred_data[f'pred_{t1}']
                pred_t2 = pred_data[f'pred_{t2}']
                
                recommendations, report_path = run_policy_analysis(
                    change_map, pred_t1, pred_t2,
                    cfg.temporal.pixel_size_m,
                    city_name,
                    out_dir,
                    logger
                )
        
        # Phase 4: Visualizations
        logger.info("\n[PHASE 4] GENERATING VISUALIZATIONS")
        logger.info("-" * 60)
        
        logger.info("  Creating land cover maps...")
        plot_land_cover_maps(predictions, class_names, class_colors, str(out_dir))
        
        if change_map is not None and change_map.size > 0:
            logger.info("  Creating change map...")
            plot_change_map(change_map, str(out_dir))
        
        if results.get('transition_matrix'):
            logger.info("  Creating transition matrix...")
            matrix = np.array(results['transition_matrix'])
            plot_transition_matrix(matrix, class_names, str(out_dir))
        
        if results.get('stats'):
            logger.info("  Creating area comparison plots...")
            plot_area_comparison(results['stats'], class_names, class_colors, str(out_dir))
        
        logger.info(f"\n  All outputs saved to: {out_dir}")
    
    # Cross-city summary
    if len(all_city_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info("CROSS-CITY GREEN SPACE CHANGE SUMMARY")
        logger.info(f"{'='*80}\n")
        
        summary_data = []
        for city_key, results in all_city_results.items():
            change = results.get('green_space_change_pct')
            sig_test = results.get('statistical_test', {})
            
            if change is not None:
                sig_marker = "***" if sig_test.get('is_significant') else "n.s."
                effect = sig_test.get('interpretation', 'unknown')
                
                logger.info(f"  {city_key:<15} Change: {change:+6.1f}%  "
                          f"[p={sig_test.get('p_value', 1.0):.4f} {sig_marker}, "
                          f"effect: {effect}]")
                
                summary_data.append({
                    'city': city_key,
                    'change_pct': change,
                    'p_value': sig_test.get('p_value'),
                    'is_significant': sig_test.get('is_significant'),
                    'effect_size': sig_test.get('effect_size'),
                })
        
        # Save cross-city summary
        summary_path = Path(cfg.project.output_dir) / 'cross_city_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"\nCross-city summary saved to: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("ENHANCED TEMPORAL ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info("\nKey outputs for each city:")
    logger.info("  - temporal_results.json: Comprehensive analysis results")
    logger.info("  - quality_report.json: Data quality assessment")
    logger.info("  - policy_recommendations.{json,md}: Evidence-based recommendations")
    logger.info("  - prediction_maps.npz: Predictions with uncertainty")
    logger.info("  - Visualizations: Land cover maps, change maps, transition matrices")


if __name__ == '__main__':
    main()
