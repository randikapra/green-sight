#!/usr/bin/env python3
"""
Generate comprehensive research summary report from temporal analysis results.
Creates publication-ready tables and statistics.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

from src.utils.config import load_config, get_study_areas
from src.utils.logger import get_logger


def format_table(headers, rows, col_widths=None):
    """Format data as ASCII table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                     for i in range(len(headers))]
    
    sep = '+' + '+'.join('-' * w for w in col_widths) + '+'
    header_row = '|' + '|'.join(f'{h:^{w}}' for h, w in zip(headers, col_widths)) + '|'
    
    lines = [sep, header_row, sep]
    for row in rows:
        lines.append('|' + '|'.join(f'{str(c):^{w}}' for c, w in zip(row, col_widths)) + '|')
    lines.append(sep)
    
    return '\n'.join(lines)


def generate_report(cfg, logger):
    """Generate comprehensive research summary."""
    study_areas = get_study_areas(cfg)
    
    logger.info("=" * 80)
    logger.info("GREENSIGHT RESEARCH SUMMARY REPORT")
    logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Collect all results
    all_results = {}
    for city_key, city_cfg in study_areas.items():
        result_file = Path(cfg.project.output_dir) / city_key / 'temporal_analysis' / 'temporal_results.json'
        if result_file.exists():
            with open(result_file) as f:
                all_results[city_key] = json.load(f)
    
    if not all_results:
        logger.warning("No temporal analysis results found! Run scripts/analyze.py first.")
        return
    
    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("1. MULTI-CITY GREEN SPACE CHANGE SUMMARY")
    logger.info("=" * 80)
    
    headers = ['City', 'Climate Zone', 'Model Used', 'Change (%)', 'Status']
    rows = []
    
    for city_key, result in all_results.items():
        city_cfg = study_areas[city_key]
        change_pct = result.get('green_space_change_pct', 0)
        status = result.get('validation_status', 'unknown')
        
        # Status emoji
        if status == 'valid':
            status_display = '✓ Valid'
        elif status == 'flagged':
            status_display = '⚠ Flagged'
        elif status == 'rejected':
            status_display = '✗ Rejected'
        else:
            status_display = '? Unknown'
        
        rows.append([
            result['city'],
            city_cfg.get('climate_zone', 'N/A'),
            result['model_used'],
            f"{change_pct:+.1f}%",
            status_display
        ])
    
    print("\n" + format_table(headers, rows))
    
    # Detailed statistics per city
    logger.info("\n" + "=" * 80)
    logger.info("2. DETAILED LAND COVER STATISTICS")
    logger.info("=" * 80)
    
    for city_key, result in all_results.items():
        stats = result.get('stats', {})
        if not stats:
            continue
        
        periods = sorted(stats.keys())
        
        print(f"\n{result['city']} ({study_areas[city_key].get('climate_zone', 'N/A')})")
        print("-" * 70)
        
        # Get class names from first period
        first_period = periods[0]
        class_names = list(stats[first_period].keys())
        
        headers = ['Land Cover'] + [f"{p} (ha)" for p in periods] + [f"{p} (%)" for p in periods]
        rows = []
        
        for class_name in class_names:
            row = [class_name]
            # Areas
            for p in periods:
                area = stats[p].get(class_name, {}).get('area_hectares', 0)
                row.append(f"{area:.1f}")
            # Percentages
            for p in periods:
                pct = stats[p].get(class_name, {}).get('percentage', 0)
                row.append(f"{pct:.1f}")
            rows.append(row)
        
        print(format_table(headers, rows))
        
        # Change validation
        if result.get('validation_message'):
            print(f"\nValidation: {result['validation_message']}")
    
    # Model performance summary
    logger.info("\n" + "=" * 80)
    logger.info("3. MODEL PERFORMANCE")
    logger.info("=" * 80)
    
    headers = ['City', 'Model', 'Test F1', 'Test Accuracy', 'Status']
    rows = []
    
    for city_key, result in all_results.items():
        model_name = result['model_used']
        test_result_file = Path(cfg.project.output_dir) / city_key / model_name / 'test_results.json'
        
        if test_result_file.exists():
            with open(test_result_file) as f:
                test_res = json.load(f)
            
            rows.append([
                result['city'],
                model_name,
                f"{test_res.get('test_f1_weighted', 0):.4f}",
                f"{test_res.get('test_accuracy', 0):.4f}",
                '✓'
            ])
        else:
            rows.append([
                result['city'],
                model_name,
                'N/A',
                'N/A',
                '✗'
            ])
    
    print("\n" + format_table(headers, rows))
    
    # Configuration summary
    logger.info("\n" + "=" * 80)
    logger.info("4. CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    
    # Get config from first result
    first_result = list(all_results.values())[0]
    config_info = first_result.get('config', {})
    
    print(f"""
Prediction Confidence Threshold:  {config_info.get('min_prediction_confidence', 'N/A')}
Change Detection Confidence:      {config_info.get('min_confidence_for_change', 'N/A')}
Flag Threshold:                   ±{config_info.get('flag_threshold_pct', 'N/A')}%
Reject Threshold:                 ±{config_info.get('reject_threshold_pct', 'N/A')}%
""")
    
    # Validation recommendations
    logger.info("\n" + "=" * 80)
    logger.info("5. VALIDATION CHECKLIST")
    logger.info("=" * 80)
    
    flagged_cities = [city for city, res in all_results.items() 
                     if res.get('validation_status') == 'flagged']
    rejected_cities = [city for city, res in all_results.items() 
                      if res.get('validation_status') == 'rejected']
    
    print("""
Required Actions:
[ ] Visual inspection of validation grids (validation_grid.png)
[ ] Verify RGB imagery quality (clouds, haze)
[ ] Check NDVI-prediction alignment
[ ] Confirm spatial coherence of changes
[ ] Review per-class metrics for imbalances
""")
    
    if rejected_cities:
        print(f"\n⚠️  CRITICAL: {len(rejected_cities)} cities with REJECTED results:")
        for city in rejected_cities:
            print(f"    - {city}: {all_results[city].get('validation_message', 'Unknown issue')}")
        print("\n    → Must investigate before using in publication!")
    
    if flagged_cities:
        print(f"\n⚠️  WARNING: {len(flagged_cities)} cities with FLAGGED results:")
        for city in flagged_cities:
            print(f"    - {city}: {all_results[city].get('validation_message', 'Unexpected change')}")
        print("\n    → Recommend manual verification")
    
    valid_cities = [city for city, res in all_results.items() 
                   if res.get('validation_status') == 'valid']
    
    if valid_cities:
        print(f"\n✓ {len(valid_cities)} cities with valid results:")
        for city in valid_cities:
            change = all_results[city].get('green_space_change_pct', 0)
            print(f"    - {city}: {change:+.1f}%")
    
    # Save report
    report_dir = Path(cfg.project.output_dir) / 'summary'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Export as JSON
    summary_json = {
        'generated': datetime.now().isoformat(),
        'cities': all_results,
        'validation_summary': {
            'valid': valid_cities,
            'flagged': flagged_cities,
            'rejected': rejected_cities
        }
    }
    
    json_path = report_dir / 'research_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    logger.info(f"\n✓ Summary saved: {json_path}")
    
    # Next steps
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    print("""
1. Review validation visualizations:
   results/*/temporal_analysis/validation_grid.png

2. For flagged/rejected cities:
   - Check preprocessing logs for threshold issues
   - Inspect raw imagery for cloud contamination
   - Consider increasing confidence thresholds

3. For publication:
   - Use only 'valid' cities, or
   - Manually verify 'flagged' cities with domain knowledge
   - Document validation methodology in methods section

4. Generate figures:
   python scripts/generate_figures.py
""")


def main():
    parser = argparse.ArgumentParser(
        description='Generate research summary report'
    )
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    logger = get_logger('summary', cfg.project.log_dir)
    
    generate_report(cfg, logger)


if __name__ == '__main__':
    main()
