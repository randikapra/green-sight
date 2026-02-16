"""
Policy Recommendation Engine for Urban Green Space Planning

This module analyzes spatial patterns of green space change and generates
actionable recommendations for urban planners and policymakers.

Key features:
1. Hotspot detection (areas of rapid green loss/gain)
2. Spatial clustering and neighborhood analysis
3. Green space accessibility metrics
4. Priority zone identification
5. Evidence-based policy recommendations
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Hotspot:
    """Represents a spatial hotspot of change."""
    type: str  # 'loss' or 'gain'
    center: Tuple[int, int]  # (row, col)
    area_ha: float
    intensity: float  # Average change magnitude
    pixels: np.ndarray  # Mask of affected pixels


@dataclass
class PolicyRecommendation:
    """Structured policy recommendation."""
    priority: str  # 'high', 'medium', 'low'
    category: str  # 'conservation', 'restoration', 'development_control'
    location: str  # Descriptive location
    evidence: str  # What pattern triggered this
    action: str  # Recommended action
    metrics: Dict  # Supporting quantitative metrics


def detect_change_hotspots(change_map: np.ndarray, pred_t1: np.ndarray, 
                          pred_t2: np.ndarray, pixel_size_m: float,
                          min_cluster_size: int = 100) -> List[Hotspot]:
    """
    Detect spatial hotspots of green space change using DBSCAN clustering.
    
    Args:
        change_map: Change type map (0=no change, 1=gain, 2=loss, 3=other)
        min_cluster_size: Minimum cluster size in pixels
    
    Returns:
        List of detected hotspots
    """
    pixel_area_ha = (pixel_size_m ** 2) / 10000
    hotspots = []
    
    for change_type, change_label in [(1, 'gain'), (2, 'loss')]:
        # Get pixels of this change type
        mask = (change_map == change_type)
        if mask.sum() < min_cluster_size:
            continue
        
        # Get coordinates of changed pixels
        coords = np.argwhere(mask)
        
        # DBSCAN clustering (eps in pixel units)
        eps = 50  # 50 pixels (~500m for 10m resolution)
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(coords)
        
        # Process each cluster
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_coords = coords[cluster_mask]
            
            # Create binary mask for this cluster
            cluster_pixels = np.zeros_like(change_map, dtype=bool)
            cluster_pixels[cluster_coords[:, 0], cluster_coords[:, 1]] = True
            
            # Compute metrics
            center = tuple(cluster_coords.mean(axis=0).astype(int))
            area = cluster_pixels.sum() * pixel_area_ha
            
            # Intensity: measure how "pure" the change is in this cluster
            # (vs mixed with other change types)
            neighborhood = ndimage.binary_dilation(cluster_pixels, iterations=5)
            local_change = change_map[neighborhood]
            intensity = (local_change == change_type).sum() / neighborhood.sum()
            
            hotspots.append(Hotspot(
                type=change_label,
                center=center,
                area_ha=area,
                intensity=intensity,
                pixels=cluster_pixels,
            ))
    
    # Sort by area (largest first)
    hotspots.sort(key=lambda h: h.area_ha, reverse=True)
    
    return hotspots


def compute_fragmentation_metrics(green_mask: np.ndarray) -> Dict:
    """
    Compute landscape fragmentation metrics.
    
    Returns:
        - num_patches: Number of distinct green patches
        - mean_patch_size: Average patch size
        - largest_patch_ratio: Fraction of green space in largest patch
        - edge_density: Perimeter-to-area ratio (higher = more fragmented)
    """
    # Label connected components
    labeled, num_patches = ndimage.label(green_mask)
    
    if num_patches == 0:
        return {
            'num_patches': 0,
            'mean_patch_size': 0,
            'largest_patch_ratio': 0,
            'edge_density': 0,
        }
    
    # Patch sizes
    patch_sizes = ndimage.sum(green_mask, labeled, range(1, num_patches + 1))
    mean_patch_size = float(np.mean(patch_sizes))
    largest_patch = int(np.max(patch_sizes))
    total_green = green_mask.sum()
    largest_patch_ratio = largest_patch / total_green if total_green > 0 else 0
    
    # Edge density (approximate using Sobel filter)
    from scipy import ndimage
    edges = ndimage.sobel(green_mask.astype(float))
    edge_pixels = (edges > 0).sum()
    edge_density = edge_pixels / total_green if total_green > 0 else 0
    
    return {
        'num_patches': int(num_patches),
        'mean_patch_size': mean_patch_size,
        'largest_patch_ratio': float(largest_patch_ratio),
        'edge_density': float(edge_density),
    }


def assess_green_space_accessibility(green_mask: np.ndarray, 
                                    urban_mask: np.ndarray,
                                    pixel_size_m: float) -> Dict:
    """
    Assess accessibility of green spaces to urban areas.
    
    Computes:
    - Mean distance from urban pixels to nearest green space
    - Fraction of urban area within 300m of green space (WHO recommendation)
    """
    # Distance transform: distance of each non-green pixel to nearest green pixel
    distance_map = ndimage.distance_transform_edt(~green_mask) * pixel_size_m
    
    # Focus on urban areas
    urban_distances = distance_map[urban_mask]
    
    if len(urban_distances) == 0:
        return {
            'mean_distance_to_green_m': 0,
            'within_300m_ratio': 0,
            'within_500m_ratio': 0,
        }
    
    mean_dist = float(np.mean(urban_distances))
    within_300m = (urban_distances <= 300).sum() / len(urban_distances)
    within_500m = (urban_distances <= 500).sum() / len(urban_distances)
    
    return {
        'mean_distance_to_green_m': mean_dist,
        'within_300m_ratio': float(within_300m),
        'within_500m_ratio': float(within_500m),
    }


def generate_policy_recommendations(change_map: np.ndarray, 
                                    pred_t1: np.ndarray, 
                                    pred_t2: np.ndarray,
                                    pixel_size_m: float,
                                    city_name: str) -> List[PolicyRecommendation]:
    """
    Generate evidence-based policy recommendations from spatial analysis.
    """
    recommendations = []
    pixel_area_ha = (pixel_size_m ** 2) / 10000
    
    # Extract masks
    valid = (pred_t1 >= 0) & (pred_t2 >= 0)
    green_t1 = (pred_t1 == 3) & valid  # Assuming class 3 is green
    green_t2 = (pred_t2 == 3) & valid
    urban_t2 = (pred_t2 == 1) & valid  # Assuming class 1 is built-up
    
    # 1. Detect hotspots
    hotspots = detect_change_hotspots(change_map, pred_t1, pred_t2, pixel_size_m)
    
    # Generate recommendations for major loss hotspots
    loss_hotspots = [h for h in hotspots if h.type == 'loss'][:3]  # Top 3
    for i, hs in enumerate(loss_hotspots, 1):
        if hs.area_ha > 5:  # Only if substantial (>5 ha)
            recommendations.append(PolicyRecommendation(
                priority='high' if i == 1 else 'medium',
                category='conservation',
                location=f"Hotspot #{i} near pixel ({hs.center[0]}, {hs.center[1]})",
                evidence=f"{hs.area_ha:.1f} ha of green space lost with {hs.intensity:.0%} local concentration",
                action=f"Implement immediate conservation measures and investigate drivers of loss. "
                      f"Consider zoning restrictions to prevent further degradation.",
                metrics={
                    'area_lost_ha': hs.area_ha,
                    'intensity': hs.intensity,
                    'priority_rank': i,
                }
            ))
    
    # 2. Fragmentation analysis
    frag_t1 = compute_fragmentation_metrics(green_t1)
    frag_t2 = compute_fragmentation_metrics(green_t2)
    
    if frag_t2['num_patches'] > frag_t1['num_patches'] * 1.2:
        recommendations.append(PolicyRecommendation(
            priority='medium',
            category='conservation',
            location=f"{city_name} - citywide",
            evidence=f"Green space fragmentation increased by {(frag_t2['num_patches'] - frag_t1['num_patches']) / frag_t1['num_patches'] * 100:.0f}% "
                    f"({frag_t1['num_patches']} → {frag_t2['num_patches']} patches)",
            action="Establish ecological corridors to connect fragmented green patches. "
                  "Priority should be given to linking the largest remaining patches.",
            metrics={
                'fragmentation_increase': (frag_t2['num_patches'] - frag_t1['num_patches']) / frag_t1['num_patches'],
                'current_patches': frag_t2['num_patches'],
            }
        ))
    
    # 3. Accessibility analysis
    accessibility = assess_green_space_accessibility(green_t2, urban_t2, pixel_size_m)
    
    if accessibility['within_300m_ratio'] < 0.5:  # Less than 50% within WHO recommendation
        recommendations.append(PolicyRecommendation(
            priority='high',
            category='restoration',
            location=f"{city_name} - urban areas",
            evidence=f"Only {accessibility['within_300m_ratio']:.0%} of urban area within 300m of green space "
                    f"(WHO recommends 100%)",
            action="Prioritize creation of pocket parks and urban greening in underserved neighborhoods. "
                  f"Target areas currently >300m from nearest green space.",
            metrics={
                'accessibility_ratio': accessibility['within_300m_ratio'],
                'mean_distance_m': accessibility['mean_distance_to_green_m'],
            }
        ))
    
    # 4. Overall change trend
    green_area_t1 = green_t1.sum() * pixel_area_ha
    green_area_t2 = green_t2.sum() * pixel_area_ha
    change_pct = ((green_area_t2 - green_area_t1) / green_area_t1 * 100) if green_area_t1 > 0 else 0
    
    if change_pct < -5:  # More than 5% loss
        recommendations.append(PolicyRecommendation(
            priority='high',
            category='development_control',
            location=f"{city_name} - citywide",
            evidence=f"Overall green space declined by {abs(change_pct):.1f}% "
                    f"({green_area_t1:.0f} → {green_area_t2:.0f} ha)",
            action="Implement stricter development controls and mandatory green space quotas. "
                  "Consider urban growth boundaries to protect remaining green areas.",
            metrics={
                'total_loss_pct': abs(change_pct),
                'total_loss_ha': green_area_t1 - green_area_t2,
            }
        ))
    elif change_pct > 5:  # More than 5% gain (verify it's real!)
        gain_hotspots = [h for h in hotspots if h.type == 'gain']
        if len(gain_hotspots) > 0:
            recommendations.append(PolicyRecommendation(
                priority='medium',
                category='conservation',
                location=f"{city_name} - citywide",
                evidence=f"Green space increased by {change_pct:.1f}% "
                        f"({len(gain_hotspots)} growth hotspots detected)",
                action="Document successful greening initiatives and replicate best practices. "
                      "Ensure new green areas are properly maintained and protected.",
                metrics={
                    'total_gain_pct': change_pct,
                    'total_gain_ha': green_area_t2 - green_area_t1,
                    'num_gain_hotspots': len(gain_hotspots),
                }
            ))
    
    # 5. Identify gain hotspots for replication
    gain_hotspots = [h for h in hotspots if h.type == 'gain' and h.area_ha > 3][:2]
    for i, hs in enumerate(gain_hotspots, 1):
        recommendations.append(PolicyRecommendation(
            priority='low',
            category='conservation',
            location=f"Success zone #{i} near pixel ({hs.center[0]}, {hs.center[1]})",
            evidence=f"{hs.area_ha:.1f} ha of new green space established",
            action="Study and document greening strategies used here. "
                  "Consider as case study for replication in other areas.",
            metrics={
                'area_gained_ha': hs.area_ha,
                'intensity': hs.intensity,
            }
        ))
    
    # Sort by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations.sort(key=lambda r: priority_order[r.priority])
    
    return recommendations


def create_policy_report(recommendations: List[PolicyRecommendation], 
                        output_path) -> str:
    """
    Generate a formatted policy report in Markdown.
    """
    report = f"""# Urban Green Space Policy Recommendations

## Executive Summary

This report provides evidence-based recommendations for urban green space management 
based on satellite-derived change detection analysis.

**Total Recommendations: {len(recommendations)}**
- High Priority: {sum(1 for r in recommendations if r.priority == 'high')}
- Medium Priority: {sum(1 for r in recommendations if r.priority == 'medium')}
- Low Priority: {sum(1 for r in recommendations if r.priority == 'low')}

---

## Recommendations by Priority

"""
    
    for priority in ['high', 'medium', 'low']:
        recs_in_priority = [r for r in recommendations if r.priority == priority]
        if not recs_in_priority:
            continue
        
        report += f"\n### {priority.upper()} PRIORITY\n\n"
        
        for i, rec in enumerate(recs_in_priority, 1):
            report += f"#### {i}. {rec.category.upper()}: {rec.location}\n\n"
            report += f"**Evidence:** {rec.evidence}\n\n"
            report += f"**Recommended Action:** {rec.action}\n\n"
            
            if rec.metrics:
                report += "**Supporting Metrics:**\n"
                for key, value in rec.metrics.items():
                    if isinstance(value, float):
                        report += f"- {key}: {value:.2f}\n"
                    else:
                        report += f"- {key}: {value}\n"
            
            report += "\n---\n\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report


def run_policy_analysis(change_map: np.ndarray, 
                       pred_t1: np.ndarray, 
                       pred_t2: np.ndarray,
                       pixel_size_m: float,
                       city_name: str,
                       output_dir,
                       logger):
    """
    Complete policy analysis pipeline.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"POLICY RECOMMENDATION ANALYSIS - {city_name}")
    logger.info(f"{'='*60}")
    
    # Generate recommendations
    recommendations = generate_policy_recommendations(
        change_map, pred_t1, pred_t2, pixel_size_m, city_name
    )
    
    logger.info(f"\nGenerated {len(recommendations)} recommendations:")
    for rec in recommendations:
        logger.info(f"  [{rec.priority.upper()}] {rec.category}: {rec.location}")
        logger.info(f"    → {rec.action[:100]}...")
    
    # Create report
    report_path = output_dir / 'policy_recommendations.md'
    create_policy_report(recommendations, report_path)
    logger.info(f"\nPolicy report saved to: {report_path}")
    
    # Save recommendations as JSON
    import json
    json_path = output_dir / 'policy_recommendations.json'
    with open(json_path, 'w') as f:
        json.dump([{
            'priority': r.priority,
            'category': r.category,
            'location': r.location,
            'evidence': r.evidence,
            'action': r.action,
            'metrics': r.metrics,
        } for r in recommendations], f, indent=2)
    
    return recommendations, str(report_path)
