"""
Interactive Dashboard for GreenSight Results

Streamlit-based dashboard for exploring temporal analysis results with:
- Quality assessment visualization
- Temporal change maps
- Uncertainty visualization
- Policy recommendations
- Cross-city comparisons

Run with: streamlit run dashboard_enhanced.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List


# Page config
st.set_page_config(
    page_title="GreenSight Analysis Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_city_results(results_dir: Path, city_key: str) -> Dict:
    """Load all results for a city."""
    city_dir = results_dir / city_key / 'temporal_analysis'
    
    data = {
        'city_key': city_key,
        'temporal_results': None,
        'quality_report': None,
        'policy_recommendations': None,
        'prediction_maps': None,
    }
    
    # Load JSON files
    if (city_dir / 'temporal_results.json').exists():
        with open(city_dir / 'temporal_results.json') as f:
            data['temporal_results'] = json.load(f)
    
    if (city_dir / 'quality_report.json').exists():
        with open(city_dir / 'quality_report.json') as f:
            data['quality_report'] = json.load(f)
    
    if (city_dir / 'policy_recommendations.json').exists():
        with open(city_dir / 'policy_recommendations.json') as f:
            data['policy_recommendations'] = json.load(f)
    
    # Load numpy arrays
    if (city_dir / 'prediction_maps.npz').exists():
        data['prediction_maps'] = np.load(city_dir / 'prediction_maps.npz')
    
    return data


def plot_quality_metrics(quality_report: Dict):
    """Visualize data quality metrics."""
    periods = [k for k in quality_report.keys() if k != 'temporal_consistency']
    
    if not periods:
        st.warning("No quality data available")
        return
    
    metrics_data = []
    for period in periods:
        metrics = quality_report[period]
        metrics_data.append({
            'Period': period,
            'Quality Score': metrics['quality_score'],
            'Valid Pixels': metrics['valid_pixel_ratio'] * 100,
            'Cloud/Shadow': metrics['cloud_shadow_estimate'] * 100,
            'Saturation': metrics['saturation_ratio'] * 100,
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Quality score gauge
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quality Scores")
        for _, row in df.iterrows():
            score = row['Quality Score']
            color = 'green' if score >= 80 else 'orange' if score >= 70 else 'red'
            st.metric(
                label=f"{row['Period']} Quality",
                value=f"{score:.1f}/100",
                delta=None,
                delta_color="off"
            )
    
    with col2:
        st.subheader("Coverage Metrics")
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Valid Pixels',
            x=df['Period'],
            y=df['Valid Pixels'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Cloud/Shadow',
            x=df['Period'],
            y=df['Cloud/Shadow'],
            marker_color='gray'
        ))
        
        fig.update_layout(
            barmode='group',
            yaxis_title='Percentage (%)',
            height=300,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show warnings
    if any(quality_report[p].get('warnings') for p in periods):
        st.subheader("⚠️ Quality Warnings")
        for period in periods:
            warnings = quality_report[period].get('warnings', [])
            if warnings:
                st.warning(f"**{period}:**")
                for w in warnings:
                    st.write(f"  - {w}")


def plot_temporal_change(temporal_results: Dict, prediction_maps):
    """Visualize temporal change with uncertainty."""
    stats = temporal_results.get('stats', {})
    
    if not stats:
        st.warning("No temporal statistics available")
        return
    
    # Extract periods and green space data
    periods = sorted(stats.keys())
    
    if len(periods) < 2:
        st.warning("Need at least 2 time periods for change analysis")
        return
    
    # Green space area over time
    st.subheader("🌳 Green Space Temporal Trend")
    
    green_data = []
    for period in periods:
        gs = stats[period].get('Green Space', {})
        green_data.append({
            'Period': period,
            'Area (ha)': gs.get('area_hectares', 0),
            'CI Lower': gs.get('area_ci_lower', 0),
            'CI Upper': gs.get('area_ci_upper', 0),
        })
    
    df = pd.DataFrame(green_data)
    
    fig = go.Figure()
    
    # Area with confidence interval
    fig.add_trace(go.Scatter(
        x=df['Period'],
        y=df['Area (ha)'],
        mode='lines+markers',
        name='Green Space Area',
        line=dict(color='green', width=3),
        marker=dict(size=10),
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=df['Period'].tolist() + df['Period'].tolist()[::-1],
        y=df['CI Upper'].tolist() + df['CI Lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
    ))
    
    fig.update_layout(
        xaxis_title='Time Period',
        yaxis_title='Green Space Area (hectares)',
        hovermode='x unified',
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical significance
    sig_test = temporal_results.get('statistical_test')
    if sig_test:
        st.subheader("📊 Statistical Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            change_pct = temporal_results.get('green_space_change_pct', 0)
            st.metric(
                "Total Change",
                f"{change_pct:+.1f}%",
                delta=None
            )
        
        with col2:
            p_val = sig_test.get('p_value', 1.0)
            is_sig = sig_test.get('is_significant', False)
            st.metric(
                "P-value",
                f"{p_val:.4f}",
                delta="Significant" if is_sig else "Not significant",
                delta_color="normal" if is_sig else "off"
            )
        
        with col3:
            effect = sig_test.get('effect_size', 0)
            st.metric(
                "Effect Size",
                f"{effect:.2f}",
                delta=sig_test.get('interpretation', ''),
                delta_color="off"
            )
        
        with col4:
            ci_overlap = sig_test.get('confidence_intervals_overlap', False)
            st.metric(
                "CI Overlap",
                "Yes" if ci_overlap else "No",
                delta=None
            )


def plot_land_cover_distribution(temporal_results: Dict):
    """Plot land cover class distribution."""
    stats = temporal_results.get('stats', {})
    
    if not stats:
        return
    
    st.subheader("📈 Land Cover Distribution")
    
    periods = sorted(stats.keys())
    
    # Prepare data for stacked bar chart
    classes = list(stats[periods[0]].keys())
    
    fig = go.Figure()
    
    for cls in classes:
        areas = [stats[p][cls]['area_hectares'] for p in periods]
        fig.add_trace(go.Bar(
            name=cls,
            x=periods,
            y=areas,
        ))
    
    fig.update_layout(
        barmode='stack',
        yaxis_title='Area (hectares)',
        xaxis_title='Time Period',
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_policy_recommendations(recommendations: List[Dict]):
    """Display policy recommendations in organized format."""
    if not recommendations:
        st.info("No policy recommendations available")
        return
    
    st.subheader("📋 Policy Recommendations")
    
    # Group by priority
    priorities = {'high': [], 'medium': [], 'low': []}
    for rec in recommendations:
        priorities[rec['priority']].append(rec)
    
    # Display by priority
    for priority, color in [('high', '🔴'), ('medium', '🟡'), ('low', '🟢')]:
        recs = priorities[priority]
        if not recs:
            continue
        
        st.markdown(f"### {color} {priority.upper()} Priority ({len(recs)} recommendations)")
        
        for i, rec in enumerate(recs, 1):
            with st.expander(f"{i}. {rec['category'].upper()}: {rec['location']}"):
                st.markdown(f"**Evidence:** {rec['evidence']}")
                st.markdown(f"**Recommended Action:** {rec['action']}")
                
                if rec.get('metrics'):
                    st.markdown("**Supporting Metrics:**")
                    cols = st.columns(len(rec['metrics']))
                    for j, (key, value) in enumerate(rec['metrics'].items()):
                        with cols[j]:
                            if isinstance(value, float):
                                st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                            else:
                                st.metric(key.replace('_', ' ').title(), str(value))


def plot_uncertainty_map(prediction_maps, period: str):
    """Visualize uncertainty for a specific period."""
    unc_key = f'unc_{period}'
    
    if unc_key not in prediction_maps:
        st.warning(f"No uncertainty data for {period}")
        return
    
    unc_map = prediction_maps[unc_key]
    
    fig = px.imshow(
        unc_map,
        color_continuous_scale='YlOrRd',
        labels={'color': 'Uncertainty (entropy)'},
        title=f'Prediction Uncertainty - {period}'
    )
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    valid_unc = unc_map[unc_map > 0]
    if len(valid_unc) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Uncertainty", f"{valid_unc.mean():.3f}")
        with col2:
            st.metric("Max Uncertainty", f"{valid_unc.max():.3f}")
        with col3:
            high_unc_pct = (valid_unc > 0.5).sum() / len(valid_unc) * 100
            st.metric("High Uncertainty %", f"{high_unc_pct:.1f}%")


def main():
    st.title("🌿 GreenSight Analysis Dashboard")
    st.markdown("Interactive exploration of urban green space change detection results")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    results_dir = st.sidebar.text_input(
        "Results Directory",
        value="./results",
        help="Path to results directory"
    )
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        st.error(f"Results directory not found: {results_dir}")
        st.info("Run the analysis pipeline first: `python scripts/analyze_enhanced.py`")
        return
    
    # Find available cities
    cities = [d.name for d in results_path.iterdir() 
              if d.is_dir() and (d / 'temporal_analysis').exists()]
    
    if not cities:
        st.warning("No analysis results found!")
        st.info("Run the analysis pipeline first: `python scripts/analyze_enhanced.py`")
        return
    
    selected_city = st.sidebar.selectbox(
        "Select City",
        cities,
        index=0
    )
    
    # Load data
    data = load_city_results(results_path, selected_city)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Data Quality",
        "📈 Temporal Change",
        "🗺️ Uncertainty Maps",
        "📋 Policy Recommendations"
    ])
    
    with tab1:
        st.header(f"Overview - {selected_city.title()}")
        
        if data['temporal_results']:
            results = data['temporal_results']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Analysis Summary")
                st.write(f"**City:** {results['city']}")
                st.write(f"**Model Used:** {results['model_used']}")
                
                change = results.get('green_space_change_pct')
                if change is not None:
                    color = "green" if change > 0 else "red"
                    st.markdown(f"**Green Space Change:** <span style='color:{color};font-size:24px;'>{change:+.1f}%</span>", 
                              unsafe_allow_html=True)
            
            with col2:
                plot_land_cover_distribution(results)
        
        else:
            st.warning("No temporal results available")
    
    with tab2:
        st.header("Data Quality Assessment")
        
        if data['quality_report']:
            plot_quality_metrics(data['quality_report'])
        else:
            st.warning("No quality report available")
    
    with tab3:
        st.header("Temporal Change Analysis")
        
        if data['temporal_results'] and data['prediction_maps']:
            plot_temporal_change(data['temporal_results'], data['prediction_maps'])
        else:
            st.warning("Temporal analysis data not available")
    
    with tab4:
        st.header("Uncertainty Visualization")
        
        if data['prediction_maps']:
            # Get available periods
            periods = sorted(set(k.replace('unc_', '') for k in data['prediction_maps'].keys() 
                               if k.startswith('unc_')))
            
            if periods:
                selected_period = st.selectbox("Select Period", periods)
                plot_uncertainty_map(data['prediction_maps'], selected_period)
            else:
                st.warning("No uncertainty maps available")
        else:
            st.warning("Prediction maps not available")
    
    with tab5:
        st.header("Policy Recommendations")
        
        if data['policy_recommendations']:
            display_policy_recommendations(data['policy_recommendations'])
        else:
            st.warning("No policy recommendations available")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "GreenSight: Deep Learning-Based Urban Green Space Monitoring\n\n"
        "Uncertainty-aware temporal analysis with policy recommendations"
    )


if __name__ == "__main__":
    main()
