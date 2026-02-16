"""
GreenSight Dashboard v4 — 3-City Study (Fixed NDVI Thresholds)
Run: streamlit run dashboard.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="GreenSight", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
h1.main-title { font-size: 3rem !important; font-weight: 800 !important; color: #065f46 !important;
                margin-bottom: 0 !important; line-height: 1.1 !important; }
p.main-sub { font-size: 1.2rem !important; color: #6b7280 !important; margin-top: 4px !important;
             margin-bottom: 28px !important; font-weight: 400 !important; }
h5 { font-size: 1.1rem !important; font-weight: 700 !important; color: #1f2937 !important; }
.mc { background: #fff; border-radius: 14px; padding: 1.2rem 1rem; text-align: center;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 5px solid #10b981; }
.mc.red { border-left-color: #ef4444; } .mc.blue { border-left-color: #3b82f6; }
.mc.amber { border-left-color: #f59e0b; } .mc.green { border-left-color: #10b981; }
.mv { font-size: 2rem; font-weight: 800; color: #111827; }
.ml { font-size: 0.85rem; color: #6b7280; margin-top: 2px; }
div[data-testid="stTabs"] button p { font-size: 1rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Water", "Built-up", "Barren/Sparse", "Green Space"]
CLASS_COLORS = ["#2196F3", "#9E9E9E", "#FFC107", "#4CAF50"]

CITIES = {
    "colombo":    {"name": "Colombo",    "lat": 6.915, "lon": 79.87,  "zone": "Wet",  "pop": 752993, "area_km2": 37.3},
    "hambantota": {"name": "Hambantota", "lat": 6.15,  "lon": 81.13,  "zone": "Dry",  "pop": 12846,  "area_km2": 15.0},
    "jaffna":     {"name": "Jaffna",     "lat": 9.68,  "lon": 80.03,  "zone": "Arid", "pop": 88138,  "area_km2": 20.0},
}

@st.cache_data
def load_results():
    """Load real model results or use demo."""
    model_perf = {}
    for ck in CITIES:
        model_perf[ck] = {}
        rdir = Path(f"results/{ck}")
        if rdir.exists():
            for mdir in rdir.iterdir():
                tf = mdir / "test_results.json"
                if tf.exists():
                    with open(tf) as f:
                        d = json.load(f)
                    model_perf[ck][mdir.name] = {
                        "acc": d.get("test_accuracy", 0),
                        "f1": d.get("test_f1_weighted", 0)
                    }
    # Demo fallback
    demo = {
        "colombo":    {"resnet50": (.871,.871), "efficientnet_b0": (.818,.819), "vit_small": (.921,.921),
                       "swin_tiny": (.942,.942), "convnext_tiny": (.946,.946)},
        "hambantota": {"resnet50": (.935,.943), "efficientnet_b0": (.917,.925), "vit_small": (.979,.980),
                       "swin_tiny": (.969,.970), "convnext_tiny": (.963,.971)},
        "jaffna":     {"resnet50": (.907,.921), "efficientnet_b0": (.861,.874), "vit_small": (.969,.970),
                       "swin_tiny": (.975,.975), "convnext_tiny": (.978,.979)},
    }
    for ck in CITIES:
        if not model_perf[ck]:
            model_perf[ck] = {m: {"acc": v[0], "f1": v[1]} for m, v in demo[ck].items()}
    return model_perf

@st.cache_data
def load_temporal():
    """Load temporal analysis or use demo."""
    temporal = {}
    for ck in CITIES:
        tf = Path(f"results/{ck}/temporal_analysis.json")
        if tf.exists():
            with open(tf) as f:
                temporal[ck] = json.load(f)
    if not temporal:
        # Demo data from previous fixed-threshold runs
        temporal = {
            "colombo": {
                "2019": {"Water": 32.6, "Built-up": 18.1, "Barren": 24.1, "Green": 25.2},
                "2025": {"Water": 31.9, "Built-up": 20.3, "Barren": 25.3, "Green": 22.5},
            },
            "hambantota": {
                "2019": {"Water": 34.2, "Built-up": 1.3, "Barren": 0.9, "Green": 63.6},
                "2025": {"Water": 33.8, "Built-up": 1.8, "Barren": 1.2, "Green": 63.2},
            },
            "jaffna": {
                "2019": {"Water": 25.8, "Built-up": 1.0, "Barren": 6.4, "Green": 66.9},
                "2025": {"Water": 26.2, "Built-up": 1.5, "Barren": 7.1, "Green": 65.2},
            },
        }
    return temporal

def card(val, label, style=""):
    return f'<div class="mc {style}"><div class="mv">{val}</div><div class="ml">{label}</div></div>'

def markov_predict(s19, s25, target_year):
    """Simple Markov projection from two observed states."""
    s0 = np.array(s25) / 100
    s_prev = np.array(s19) / 100
    # Annual change rate
    annual_delta = (s0 - s_prev) / 6
    preds = {}
    s = s0.copy()
    for y in range(2026, target_year + 1):
        s = s + annual_delta
        s = np.maximum(s, 0.001)
        s = s / s.sum()
        preds[y] = s * 100
    return preds

# === TABS ===
def tab_overview(mp, temp):
    cols = st.columns(4)
    green_changes = []
    for ck in temp:
        g19 = temp[ck]["2019"]["Green"]
        g25 = temp[ck]["2025"]["Green"]
        green_changes.append(g25 - g19)
    avg_change = np.mean(green_changes)
    best_f1 = max(max(m["f1"] for m in mp[c].values()) for c in mp)

    for col, (v, l, s) in zip(cols, [
        ("3", "Cities Monitored", "blue"),
        ("72 km²", "Area Analyzed", ""),
        (f"{avg_change:+.1f}%", "Avg Green Δ", "red" if avg_change < 0 else "green"),
        (f"{best_f1:.1%}", "Best Model F1", "green"),
    ]):
        with col: st.markdown(card(v, l, s), unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 🗺️ Study Areas — Sri Lanka")
        try:
            import folium
            from streamlit_folium import st_folium
            m = folium.Map(location=[7.87, 80.77], zoom_start=7, tiles="CartoDB positron")
            clrs = {"colombo": "green", "hambantota": "red", "jaffna": "orange"}
            for ck, info in CITIES.items():
                folium.CircleMarker(
                    [info["lat"], info["lon"]], radius=12, color=clrs[ck], fill=True, fill_opacity=0.8,
                    popup=f"<b>{info['name']}</b><br>{info['zone']} Zone<br>Pop: {info['pop']:,}"
                ).add_to(m)
            st_folium(m, height=400, returned_objects=[])
        except ImportError:
            df_map = pd.DataFrame([{"City": v["name"], "lat": v["lat"], "lon": v["lon"]}
                                   for v in CITIES.values()])
            st.map(df_map, latitude="lat", longitude="lon", size=25000)

    with c2:
        st.markdown("##### Green Space Comparison (2019 vs 2025)")
        rows = []
        for ck, info in CITIES.items():
            if ck in temp:
                rows.append({"City": info["name"], "Period": "2019", "Green (%)": temp[ck]["2019"]["Green"]})
                rows.append({"City": info["name"], "Period": "2025", "Green (%)": temp[ck]["2025"]["Green"]})
        fig = px.bar(pd.DataFrame(rows), x="City", y="Green (%)", color="Period", barmode="group",
                     color_discrete_map={"2019": "#86efac", "2025": "#16a34a"}, text="Green (%)")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, margin=dict(t=10, b=50), plot_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=-0.15), yaxis_range=[0, 80])
        st.plotly_chart(fig, use_container_width=True)

def tab_models(mp):
    st.markdown("##### 🏆 F1 Score Heatmap — All Models × Cities")
    models = ["resnet50", "efficientnet_b0", "vit_small", "swin_tiny", "convnext_tiny"]
    mdisplay = ["ResNet-50", "EffNet-B0", "ViT-Small", "Swin-Tiny", "ConvNeXt-Tiny"]
    cities = list(CITIES.keys())

    z = [[mp[c].get(m, {}).get("f1", 0) for c in cities] for m in models]
    fig = go.Figure(go.Heatmap(
        z=z, x=[CITIES[c]["name"] for c in cities], y=mdisplay,
        text=[[f"{v:.4f}" for v in r] for r in z], texttemplate="%{text}",
        colorscale="Greens", zmin=0.82, zmax=1.0))
    fig.update_layout(height=280, margin=dict(t=10, b=10, l=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Detailed Results")
    rows = []
    for m, md in zip(models, mdisplay):
        mtype = "Transformer" if m in ["vit_small", "swin_tiny"] else "CNN"
        row = {"Model": md, "Type": mtype}
        f1_vals = []
        for c in cities:
            f1 = mp[c].get(m, {}).get("f1", 0)
            row[CITIES[c]["name"]] = f"{f1:.4f}"
            f1_vals.append(f1)
        row["Average"] = f"{np.mean(f1_vals):.4f}"
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Average", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("##### CNN vs Transformer")
    cnn_avg = np.mean([mp[c].get(m, {}).get("f1", 0) for c in cities
                       for m in ["resnet50", "efficientnet_b0", "convnext_tiny"]])
    trans_avg = np.mean([mp[c].get(m, {}).get("f1", 0) for c in cities
                         for m in ["vit_small", "swin_tiny"]])
    c1, c2 = st.columns(2)
    c1.metric("CNN Average F1", f"{cnn_avg:.4f}")
    c2.metric("Transformer Average F1", f"{trans_avg:.4f}")

def tab_temporal(temp):
    sel = st.selectbox("Select City", list(CITIES.keys()), format_func=lambda x: CITIES[x]["name"])
    if sel not in temp:
        st.warning("No temporal data available for this city.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Land Cover 2019 vs 2025")
        classes = ["Water", "Built-up", "Barren", "Green"]
        rows = []
        for cls in classes:
            v19 = temp[sel]["2019"].get(cls, 0)
            v25 = temp[sel]["2025"].get(cls, 0)
            rows.append({"Class": cls, "2019 (%)": v19, "2025 (%)": v25, "Change": f"{v25-v19:+.1f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with c2:
        st.markdown("##### 2025 Land Cover")
        vals = [temp[sel]["2025"].get(c, 0) for c in classes]
        fig = go.Figure(go.Pie(labels=classes, values=vals, marker_colors=CLASS_COLORS, hole=0.4,
                               textinfo="label+percent"))
        fig.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    g19 = temp[sel]["2019"]["Green"]
    g25 = temp[sel]["2025"]["Green"]
    delta = g25 - g19
    emoji = "📉" if delta < 0 else "📈"
    st.info(f"{emoji} **{CITIES[sel]['name']}**: Green space changed from {g19:.1f}% to {g25:.1f}% "
            f"({delta:+.1f} percentage points, 2019→2025)")

def tab_predict(temp):
    st.markdown("##### 🔮 Green Space Projection (Linear Markov)")
    c1, c2 = st.columns([1, 3])
    with c1:
        target = st.slider("Forecast to:", 2026, 2040, 2035)
        show_cities = st.multiselect("Cities", list(CITIES.keys()), default=list(CITIES.keys()),
                                     format_func=lambda x: CITIES[x]["name"])
    with c2:
        fig = go.Figure()
        colors = {"colombo": "#2563eb", "hambantota": "#dc2626", "jaffna": "#f59e0b"}
        for ck in show_cities:
            if ck not in temp: continue
            info = CITIES[ck]
            g19 = temp[ck]["2019"]["Green"]
            g25 = temp[ck]["2025"]["Green"]
            s19 = [temp[ck]["2019"].get(c, 0) for c in ["Water", "Built-up", "Barren", "Green"]]
            s25 = [temp[ck]["2025"].get(c, 0) for c in ["Water", "Built-up", "Barren", "Green"]]
            preds = markov_predict(s19, s25, target)

            # Observed
            fig.add_trace(go.Scatter(x=[2019, 2025], y=[g19, g25], mode="lines+markers",
                name=info["name"], line=dict(color=colors.get(ck, "#666"), width=3),
                marker=dict(size=8)))
            # Projected
            years = sorted(preds.keys())
            gvals = [preds[y][3] for y in years]
            fig.add_trace(go.Scatter(x=[2025]+years, y=[g25]+gvals, mode="lines",
                line=dict(color=colors.get(ck, "#666"), width=2, dash="dash"), showlegend=False))

        fig.add_vline(x=2025, line_dash="dot", line_color="grey", annotation_text="observed | projected")
        fig.update_layout(height=400, xaxis_title="Year", yaxis_title="Green Space (%)",
            margin=dict(t=10, b=50), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Projections assume constant annual change rates from 2019-2025 observations. "
               "Actual trajectories depend on policy, climate, and development decisions.")

def tab_risk(temp):
    st.markdown("##### ⚠️ Green Space Risk Assessment")
    risks = []
    for ck, info in CITIES.items():
        if ck not in temp: continue
        g19 = temp[ck]["2019"]["Green"]
        g25 = temp[ck]["2025"]["Green"]
        loss_pct = g19 - g25  # Positive = loss
        built = temp[ck]["2025"].get("Built-up", 0)

        # Score: higher = more at risk
        loss_score = max(loss_pct * 5, 0)  # 0-30
        coverage_score = max(50 - g25, 0) * 0.8  # Low coverage = risky
        density_score = min(info["pop"] / info["area_km2"] / 500, 1) * 25
        total = min(loss_score + coverage_score + density_score, 100)

        level = "🔴 Critical" if total >= 50 else "🟠 High" if total >= 30 else "🟡 Moderate" if total >= 15 else "🟢 Low"
        risks.append({
            "City": info["name"], "Zone": info["zone"],
            "Green 2025": f"{g25:.1f}%", "Δ Green": f"{g25-g19:+.1f}%",
            "Risk Score": f"{total:.0f}/100", "Level": level,
            "Pop Density": f"{info['pop']/info['area_km2']:,.0f}/km²"
        })
    st.dataframe(pd.DataFrame(risks), hide_index=True, use_container_width=True)

    st.markdown("##### Risk Factors")
    st.markdown("""
    - **Green Loss Rate**: Cities losing green space score higher
    - **Low Coverage**: Cities below 50% green coverage are at risk
    - **Population Density**: Higher density = more pressure on green spaces
    """)

# === MAIN ===
def main():
    mp = load_results()
    temp = load_temporal()

    with st.sidebar:
        st.markdown("### 🌿 GreenSight")
        st.markdown("Urban Green Space Intelligence")
        st.markdown("---")
        st.markdown("📅 **Period:** 2019 – 2025")
        st.markdown("🛰️ **Sensor:** Sentinel-2 L2A")
        st.markdown("📐 **Resolution:** 10m")
        st.markdown("🏙️ **Cities:** 3 climate zones")
        st.markdown("🤖 **Models:** 5 architectures")
        st.markdown("📊 **Thresholds:** Fixed NDVI")
        st.markdown("---")
        for n, c in zip(CLASS_NAMES, CLASS_COLORS):
            st.markdown(f'<span style="color:{c}; font-size:20px;">■</span> {n}', unsafe_allow_html=True)
        st.markdown("---")
        st.caption("GreenSight — AI for Sustainable Cities")

    st.markdown('<h1 class="main-title">🌿 GreenSight</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Deep Learning Urban Green Space Monitor — Sri Lanka Multi-Climate Study (2019–2025)</p>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🤖 Model Comparison", "🕐 Temporal Analysis",
                                    "🔮 Prediction", "⚠️ Risk Assessment"])
    with t1: tab_overview(mp, temp)
    with t2: tab_models(mp)
    with t3: tab_temporal(temp)
    with t4: tab_predict(temp)
    with t5: tab_risk(temp)

if __name__ == "__main__":
    main()
