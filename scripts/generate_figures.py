"""
Generate Publication Figures for GreenSight Paper
==================================================
Generates:
1. Confusion matrices (per city, best model)
2. Per-class F1 bar chart (all models, all cities)
3. Model comparison radar chart
4. Training curves (loss + F1)

Usage: python scripts/generate_figures.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Config
CLASS_NAMES = ["Water", "Built-up", "Barren/\nSparse", "Green\nSpace"]
CLASS_NAMES_SHORT = ["Water", "Built-up", "Barren", "Green"]
CLASS_COLORS = ["#2196F3", "#9E9E9E", "#FFC107", "#4CAF50"]

CITIES = {
    "colombo": "Colombo",
    "badulla": "Badulla", 
    "hambantota": "Hambantota",
    "jaffna": "Jaffna",
}

MODELS = ["resnet50", "vit_small", "efficientnet_b0", "swin_tiny", "convnext_tiny"]
MODEL_DISPLAY = ["ResNet-50", "ViT-S", "EffNet-B0", "Swin-T", "ConvNeXt-T"]

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load all test results."""
    results = {}
    for city_key in CITIES:
        results[city_key] = {}
        for model in MODELS:
            path = RESULTS_DIR / city_key / model / "test_results.json"
            if path.exists():
                with open(path) as f:
                    results[city_key][model] = json.load(f)
    return results


def plot_confusion_matrices(results):
    """Plot confusion matrices for best model per city."""
    fig, axes = plt.subplots(1, len(CITIES), figsize=(4 * len(CITIES), 3.5))
    if len(CITIES) == 1:
        axes = [axes]
    
    for idx, (city_key, city_name) in enumerate(CITIES.items()):
        # Find best model
        best_model = None
        best_f1 = 0
        for model in MODELS:
            if model in results.get(city_key, {}):
                f1 = results[city_key][model].get("test_f1_weighted", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
        
        if best_model and "confusion_matrix" in results[city_key][best_model]:
            cm = np.array(results[city_key][best_model]["confusion_matrix"])
            
            # Normalize
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
            
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                       xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                       ax=axes[idx], vmin=0, vmax=1,
                       cbar=idx == len(CITIES) - 1)
            
            model_disp = MODEL_DISPLAY[MODELS.index(best_model)]
            axes[idx].set_title(f"{city_name}\n({model_disp}, F1={best_f1:.3f})", fontsize=10)
            axes[idx].set_ylabel("True" if idx == 0 else "")
            axes[idx].set_xlabel("Predicted")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "confusion_matrices.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices saved")


def plot_model_comparison(results):
    """Plot model F1 scores across all cities."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x = np.arange(len(MODELS))
    width = 0.18
    city_colors = ["#2563eb", "#10b981", "#f59e0b", "#dc2626"]
    
    for i, (city_key, city_name) in enumerate(CITIES.items()):
        f1_scores = []
        for model in MODELS:
            if model in results.get(city_key, {}):
                f1_scores.append(results[city_key][model].get("test_f1_weighted", 0))
            else:
                f1_scores.append(0)
        
        offset = (i - len(CITIES)/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=city_name, color=city_colors[i], alpha=0.85)
        
        # Add value labels
        for bar, val in zip(bars, f1_scores):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)
    
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_DISPLAY, fontsize=9)
    ax.set_ylabel("Weighted F1 Score")
    ax.set_ylim(0.82, 1.02)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "model_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Model comparison saved")


def plot_per_class_f1(results):
    """Plot per-class F1 for best model per city."""
    fig, axes = plt.subplots(1, len(CITIES), figsize=(3.5 * len(CITIES), 3))
    if len(CITIES) == 1:
        axes = [axes]
    
    for idx, (city_key, city_name) in enumerate(CITIES.items()):
        # Find best model
        best_model = max(
            (m for m in MODELS if m in results.get(city_key, {})),
            key=lambda m: results[city_key][m].get("test_f1_weighted", 0),
            default=None
        )
        
        if best_model:
            report = results[city_key][best_model].get("classification_report", {})
            class_f1 = []
            for cls in ["Water", "Built-up", "Barren/Sparse", "Green Space"]:
                if cls in report:
                    class_f1.append(report[cls].get("f1-score", 0))
                else:
                    class_f1.append(0)
            
            bars = axes[idx].bar(range(4), class_f1, color=CLASS_COLORS, alpha=0.85, edgecolor='white')
            axes[idx].set_xticks(range(4))
            axes[idx].set_xticklabels(CLASS_NAMES_SHORT, fontsize=8)
            axes[idx].set_ylim(0, 1.1)
            axes[idx].set_title(city_name, fontsize=10)
            axes[idx].set_ylabel("F1 Score" if idx == 0 else "")
            
            for bar, val in zip(bars, class_f1):
                axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                             f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_class_f1.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "per_class_f1.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Per-class F1 saved")


def plot_class_distributions(results=None):
    """Plot class distribution for each city."""
    fig, axes = plt.subplots(1, len(CITIES), figsize=(3 * len(CITIES), 3))
    if len(CITIES) == 1:
        axes = [axes]
    
    for idx, (city_key, city_name) in enumerate(CITIES.items()):
        data_path = Path(f"data/processed/{city_key}/dataset.npz")
        if data_path.exists():
            data = np.load(str(data_path))
            y = np.concatenate([data['y_train'], data['y_val'], data['y_test']])
            counts = [np.sum(y == c) for c in range(4)]
            pcts = [c / len(y) * 100 for c in counts]
            
            bars = axes[idx].bar(range(4), pcts, color=CLASS_COLORS, alpha=0.85, edgecolor='white')
            axes[idx].set_xticks(range(4))
            axes[idx].set_xticklabels(CLASS_NAMES_SHORT, fontsize=8, rotation=30)
            axes[idx].set_title(city_name, fontsize=10)
            axes[idx].set_ylabel("Percentage (%)" if idx == 0 else "")
            
            for bar, pct, cnt in zip(bars, pcts, counts):
                axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                             f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "class_distributions.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Class distributions saved")


def main():
    print("Generating publication figures...")
    results = load_results()
    
    plot_confusion_matrices(results)
    plot_model_comparison(results)
    plot_per_class_f1(results)
    plot_class_distributions(results)
    
    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Files: confusion_matrices.pdf, model_comparison.pdf, per_class_f1.pdf, class_distributions.pdf")


if __name__ == "__main__":
    main()
