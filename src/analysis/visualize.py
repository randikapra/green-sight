"""
Publication-quality visualization functions.
All figures saved at 300 DPI.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path


def plot_training_curves(histories: dict, save_dir: str):
    """
    Plot training curves for all models.
    
    Args:
        histories: dict of {model_name: history_dict}
        save_dir: output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_acc', 'val_acc', 'Accuracy'),
        ('train_f1', 'val_f1', 'F1 Score (Weighted)'),
    ]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(histories), 3)))
    
    for ax, (tk, vk, title) in zip(axes, metrics):
        for idx, (name, hist) in enumerate(histories.items()):
            epochs = range(1, len(hist[tk]) + 1)
            c = colors[idx]
            ax.plot(epochs, hist[tk], '-', color=c, alpha=0.5, label=f'{name} (train)')
            ax.plot(epochs, hist[vk], '--', color=c, linewidth=2, label=f'{name} (val)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str, save_dir: str):
    """Plot normalized confusion matrix."""
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (%) — {model_name}')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(all_results: list, save_dir: str):
    """Bar chart comparing all models on test set."""
    models = [r['model'] for r in all_results]
    accs = [r['test_accuracy'] for r in all_results]
    f1s = [r['test_f1_weighted'] for r in all_results]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#3498db')
    b2 = ax.bar(x + width/2, f1s, width, label='F1 Score', color='#e74c3c')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_land_cover_maps(predictions: dict, class_names: list, class_colors: list, save_dir: str):
    """Side-by-side land cover maps for each period."""
    labels = sorted(predictions.keys())
    n = len(labels)
    cmap = ListedColormap(class_colors)
    
    fig, axes = plt.subplots(1, n, figsize=(8*n, 8))
    if n == 1:
        axes = [axes]
    
    for ax, label in zip(axes, labels):
        masked = np.ma.masked_where(predictions[label] < 0, predictions[label])
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=len(class_names)-1)
        ax.set_title(f'Land Cover — {label}', fontsize=14)
        ax.axis('off')
    
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(class_colors, class_names)]
    fig.legend(handles=patches, loc='lower center', ncol=len(class_names), fontsize=11,
               bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Urban Land Cover Classification — Colombo, Sri Lanka', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(Path(save_dir) / 'land_cover_maps.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_change_map(change_map: np.ndarray, save_dir: str):
    """Green space change detection map."""
    colors = ['#E0E0E0', '#4CAF50', '#F44336', '#FF9800']
    labels = ['No Change', 'Green Gained', 'Green Lost', 'Other Change']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    masked = np.ma.masked_where(change_map < 0, change_map)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=3)
    ax.set_title('Green Space Change Detection (2018 → 2024)', fontsize=14)
    ax.axis('off')
    
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'change_detection_map.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_transition_matrix(matrix: np.ndarray, class_names: list, save_dir: str):
    """Transition matrix heatmap."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    pct = matrix / (row_sums + 1e-6) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('To (2024)')
    ax.set_ylabel('From (2018)')
    ax.set_title('Land Cover Transition Matrix (%)')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'transition_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_area_comparison(stats: dict, class_names: list, class_colors: list, save_dir: str):
    """Bar chart comparing areas across time periods."""
    periods = sorted(stats.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(periods)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, period in enumerate(periods):
        areas = [stats[period].get(c, {}).get('area_hectares', 0) for c in class_names]
        offset = (i - len(periods)/2 + 0.5) * width
        bars = ax.bar(x + offset, areas, width, label=period, 
                      color=class_colors, alpha=0.5 + 0.5*i/max(len(periods)-1, 1),
                      edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Area (Hectares)')
    ax.set_title('Land Cover Area Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'area_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
