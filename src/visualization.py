"""
Visualization module for plotting results and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import os
from sklearn.metrics import roc_curve, roc_auc_score, auc


# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def plot_trust_distribution(
    clients: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of trust scores across clients.
    
    Args:
        clients: List of client information dictionaries
        save_path: Optional path to save the figure
    """
    trust_scores = [c.get('trust_score', 0) for c in clients if c.get('trust_score') is not None]
    client_ids = [c.get('client_id', f'Client_{i}') for i, c in enumerate(clients) if c.get('trust_score') is not None]
    
    if not trust_scores:
        print("No trust scores available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    ax1.bar(range(len(trust_scores)), trust_scores, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Client Index', fontsize=12)
    ax1.set_ylabel('Trust Score', fontsize=12)
    ax1.set_title('Trust Scores by Client', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(client_ids)))
    ax1.set_xticklabels(client_ids, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Histogram
    ax2.hist(trust_scores, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Trust Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Trust Scores', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(np.mean(trust_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(trust_scores):.3f}')
    ax2.axvline(np.median(trust_scores), color='green', linestyle='--', 
                label=f'Median: {np.median(trust_scores):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trust distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1_score', 'precision', 'recall'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot performance comparison across different approaches.
    
    Args:
        results: Dictionary mapping approach names to their metrics
        metrics: List of metrics to plot
        save_path: Optional path to save the figure
    """
    approaches = list(results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        values = [results[approach].get(metric, 0) for approach in approaches]
        
        bars = axes[idx].bar(approaches, values, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', 
                           fontsize=14, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}',
                          ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrices for all approaches.
    
    Args:
        results: Dictionary mapping approach names to their results (including confusion_matrix)
        save_path: Optional path to save the figure
    """
    n_approaches = len(results)
    fig, axes = plt.subplots(1, n_approaches, figsize=(5*n_approaches, 4))
    if n_approaches == 1:
        axes = [axes]
    
    for idx, (approach, result) in enumerate(results.items()):
        cm = result.get('confusion_matrix', [[0, 0], [0, 0]])
        cm_array = np.array(cm)
        
        # Handle different confusion matrix shapes
        if cm_array.shape == (1, 1):
            # Single class (all predictions are the same)
            labels = ['Attack']  # Assuming attack class
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar_kws={'label': 'Count'},
                       xticklabels=labels, yticklabels=labels)
        elif cm_array.shape == (2, 2):
            # Binary classification
            labels = ['Benign', 'Attack']
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar_kws={'label': 'Count'},
                       xticklabels=labels, yticklabels=labels)
        else:
            # Multi-class or unexpected shape
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar_kws={'label': 'Count'})
        
        axes[idx].set_xlabel('Predicted', fontsize=12)
        axes[idx].set_ylabel('Actual', fontsize=12)
        axes[idx].set_title(f'{approach.replace("_", " ").title()}\nConfusion Matrix', 
                           fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrices plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    y_true: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "ROC Curves"
) -> None:
    """
    Plot ROC curves for all approaches.
    
    Args:
        results: Dictionary mapping approach names to their results. Each result should contain:
            - 'y_true': True labels (if not provided separately)
            - 'y_pred_proba': Predicted probabilities for positive class (required)
            OR 'y_pred': Predicted labels (will compute probabilities if needed)
        y_true: True labels (optional, can be in results)
        save_path: Optional path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = sns.color_palette("husl", len(results))
    
    for idx, (approach, result) in enumerate(results.items()):
        # Get true labels
        if y_true is not None:
            true_labels = y_true
        elif 'y_true' in result:
            true_labels = result['y_true']
        else:
            print(f"Warning: No true labels found for {approach}, skipping ROC curve")
            continue
        
        # Get predicted probabilities
        if 'y_pred_proba' in result:
            y_proba = result['y_pred_proba']
        elif 'y_pred' in result:
            # Convert predictions to probabilities (simple approach)
            # For binary classification, use prediction as probability
            y_pred = result['y_pred']
            y_proba = np.asarray(y_pred).astype(float)  # This is a simplification
            print(f"Warning: Using predictions as probabilities for {approach} (not ideal)")
        else:
            print(f"Warning: No predictions found for {approach}, skipping ROC curve")
            continue
        
        # Compute ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(true_labels, y_proba)
            roc_auc = roc_auc_score(true_labels, y_proba)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=colors[idx], lw=2, 
                   label=f'{approach.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"Error computing ROC curve for {approach}: {e}")
            continue
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
           label='Random Classifier (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trust_vs_performance(
    clients: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot trust scores vs validation performance.
    
    Args:
        clients: List of client information dictionaries
        save_path: Optional path to save the figure
    """
    trust_scores = []
    val_accuracies = []
    val_f1_scores = []
    client_ids = []
    
    for client in clients:
        if client.get('trust_score') is not None:
            trust_scores.append(client['trust_score'])
            val_accuracies.append(client.get('val_accuracy', 0))
            val_f1_scores.append(client.get('val_f1', 0))
            client_ids.append(client.get('client_id', 'Unknown'))
    
    if not trust_scores:
        print("No data available for trust vs performance plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trust vs Accuracy
    ax1.scatter(trust_scores, val_accuracies, s=100, alpha=0.6, color='steelblue')
    for i, client_id in enumerate(client_ids):
        ax1.annotate(client_id, (trust_scores[i], val_accuracies[i]), 
                    fontsize=8, alpha=0.7)
    ax1.set_xlabel('Trust Score', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Trust Score vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='y=x')
    ax1.legend()
    
    # Trust vs F1
    ax2.scatter(trust_scores, val_f1_scores, s=100, alpha=0.6, color='coral')
    for i, client_id in enumerate(client_ids):
        ax2.annotate(client_id, (trust_scores[i], val_f1_scores[i]), 
                    fontsize=8, alpha=0.7)
    ax2.set_xlabel('Trust Score', fontsize=12)
    ax2.set_ylabel('Validation F1-Score', fontsize=12)
    ax2.set_title('Trust Score vs Validation F1-Score', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='y=x')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trust vs performance plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_radar(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1_score', 'precision', 'recall'],
    save_path: Optional[str] = None
) -> None:
    """
    Create a radar chart comparing metrics across approaches.
    
    Args:
        results: Dictionary mapping approach names to their metrics
        metrics: List of metrics to include
        save_path: Optional path to save the figure
    """
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot each approach
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for idx, (approach, result) in enumerate(results.items()):
        values = [result.get(metric, 0) for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=approach.replace('_', ' ').title(), 
               color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Performance Metrics Comparison (Radar Chart)', 
             fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved radar chart to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trust_evolution(
    clients: List[Dict[str, Any]],
    trust_manager: Optional[Any] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot trust evolution over rounds (time series).
    
    Args:
        clients: List of client information dictionaries
        trust_manager: Optional TrustManager instance with trust histories
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect trust evolution data
    client_evolutions = {}
    
    # From client performance history
    for client in clients:
        client_id = client.get('client_id', 'unknown')
        perf_history = client.get('performance_history', [])
        
        if perf_history and len(perf_history) > 1:
            rounds = [p.get('round', i+1) for i, p in enumerate(perf_history)]
            trust_scores = [p.get('trust_score', 0) for p in perf_history]
            client_evolutions[client_id] = {
                'rounds': rounds,
                'trust_scores': trust_scores
            }
    
    # From TrustManager if available
    if trust_manager and hasattr(trust_manager, 'trust_histories'):
        for client_id, history in trust_manager.trust_histories.items():
            if len(history.trust_scores) > 1:
                # Skip initial round 0
                rounds = history.round_numbers[1:] if len(history.round_numbers) > 1 else list(range(1, len(history.trust_scores)))
                trust_scores = history.trust_scores[1:] if len(history.trust_scores) > 1 else history.trust_scores
                client_evolutions[client_id] = {
                    'rounds': rounds,
                    'trust_scores': trust_scores
                }
    
    if not client_evolutions:
        print("No trust evolution data available for plotting")
        return
    
    # Plot each client's trust evolution
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_evolutions)))
    for idx, (client_id, data) in enumerate(client_evolutions.items()):
        ax.plot(data['rounds'], data['trust_scores'], 
               marker='o', linewidth=2, markersize=6,
               label=client_id, color=colors[idx], alpha=0.7)
    
    ax.set_xlabel('Round Number', fontsize=12)
    ax.set_ylabel('Trust Score', fontsize=12)
    ax.set_title('Trust Evolution Over Rounds', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trust evolution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trust_trends(
    clients: List[Dict[str, Any]],
    trust_manager: Optional[Any] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot trust trends showing how trust changes for each client over time.
    Includes trend indicators (improving/declining/stable).
    
    Args:
        clients: List of client information dictionaries
        trust_manager: Optional TrustManager instance with trust histories
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Collect data
    client_data = {}
    
    # From client performance history
    for client in clients:
        client_id = client.get('client_id', 'unknown')
        perf_history = client.get('performance_history', [])
        trend = client.get('trend', 'stable')
        
        if perf_history and len(perf_history) > 1:
            rounds = [p.get('round', i+1) for i, p in enumerate(perf_history)]
            trust_scores = [p.get('trust_score', 0) for p in perf_history]
            client_data[client_id] = {
                'rounds': rounds,
                'trust_scores': trust_scores,
                'trend': trend,
                'initial': trust_scores[0],
                'final': trust_scores[-1],
                'change': trust_scores[-1] - trust_scores[0]
            }
    
    # From TrustManager if available
    if trust_manager and hasattr(trust_manager, 'trust_histories'):
        for client_id, history in trust_manager.trust_histories.items():
            if len(history.trust_scores) > 1:
                rounds = history.round_numbers[1:] if len(history.round_numbers) > 1 else list(range(1, len(history.trust_scores)))
                trust_scores = history.trust_scores[1:] if len(history.trust_scores) > 1 else history.trust_scores
                trend = history.get_trend()
                
                client_data[client_id] = {
                    'rounds': rounds,
                    'trust_scores': trust_scores,
                    'trend': trend,
                    'initial': trust_scores[0],
                    'final': trust_scores[-1],
                    'change': trust_scores[-1] - trust_scores[0]
                }
    
    if not client_data:
        print("No trust trend data available for plotting")
        return
    
    # Plot 1: Trust evolution with trend colors
    colors_map = {'improving': '#2ecc71', 'declining': '#e74c3c', 'stable': '#95a5a6'}
    for client_id, data in client_data.items():
        color = colors_map.get(data['trend'], '#3498db')
        ax1.plot(data['rounds'], data['trust_scores'], 
                marker='o', linewidth=2, markersize=6,
                label=f"{client_id} ({data['trend']})", 
                color=color, alpha=0.7)
    
    ax1.set_xlabel('Round Number', fontsize=12)
    ax1.set_ylabel('Trust Score', fontsize=12)
    ax1.set_title('Trust Evolution with Trends', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Trust change summary
    client_ids = list(client_data.keys())
    changes = [client_data[cid]['change'] for cid in client_ids]
    trends = [client_data[cid]['trend'] for cid in client_ids]
    
    # Color bars by trend
    bar_colors = [colors_map.get(trend, '#3498db') for trend in trends]
    bars = ax2.barh(client_ids, changes, color=bar_colors, alpha=0.7)
    
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Trust Change (Final - Initial)', fontsize=12)
    ax2.set_ylabel('Client', fontsize=12)
    ax2.set_title('Trust Change Summary by Client', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, change) in enumerate(zip(bars, changes)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{change:+.3f}',
                ha='left' if width >= 0 else 'right',
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trust trends plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trust_explanation(
    clients: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Create a beginner-friendly visualization explaining trust scores.
    
    This plot is designed for educational purposes, showing:
    - What trust scores mean
    - How they relate to client quality
    - Visual categorization of clients
    
    Args:
        clients: List of client information dictionaries
        save_path: Optional path to save the figure
    """
    # Extract data
    client_data = []
    for c in clients:
        if c.get('trust_score') is not None:
            client_data.append({
                'id': c.get('client_id', 'Unknown'),
                'trust': c.get('trust_score', 0),
                'quality': c.get('client_quality', 'unknown')
            })
    
    if not client_data:
        print("No trust scores available for explanation plot")
        return
    
    df = pd.DataFrame(client_data)
    
    # Categorize trust levels
    def categorize_trust(trust):
        if trust >= 0.9:
            return 'Very High\n(Excellent)'
        elif trust >= 0.7:
            return 'High\n(Good)'
        elif trust >= 0.5:
            return 'Medium\n(Fair)'
        elif trust >= 0.3:
            return 'Low\n(Poor)'
        else:
            return 'Very Low\n(Untrustworthy)'
    
    df['trust_category'] = df['trust'].apply(categorize_trust)
    
    # Create figure with explanation
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Understanding Trust Scores in Federated Learning', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Trust scores with color coding
    ax1 = fig.add_subplot(gs[0, :])
    colors = []
    for trust in df['trust']:
        if trust >= 0.9:
            colors.append('#2ecc71')  # Green
        elif trust >= 0.7:
            colors.append('#3498db')  # Blue
        elif trust >= 0.5:
            colors.append('#f39c12')  # Orange
        elif trust >= 0.3:
            colors.append('#e67e22')  # Dark orange
        else:
            colors.append('#e74c3c')  # Red
    
    bars = ax1.barh(range(len(df)), df['trust'], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['id'], fontsize=9)
    ax1.set_xlabel('Trust Score (0 = Untrustworthy, 1 = Fully Trusted)', fontsize=12, fontweight='bold')
    ax1.set_title('Trust Scores by Client', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1.1])
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(0.9, color='green', linestyle='--', alpha=0.5, label='Very High (≥0.9)')
    ax1.axvline(0.7, color='blue', linestyle='--', alpha=0.5, label='High (≥0.7)')
    ax1.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (≥0.5)')
    ax1.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='Low (<0.3)')
    ax1.legend(loc='lower right', fontsize=9)
    
    # Add trust values on bars
    for i, (bar, trust) in enumerate(zip(bars, df['trust'])):
        ax1.text(trust + 0.02, i, f'{trust:.3f}', 
                va='center', fontsize=8, fontweight='bold')
    
    # Plot 2: Trust distribution by category
    ax2 = fig.add_subplot(gs[1, 0])
    category_counts = df['trust_category'].value_counts()
    category_order = ['Very High\n(Excellent)', 'High\n(Good)', 'Medium\n(Fair)', 
                     'Low\n(Poor)', 'Very Low\n(Untrustworthy)']
    category_counts = category_counts.reindex([c for c in category_order if c in category_counts.index])
    
    colors_map = {
        'Very High\n(Excellent)': '#2ecc71',
        'High\n(Good)': '#3498db',
        'Medium\n(Fair)': '#f39c12',
        'Low\n(Poor)': '#e67e22',
        'Very Low\n(Untrustworthy)': '#e74c3c'
    }
    bar_colors = [colors_map.get(cat, '#95a5a6') for cat in category_counts.index]
    
    bars2 = ax2.bar(range(len(category_counts)), category_counts.values, 
                    color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(category_counts)))
    ax2.set_xticklabels(category_counts.index, fontsize=9, rotation=15, ha='right')
    ax2.set_ylabel('Number of Clients', fontsize=11)
    ax2.set_title('Client Distribution by Trust Level', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add counts on bars
    for bar, count in zip(bars2, category_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Trust statistics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats_text = f"""
    Trust Score Statistics
    
    Mean:     {df['trust'].mean():.3f}
    Median:   {df['trust'].median():.3f}
    Std Dev:  {df['trust'].std():.3f}
    Min:      {df['trust'].min():.3f}
    Max:      {df['trust'].max():.3f}
    Range:    {df['trust'].max() - df['trust'].min():.3f}
    
    Total Clients: {len(df)}
    
    Interpretation:
    • Higher trust = More reliable client
    • Trust = Validation accuracy
    • Clients with high trust contribute
      more to the global model
    • Low trust clients are down-weighted
      to prevent poisoning
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    # Plot 4: Explanation text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    explanation = """
    What is Trust in Federated Learning?
    
    Trust scores measure how reliable each federated client is. They are calculated as the validation accuracy
    of each client's locally trained model. Higher trust means the client's data and model are more reliable.
    
    How Trust Affects Aggregation:
    • High Trust (≥0.7): These clients contribute significantly to the global model. Their data and model
      updates are weighted heavily during aggregation.
    • Medium Trust (0.5-0.7): These clients contribute moderately. They are included but with reduced weight.
    • Low Trust (<0.5): These clients are down-weighted or excluded. They may have noisy data, be poorly
      configured, or be compromised sites.
    
    Why Trust Matters:
    In federated learning, not all clients are equally reliable. Some clients may produce noisy data,
    be misconfigured, or even be compromised. Trust-aware aggregation ensures that reliable clients
    have more influence on the global intrusion detection model, improving overall accuracy and robustness.
    """
    
    ax4.text(0.05, 0.5, explanation, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trust explanation plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_all_visualizations(
    clients: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    output_dir: str = 'results/plots',
    trust_manager: Optional[Any] = None
) -> None:
    """
    Save all visualizations to files.
    
    Args:
        clients: List of client information
        results: Dictionary with results from all approaches
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot trust distribution
    plot_trust_distribution(clients, 
                           save_path=os.path.join(output_dir, 'trust_distribution.png'))
    
    # Plot performance comparison
    plot_performance_comparison(
        {k: v for k, v in results.items() if isinstance(v, dict) and 'accuracy' in v},
        save_path=os.path.join(output_dir, 'performance_comparison.png')
    )
    
    # Plot confusion matrices
    plot_confusion_matrices(results, 
                           save_path=os.path.join(output_dir, 'confusion_matrices.png'))
    
    # Plot ROC curves (if probabilities available)
    # Extract y_true from first result that has it
    y_true_for_roc = None
    for result in results.values():
        if 'y_true' in result:
            y_true_for_roc = np.array(result['y_true'])
            break
    
    if y_true_for_roc is not None:
        plot_roc_curves(results, y_true=y_true_for_roc,
                       save_path=os.path.join(output_dir, 'roc_curves.png'),
                       title='ROC Curves Comparison')
    
    # Plot trust vs performance
    plot_trust_vs_performance(clients, 
                             save_path=os.path.join(output_dir, 'trust_vs_performance.png'))
    
    # Plot radar chart
    plot_metrics_radar(
        {k: v for k, v in results.items() if isinstance(v, dict) and 'accuracy' in v},
        save_path=os.path.join(output_dir, 'metrics_radar.png')
    )
    
    # Plot trust evolution (if multi-round data available)
    plot_trust_evolution(
        clients,
        trust_manager=trust_manager,
        save_path=os.path.join(output_dir, 'trust_evolution.png')
    )
    
    # Plot trust trends
    plot_trust_trends(
        clients,
        trust_manager=trust_manager,
        save_path=os.path.join(output_dir, 'trust_trends.png')
    )
    
    # Plot trust explanation (beginner-friendly)
    plot_trust_explanation(
        clients,
        save_path=os.path.join(output_dir, 'trust_explanation.png')
    )
    
    print(f"All visualizations saved to {output_dir}")
