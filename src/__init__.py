"""
TrustFed-RL source package (federated IoMT IDS).
"""

from .preprocessing import load_client_data, prepare_labels, split_data, prepare_features
from .local_training import train_local_model, evaluate_model, get_model_parameters
from .federated_client import FederatedClient
from .federated_server import FedAvgAggregator, TrustAwareAggregator, EnsembleAggregator, TrustManager, TrustHistory
from .evaluation import compute_metrics, evaluate_model_on_test, compare_approaches, generate_results_summary
from .visualization import (
    plot_trust_distribution, plot_performance_comparison, plot_confusion_matrices,
    plot_trust_vs_performance, plot_metrics_radar, plot_trust_evolution,
    plot_trust_trends, save_all_visualizations
)

__all__ = [
    'load_client_data', 'prepare_labels', 'split_data', 'prepare_features',
    'train_local_model', 'evaluate_model', 'get_model_parameters',
    'FederatedClient',
    'FedAvgAggregator', 'TrustAwareAggregator', 'EnsembleAggregator',
    'TrustManager', 'TrustHistory',
    'compute_metrics', 'evaluate_model_on_test', 'compare_approaches', 'generate_results_summary',
    'plot_trust_distribution', 'plot_performance_comparison', 'plot_confusion_matrices',
    'plot_trust_vs_performance', 'plot_metrics_radar', 'plot_trust_evolution',
    'plot_trust_trends', 'save_all_visualizations'
]
