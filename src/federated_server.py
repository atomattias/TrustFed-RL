"""
Federated server module for aggregating client model updates.

This module implements both standard (equal-weight) and trust-aware aggregation.
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _as_feature_frame(X: Any) -> pd.DataFrame:
    """Coerce client features to DataFrame (co-adaptive / resample may yield ndarray)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _as_label_series(y: Any) -> pd.Series:
    """Coerce client labels to Series (co-adaptive poisoning stores ndarray)."""
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    return pd.Series(np.asarray(y).ravel())


class FedAvgAggregator:
    """
    Standard federated averaging aggregator (equal weights for all clients).
    
    This serves as the baseline federated learning approach.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the aggregator.
        
        Args:
            model_type: Type of model ('random_forest', 'logistic_regression', 'mlp', or 'xgboost')
        """
        self.model_type = model_type
        self.global_model = None
    
    def aggregate(self, client_updates: List[Dict[str, Any]], use_retraining: bool = True) -> Any:
        """
        Aggregate client model updates with equal weights.
        
        If use_retraining=True, collects data samples from clients and retrains a global model
        (true FedAvg approach). Otherwise, uses parameter averaging (legacy approach).
        
        Args:
            client_updates: List of client updates, each containing:
                - 'parameters': Model parameters dict
                - 'model': The trained model object
                - 'X_train': Training features (if use_retraining=True)
                - 'y_train': Training labels (if use_retraining=True)
            use_retraining: If True, retrain global model on aggregated data (recommended)
                
        Returns:
            Global aggregated model
        """
        if not client_updates:
            raise ValueError("No client updates provided")
        
        num_clients = len(client_updates)
        
        # True FedAvg: Retrain global model on aggregated data
        if use_retraining:
            # Collect data samples from all clients
            X_aggregated_list = []
            y_aggregated_list = []
            
            for update in client_updates:
                if 'X_train' in update and 'y_train' in update:
                    X_train = update['X_train']
                    y_train = update['y_train']
                    
                    # Sample data (to avoid memory issues with large datasets)
                    # Use all data if small, otherwise sample proportionally
                    max_samples_per_client = 15000
                    if len(X_train) > max_samples_per_client:
                        # Sample proportionally to maintain class balance
                        from sklearn.utils import resample
                        X_train_sampled, y_train_sampled = resample(
                            X_train, y_train,
                            n_samples=max_samples_per_client,
                            random_state=42,
                            stratify=y_train
                        )
                        X_aggregated_list.append(X_train_sampled)
                        y_aggregated_list.append(y_train_sampled)
                    else:
                        X_aggregated_list.append(X_train)
                        y_aggregated_list.append(y_train)
            
            if X_aggregated_list:
                # Combine all client data
                # Coerce types: co-adaptive poisoning / resample may yield ndarrays
                X_aggregated_list = [_as_feature_frame(x) for x in X_aggregated_list]
                y_aggregated_list = [_as_label_series(y) for y in y_aggregated_list]
                # Ensure all DataFrames have the same columns before concatenating
                all_columns = set()
                for df in X_aggregated_list:
                    all_columns.update(df.columns)
                all_columns = sorted(list(all_columns))
                
                # Reindex all DataFrames to have the same columns (missing columns will be NaN)
                X_aggregated_list_aligned = [df.reindex(columns=all_columns) for df in X_aggregated_list]
                
                X_global = pd.concat(X_aggregated_list_aligned, ignore_index=True)
                y_global = pd.concat(y_aggregated_list, ignore_index=True)
                
                # Fill any NaN values that may have been created by missing columns
                X_global = X_global.fillna(0)
                
                # Store feature columns for test set alignment
                self.training_feature_columns = X_global.columns.tolist()
                
                print(f"  Retraining global model on aggregated data: {len(X_global)} samples")
                print(f"    Benign: {(y_global == 0).sum()} ({(y_global == 0).sum()/len(y_global)*100:.1f}%)")
                print(f"    Attack: {(y_global == 1).sum()} ({(y_global == 1).sum()/len(y_global)*100:.1f}%)")
                
                # Get model parameters from first client
                base_model = client_updates[0]['model']
                
                # Retrain global model
                if self.model_type == 'random_forest':
                    self.global_model = RandomForestClassifier(
                        n_estimators=base_model.n_estimators if hasattr(base_model, 'n_estimators') else 100,
                        random_state=base_model.random_state if hasattr(base_model, 'random_state') else 42,
                        n_jobs=-1
                    )
                elif self.model_type == 'mlp':
                    self.global_model = MLPClassifier(
                        hidden_layer_sizes=base_model.hidden_layer_sizes if hasattr(base_model, 'hidden_layer_sizes') else (100, 50),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        tol=1e-4
                    )
                elif self.model_type == 'xgboost':
                    if not XGBOOST_AVAILABLE:
                        raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
                    self.global_model = XGBClassifier(
                        n_estimators=base_model.n_estimators if hasattr(base_model, 'n_estimators') else 100,
                        max_depth=base_model.max_depth if hasattr(base_model, 'max_depth') else 6,
                        learning_rate=base_model.learning_rate if hasattr(base_model, 'learning_rate') else 0.1,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric='logloss',
                        use_label_encoder=False
                    )
                elif self.model_type == 'logistic_regression':
                    # Logistic regression requires at least 2 classes; fall back if aggregated data is single-class
                    unique_classes = np.unique(y_global.values) if hasattr(y_global, "values") else np.unique(y_global)
                    if len(unique_classes) < 2:
                        from sklearn.dummy import DummyClassifier
                        self.global_model = DummyClassifier(strategy='constant', constant=int(unique_classes[0]) if len(unique_classes) else 0)
                        self.global_model.fit(X_global, y_global)
                        return self.global_model

                    self.global_model = SGDClassifier(
                        loss='log_loss',
                        random_state=42,
                        max_iter=1000
                    )
                else:
                    raise ValueError(f"Unknown model_type: {self.model_type}")
                
                # Train on aggregated data
                self.global_model.fit(X_global, y_global)
                
                return self.global_model
            else:
                # Fallback to parameter averaging if no data provided
                print("  Warning: No training data provided, falling back to parameter averaging")
                use_retraining = False
        
        # Legacy approach: Parameter averaging (fallback)
        if not use_retraining:
            if self.model_type == 'random_forest':
                # Aggregate feature importances
                feature_importances_list = [
                    update['parameters']['feature_importances'] 
                    for update in client_updates
                ]
                
                # Stack and average
                importances_array = np.stack(feature_importances_list, axis=0)
                avg_importances = np.mean(importances_array, axis=0)
                
                # Use first model as base (legacy behavior)
                base_model = client_updates[0]['model']
                self.global_model = base_model
                self.aggregated_importances = avg_importances

            elif self.model_type == 'mlp':
                # For MLP, averaging weights/biases is complex and not recommended
                # Instead, use the model with highest trust score as base
                # This is a simplified approach - retraining is preferred
                base_model = client_updates[0]['model']
                self.global_model = base_model
                # Note: True MLP aggregation requires retraining (use_retraining=True)
                
            elif self.model_type == 'xgboost':
                # For XGBoost, aggregate feature importances with trust weights
                feature_importances_list = [
                    update['parameters']['feature_importances'] 
                    for update in client_updates
                ]
                
                # Weighted average
                weighted_sum = np.zeros_like(feature_importances_list[0])
                for i, importances in enumerate(feature_importances_list):
                    weighted_sum += self.client_weights[i] * importances
                
                self.aggregated_importances = weighted_sum
                
                # Use first model as base structure
                base_model = client_updates[0]['model']
                self.global_model = base_model
                
            elif self.model_type == 'logistic_regression':
                # Aggregate coefficients and intercepts
                coef_list = [update['parameters']['coef'] for update in client_updates]
                intercept_list = [update['parameters']['intercept'] for update in client_updates]
                
                # Stack and average
                coef_array = np.stack(coef_list, axis=0)
                intercept_array = np.stack(intercept_list, axis=0)
                
                global_coef = np.mean(coef_array, axis=0)
                global_intercept = np.mean(intercept_array, axis=0)
                
                # Create global model
                self.global_model = SGDClassifier(
                    loss='log_loss',
                    random_state=42
                )
                
                # Set aggregated parameters
                self.global_model.coef_ = global_coef
                self.global_model.intercept_ = global_intercept
                self.global_model.classes_ = client_updates[0]['parameters']['classes']
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return self.global_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the global model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.global_model is None:
            raise ValueError("No global model. Call aggregate() first.")
        
        return self.global_model.predict(X)


class CoordinateMedianAggregator:
    """
    Coordinate-wise median aggregation of logistic-regression parameters.

    Path AGG robust baseline (Rm): same late-compromise upload path as FedAvg/B2,
    but aggregates coef/intercept by median instead of mean or sample retraining.
    Sign-flip and other upload poisons therefore affect the global model.
    """

    def __init__(self, model_type: str = "logistic_regression"):
        if model_type != "logistic_regression":
            raise ValueError(
                "CoordinateMedianAggregator supports logistic_regression only"
            )
        self.model_type = model_type
        self.global_model = None
        self.training_feature_columns: Optional[List[str]] = None
        self.client_weights: Optional[np.ndarray] = None

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Any:
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        coef_list = []
        intercept_list = []
        for update in client_updates:
            params = update.get("parameters") or {}
            coef = np.asarray(params["coef"], dtype=float)
            intercept = np.asarray(params["intercept"], dtype=float).reshape(-1)
            coef_list.append(coef)
            intercept_list.append(intercept)

        shapes = {c.shape for c in coef_list}
        if len(shapes) != 1:
            raise ValueError(
                f"Coordinate median requires aligned coef shapes; got {shapes}"
            )

        global_coef = np.median(np.stack(coef_list, axis=0), axis=0)
        global_intercept = np.median(np.stack(intercept_list, axis=0), axis=0)

        n = len(client_updates)
        self.client_weights = np.ones(n, dtype=float) / float(n)

        self.global_model = SGDClassifier(loss="log_loss", random_state=42)
        self.global_model.coef_ = global_coef
        self.global_model.intercept_ = global_intercept
        classes = client_updates[0]["parameters"].get("classes")
        if classes is not None:
            self.global_model.classes_ = np.asarray(classes)
        else:
            self.global_model.classes_ = np.array([0, 1])
        self.global_model.n_features_in_ = int(global_coef.shape[-1])

        # Optional column union for test alignment (same as FedAvg retraining path)
        cols: List[str] = []
        for update in client_updates:
            X = update.get("X_train")
            if isinstance(X, pd.DataFrame):
                cols.extend(list(X.columns))
        if cols:
            self.training_feature_columns = sorted(set(cols))

        return self.global_model

    def get_client_weights(
        self, client_updates: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        if self.client_weights is None:
            return {}
        if client_updates:
            return {
                update.get("client_id", f"client_{i}"): float(w)
                for i, (update, w) in enumerate(
                    zip(client_updates, self.client_weights)
                )
            }
        return {
            f"client_{i}": float(w) for i, w in enumerate(self.client_weights)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.global_model is None:
            raise ValueError("No global model. Call aggregate() first.")
        return self.global_model.predict(X)


class TrustAwareAggregator:
    """
    Trust-aware federated aggregator (weights clients by trust scores).
    
    This is the proposed method that weights client contributions by their trust scores.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        trust_manager: Optional['TrustManager'] = None,
        beta: float = 0.8,
        min_trust: float = 0.0
    ):
        """
        Initialize the trust-aware aggregator.
        
        Args:
            model_type: Type of model ('random_forest', 'logistic_regression', 'mlp', or 'xgboost')
            trust_manager: Optional TrustManager instance for adaptive trust (if None, uses trust from client updates)
            beta: Trust exponent for sub-linear weighting (default: 0.8, optimal)
        """
        self.model_type = model_type
        self.trust_manager = trust_manager
        self.beta = beta  # Trust exponent for sub-linear weighting
        self.min_trust = min_trust  # Optional hard threshold (exclude very low-trust clients)
        self.global_model = None
        self.client_weights = None
    
    def aggregate(self, client_updates: List[Dict[str, Any]], use_retraining: bool = True) -> Any:
        """
        Aggregate client model updates with trust-weighted averaging.
        
        If use_retraining=True, collects data samples weighted by trust and retrains a global model
        (true trust-aware FedAvg approach). Otherwise, uses parameter averaging (legacy approach).
        
        Args:
            client_updates: List of client updates, each containing:
                - 'client_id': Client identifier
                - 'parameters': Model parameters dict
                - 'trust': Trust score (used if trust_manager is None)
                - 'model': The trained model object
                - 'X_train': Training features (if use_retraining=True)
                - 'y_train': Training labels (if use_retraining=True)
            use_retraining: If True, retrain global model on trust-weighted aggregated data (recommended)
                
        Returns:
            Global aggregated model
        """
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # Prefer governance-provided aggregation weights when present (already
        # normalized + max-weight capped). Otherwise use TrustManager / update trust.
        if all("aggregation_weight" in u for u in client_updates):
            self.client_weights = np.array(
                [float(u["aggregation_weight"]) for u in client_updates], dtype=float
            )
            total_w = float(self.client_weights.sum())
            if total_w <= 0:
                self.client_weights = np.ones(len(client_updates)) / len(client_updates)
            else:
                self.client_weights = self.client_weights / total_w
            trust_scores = np.array(
                [float(u.get("trust", self.client_weights[i])) for i, u in enumerate(client_updates)]
            )
            self.original_trust_scores = trust_scores.copy()
        else:
            # Get trust scores from TrustManager if available, otherwise from client updates
            if self.trust_manager is not None:
                # Use adaptive trust from TrustManager
                trust_scores = np.array([
                    self.trust_manager.get_trust(update.get('client_id', f'client_{i}'))
                    for i, update in enumerate(client_updates)
                ])
            else:
                # Use static trust from client updates (backward compatibility)
                trust_scores = np.array([update['trust'] for update in client_updates])
            
            # Store original trust scores for threshold checks
            self.original_trust_scores = trust_scores.copy()
            
            # AGGRESSIVE TRUST WEIGHTING FOR BETTER PERFORMANCE: 
            # 1. Apply threshold: exclude clients with trust < 0.6 (very aggressive filtering)
            # 2. Apply trust^3 for remaining clients (much stronger differentiation)
            # This ensures compromised clients contribute minimally, high-trust clients dominate
            
            # Trust exponent (β) controls how much high-trust clients are weighted:
            # β=0.8 (sub-linear): Balanced, stable (optimal - tested and confirmed)
            # β=0.85-0.9: Slightly more differentiation (tested, performs worse)
            # β=1.0+ (linear/super-linear): Strong differentiation, less stable
            # Original β=0.8 is optimal after testing
            trust_clipped = np.maximum(trust_scores, 0.0)
            if self.min_trust > 0:
                trust_clipped = np.where(trust_clipped >= self.min_trust, trust_clipped, 0.0)

            trust_powered = trust_clipped ** self.beta  # Sub-linear weighting (configurable)
            
            total_trust_powered = np.sum(trust_powered)
            if total_trust_powered == 0:
                # Fallback to equal weights if all trust scores are zero
                self.client_weights = np.ones(len(client_updates)) / len(client_updates)
            else:
                # Normalize trust^1.5 weights (all clients contribute, weighted by trust^1.5)
                self.client_weights = trust_powered / total_trust_powered
        
        # CRITICAL FIX: Use retraining for BOTH Random Forest and Logistic Regression
        # FedAvg uses retraining and gets better results - Trust-Aware should too
        # Retraining on trust-weighted data is the proper Trust-Aware FedAvg approach
        if use_retraining:
            # Collect data samples from all clients, weighted by trust
            X_aggregated_list = []
            y_aggregated_list = []
            
            for i, update in enumerate(client_updates):
                if 'X_train' in update and 'y_train' in update:
                    X_train = update['X_train']
                    y_train = update['y_train']
                    
                    # Use the already-computed trust^1.5 weights (from normalization above)
                    # All clients contribute, weighted by trust^1.5 (no filtering)
                    trust_weight = float(self.client_weights[i])
                    original_trust = self.original_trust_scores[i] if hasattr(self, 'original_trust_scores') and i < len(self.original_trust_scores) else 0.5
                    
                    # Sample data proportionally to trust weight
                    # Higher trust = more samples from that client
                    # Use same max_samples_per_client as FedAvg for fair comparison
                    max_samples_per_client = 15000  # Match FedAvg's 10,000 samples per client
                    total_budget = max_samples_per_client * len(client_updates)
                    
                    # Trust-proportional sampling with minimum floor
                    # All clients get at least some samples (maintains diversity)
                    # Reduce minimum floor to avoid forcing low-trust clients into the retraining set
                    min_samples_per_client = 0
                    
                    # Use trust^1.5 weights directly (already normalized)
                    target_samples = int(total_budget * trust_weight)
                    target_samples = max(min_samples_per_client, min(target_samples, max_samples_per_client))

                    # If a client is effectively excluded (e.g., below min_trust), skip it.
                    if target_samples <= 0:
                        continue
                    
                    if len(X_train) > target_samples:
                        # Sample proportionally to trust weight, maintaining class balance
                        from sklearn.utils import resample
                        X_train_sampled, y_train_sampled = resample(
                            X_train, y_train,
                            n_samples=target_samples,
                            random_state=42,
                            stratify=y_train
                        )
                        X_aggregated_list.append(X_train_sampled)
                        y_aggregated_list.append(y_train_sampled)
                    else:
                        # Use all data if smaller than target
                        X_aggregated_list.append(X_train)
                        y_aggregated_list.append(y_train)
            
            if X_aggregated_list:
                # Combine all client data (already weighted by trust through sampling)
                # Coerce types: co-adaptive poisoning / resample may yield ndarrays
                X_aggregated_list = [_as_feature_frame(x) for x in X_aggregated_list]
                y_aggregated_list = [_as_label_series(y) for y in y_aggregated_list]
                # Ensure all DataFrames have the same columns before concatenating
                all_columns = set()
                for df in X_aggregated_list:
                    all_columns.update(df.columns)
                all_columns = sorted(list(all_columns))
                
                # Reindex all DataFrames to have the same columns (missing columns will be NaN)
                X_aggregated_list_aligned = [df.reindex(columns=all_columns) for df in X_aggregated_list]
                
                X_global = pd.concat(X_aggregated_list_aligned, ignore_index=True)
                y_global = pd.concat(y_aggregated_list, ignore_index=True)
                
                # Fill any NaN values that may have been created by missing columns
                X_global = X_global.fillna(0)
                
                # Store feature columns for test set alignment
                self.training_feature_columns = X_global.columns.tolist()
                
                print(f"  Retraining global model on trust-weighted aggregated data: {len(X_global)} samples")
                print(f"    Benign: {(y_global == 0).sum()} ({(y_global == 0).sum()/len(y_global)*100:.1f}%)")
                print(f"    Attack: {(y_global == 1).sum()} ({(y_global == 1).sum()/len(y_global)*100:.1f}%)")
                
                # Display trust weights
                print(f"    Trust weights applied:")
                for i, update in enumerate(client_updates):
                    client_id = update.get('client_id', f'client_{i}')
                    print(f"      {client_id}: {self.client_weights[i]:.4f}")
                
                # Get model parameters from first client
                base_model = client_updates[0]['model']
                
                # Retrain global model
                if self.model_type == 'random_forest':
                    self.global_model = RandomForestClassifier(
                        n_estimators=base_model.n_estimators if hasattr(base_model, 'n_estimators') else 350,
                        max_depth=25,  # Match local model settings
                        min_samples_split=8,
                        min_samples_leaf=4,
                        random_state=base_model.random_state if hasattr(base_model, 'random_state') else 42,
                        n_jobs=-1
                    )
                elif self.model_type == 'mlp':
                    self.global_model = MLPClassifier(
                        hidden_layer_sizes=base_model.hidden_layer_sizes if hasattr(base_model, 'hidden_layer_sizes') else (100, 50),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        tol=1e-4
                    )
                elif self.model_type == 'xgboost':
                    if not XGBOOST_AVAILABLE:
                        raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
                    self.global_model = XGBClassifier(
                        n_estimators=base_model.n_estimators if hasattr(base_model, 'n_estimators') else 100,
                        max_depth=base_model.max_depth if hasattr(base_model, 'max_depth') else 6,
                        learning_rate=base_model.learning_rate if hasattr(base_model, 'learning_rate') else 0.1,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric='logloss',
                        use_label_encoder=False
                    )
                elif self.model_type == 'logistic_regression':
                    unique_classes = np.unique(y_global.values) if hasattr(y_global, "values") else np.unique(y_global)
                    if len(unique_classes) < 2:
                        from sklearn.dummy import DummyClassifier
                        self.global_model = DummyClassifier(strategy='constant', constant=int(unique_classes[0]) if len(unique_classes) else 0)
                        self.global_model.fit(X_global, y_global)
                        return self.global_model

                    self.global_model = SGDClassifier(
                        loss='log_loss',
                        random_state=42,
                        max_iter=1000
                    )
                else:
                    raise ValueError(f"Unknown model_type: {self.model_type}")
                
                # Train on trust-weighted aggregated data
                self.global_model.fit(X_global, y_global)
                
                return self.global_model
            else:
                # Fallback to parameter averaging if no data provided
                print("  Warning: No training data provided, falling back to parameter averaging")
                use_retraining = False
        
        # Legacy approach: Parameter averaging (fallback)
        if not use_retraining:
            if self.model_type == 'random_forest':
                # Aggregate feature importances with trust weights
                feature_importances_list = [
                    update['parameters']['feature_importances'] 
                    for update in client_updates
                ]
                
                # Weighted average
                weighted_sum = np.zeros_like(feature_importances_list[0])
                for i, importances in enumerate(feature_importances_list):
                    weighted_sum += self.client_weights[i] * importances
                
                self.aggregated_importances = weighted_sum
                
                # Use first model as base structure
                base_model = client_updates[0]['model']
                self.global_model = base_model
                
            elif self.model_type == 'mlp':
                # For MLP, averaging weights/biases is complex and not recommended
                # Instead, use the model with highest trust score as base
                # This is a simplified approach - retraining is preferred
                base_model = client_updates[0]['model']
                self.global_model = base_model
                # Note: True MLP aggregation requires retraining (use_retraining=True)
                
            elif self.model_type == 'xgboost':
                # For XGBoost, aggregate feature importances with trust weights
                feature_importances_list = [
                    update['parameters']['feature_importances'] 
                    for update in client_updates
                ]
                
                # Weighted average
                weighted_sum = np.zeros_like(feature_importances_list[0])
                for i, importances in enumerate(feature_importances_list):
                    weighted_sum += self.client_weights[i] * importances
                
                self.aggregated_importances = weighted_sum
                
                # Use first model as base structure
                base_model = client_updates[0]['model']
                self.global_model = base_model
                
            elif self.model_type == 'logistic_regression':
                # Aggregate coefficients and intercepts with trust weights
                coef_list = [update['parameters']['coef'] for update in client_updates]
                intercept_list = [update['parameters']['intercept'] for update in client_updates]
                
                # Weighted average
                weighted_coef = np.zeros_like(coef_list[0])
                weighted_intercept = np.zeros_like(intercept_list[0])
                
                for i in range(len(client_updates)):
                    weighted_coef += self.client_weights[i] * coef_list[i]
                    weighted_intercept += self.client_weights[i] * intercept_list[i]
                
                # Create global model
                self.global_model = SGDClassifier(
                    loss='log_loss',
                    random_state=42
                )
                
                # Set aggregated parameters
                self.global_model.coef_ = weighted_coef
                self.global_model.intercept_ = weighted_intercept
                self.global_model.classes_ = client_updates[0]['parameters']['classes']
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return self.global_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the global model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.global_model is None:
            raise ValueError("No global model. Call aggregate() first.")
        
        return self.global_model.predict(X)
    
    def get_client_weights(self, client_updates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Get the trust weights assigned to each client.
        
        Args:
            client_updates: Optional list of client updates to extract client IDs
            
        Returns:
            Dictionary mapping client IDs to weights
        """
        if self.client_weights is None:
            return {}
        
        # Map weights to client IDs if available
        if client_updates:
            return {
                update.get('client_id', f'client_{i}'): float(weight)
                for i, (update, weight) in enumerate(zip(client_updates, self.client_weights))
            }
        else:
            # Fallback to indexed names
            return {f'client_{i}': float(weight) for i, weight in enumerate(self.client_weights)}


class EnsembleAggregator:
    """
    Alternative aggregator that uses ensemble voting (useful for Random Forest).
    
    This creates a meta-model that combines predictions from all client models.
    """
    
    def __init__(self, model_type: str = 'random_forest', aggregation_method: str = 'weighted_voting', beta: float = 0.8):
        """
        Initialize the ensemble aggregator.
        
        Args:
            model_type: Type of model
            aggregation_method: 'weighted_voting' or 'equal_voting'
            beta: Trust exponent for sub-linear weighting (default: 0.8, optimal)
        """
        self.model_type = model_type
        self.aggregation_method = aggregation_method
        self.beta = beta  # Trust exponent for sub-linear weighting
        self.client_models = []
        self.client_weights = None
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Any:
        """
        Create an ensemble of client models.
        
        Args:
            client_updates: List of client updates
            
        Returns:
            Ensemble model (list of models)
        """
        self.client_models = [update['model'] for update in client_updates]
        
        if self.aggregation_method == 'weighted_voting':
            trust_scores = np.array([update['trust'] for update in client_updates])
            
            # Apply beta exponentiation for sub-linear weighting (consistent with TrustAwareAggregator)
            # β < 1: sub-linear (compresses trust disparities)
            # β = 1: linear (standard weighted voting)
            # β > 1: super-linear (amplifies trust disparities)
            trust_powered = np.maximum(trust_scores, 0.0) ** self.beta
            total_trust_powered = np.sum(trust_powered)
            
            if total_trust_powered > 0:
                self.client_weights = trust_powered / total_trust_powered
            else:
                self.client_weights = np.ones(len(client_updates)) / len(client_updates)
        else:
            self.client_weights = np.ones(len(client_updates)) / len(client_updates)
        
        return self.client_models
    
    def get_client_weights(self, client_updates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Get the trust weights assigned to each client.
        
        Args:
            client_updates: Optional list of client updates to extract client IDs
            
        Returns:
            Dictionary mapping client IDs to weights
        """
        if self.client_weights is None:
            return {}
        
        # Map weights to client IDs if available
        if client_updates:
            return {
                update.get('client_id', f'client_{i}'): float(weight)
                for i, (update, weight) in enumerate(zip(client_updates, self.client_weights))
            }
        else:
            # Fallback to indexed names
            return {f'client_{i}': float(weight) for i, weight in enumerate(self.client_weights)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted voting from all models.
        
        Args:
            X: Feature matrix (can be DataFrame or numpy array)
            
        Returns:
            Predictions
        """
        if not self.client_models:
            raise ValueError("No models available. Call aggregate() first.")
        
        # Convert to numpy array if DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        
        # Get probability predictions from all models (better than hard predictions)
        probabilities_list = []
        for model in self.client_models:
            try:
                # Use predict_proba if available (gives probabilities instead of hard predictions)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_array)
                    # For binary classification, use probability of class 1 (attack)
                    if proba.shape[1] == 2:
                        proba_attack = proba[:, 1]  # Probability of attack class
                    else:
                        # Fallback to predict if predict_proba doesn't work as expected
                        proba_attack = model.predict(X_array).astype(float)
                    probabilities_list.append(proba_attack)
                else:
                    # Fallback to hard predictions if predict_proba not available
                    pred = model.predict(X_array)
                    probabilities_list.append(pred.astype(float))
            except Exception as e:
                # Fallback: try with DataFrame if array fails
                if hasattr(X, 'values'):
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        if proba.shape[1] == 2:
                            proba_attack = proba[:, 1]
                        else:
                            proba_attack = model.predict(X).astype(float)
                        probabilities_list.append(proba_attack)
                    else:
                        pred = model.predict(X)
                        probabilities_list.append(pred.astype(float))
                else:
                    raise e
        
        probabilities = np.array(probabilities_list)
        
        # Weighted average of probabilities
        if probabilities.ndim == 2 and probabilities.shape[0] > 1:
            # Multiple models: weighted average of probabilities
            weighted_proba = np.zeros(probabilities.shape[1])
            for i, prob in enumerate(probabilities):
                weighted_proba += self.client_weights[i] * prob
            
            # Threshold at 0.5 for binary classification
            final_predictions = (weighted_proba >= 0.5).astype(int)
        elif probabilities.ndim == 2:
            # Single probability array
            final_predictions = (probabilities[0] >= 0.5).astype(int)
        else:
            # Already 1D
            final_predictions = (probabilities >= 0.5).astype(int)
        
        return final_predictions


class TrustHistory:
    """
    Data structure to store trust history for a single client.
    
    Tracks trust scores, timestamps, performance metrics, and metadata
    across multiple federated learning rounds.
    """
    
    def __init__(self, client_id: str, initial_trust: float = 0.5):
        """
        Initialize trust history for a client.
        
        Args:
            client_id: Unique identifier for the client
            initial_trust: Initial trust score (default: 0.5 for neutral)
        """
        self.client_id = client_id
        self.trust_scores = [initial_trust]
        self.timestamps = [datetime.now().isoformat()]
        self.performance_metrics = []
        self.round_numbers = [0]  # Round 0 = initial
        self.metadata = {
            'initial_trust': initial_trust,
            'last_updated': datetime.now().isoformat(),
            'update_count': 0,
            'created_at': datetime.now().isoformat()
        }
    
    def add_round(
        self, 
        round_num: int, 
        trust_score: float, 
        performance_metrics: Dict[str, Any]
    ) -> None:
        """
        Add a new round's trust score and performance metrics.
        
        Args:
            round_num: Round number
            trust_score: Trust score for this round
            performance_metrics: Dictionary with performance metrics (val_acc, f1, etc.)
        """
        self.trust_scores.append(trust_score)
        self.timestamps.append(datetime.now().isoformat())
        self.round_numbers.append(round_num)
        self.performance_metrics.append(performance_metrics)
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['update_count'] += 1
    
    def get_latest_trust(self) -> float:
        """Get the most recent trust score."""
        return self.trust_scores[-1] if self.trust_scores else 0.5
    
    def get_trust_history(self, window_size: Optional[int] = None) -> List[float]:
        """
        Get trust score history, optionally limited to recent rounds.
        
        Args:
            window_size: Number of recent rounds to return (None = all)
            
        Returns:
            List of trust scores
        """
        if window_size is None:
            return self.trust_scores
        return self.trust_scores[-window_size:]
    
    def get_consistency_score(self, window_size: int = 5) -> float:
        """
        Calculate consistency score based on variance of recent trust scores.
        Lower variance = higher consistency = more reliable.
        
        Args:
            window_size: Number of recent rounds to consider
            
        Returns:
            Consistency score (1.0 = perfectly consistent, 0.0 = highly variable)
        """
        recent_trust = self.get_trust_history(window_size)
        if len(recent_trust) < 2:
            return 1.0  # Not enough data
        
        variance = np.var(recent_trust)
        # Convert variance to consistency score (inverse relationship)
        # Max variance for binary trust is 0.25 (when trust alternates 0 and 1)
        consistency = max(0.0, 1.0 - (variance / 0.25))
        return consistency
    
    def get_trend(self, window_size: int = 5) -> str:
        """
        Analyze trend in trust scores.
        
        Args:
            window_size: Number of recent rounds to consider
            
        Returns:
            'improving', 'declining', or 'stable'
        """
        recent_trust = self.get_trust_history(window_size)
        if len(recent_trust) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(recent_trust))
        slope = np.polyfit(x, recent_trust, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trust history to dictionary for serialization."""
        return {
            'client_id': self.client_id,
            'trust_scores': self.trust_scores,
            'timestamps': self.timestamps,
            'round_numbers': self.round_numbers,
            'performance_metrics': self.performance_metrics,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustHistory':
        """Create TrustHistory from dictionary."""
        history = cls(data['client_id'], data['metadata']['initial_trust'])
        history.trust_scores = data['trust_scores']
        history.timestamps = data['timestamps']
        history.round_numbers = data['round_numbers']
        history.performance_metrics = data['performance_metrics']
        history.metadata = data['metadata']
        return history


class TrustManager:
    """
    Manages trust scores for all clients in federated learning.
    
    Handles trust history storage, adaptive trust updates, decay, and anomaly detection.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        decay_rate: float = 0.95,
        anomaly_threshold: float = 0.2,
        initial_trust: float = 0.5,
        storage_dir: Optional[str] = None,
        use_multi_signal: bool = False,
        lambda_weights: Optional[Dict[str, float]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the TrustManager.

        Args:
            alpha: History weight for trust update (0.7 = 70% old, 30% new)
            decay_rate: Trust decay factor per round if client inactive (0.95 = 5% decay)
            anomaly_threshold: Threshold for detecting sudden trust drops
            initial_trust: Initial trust score for new clients
            storage_dir: Directory to store trust history (None = in-memory only)
            use_multi_signal: If True, use multi-signal trust fusion instead of simple validation accuracy
            lambda_weights: Legacy λ₁–λ₅ dict (migrated into signal_weights if signal_weights absent)
            signal_weights: Master six-signal weights {V,S,D,U,C,R} with Σwk=1
        """
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.anomaly_threshold = anomaly_threshold
        self.initial_trust = initial_trust
        self.storage_dir = storage_dir
        self.use_multi_signal = use_multi_signal

        # Master weighted-sum fusion (§III-E). Legacy lambda_weights kept for callers.
        if signal_weights is not None:
            total = sum(float(v) for v in signal_weights.values()) or 1.0
            self.signal_weights = {
                k: float(signal_weights.get(k, 0.0)) / total
                for k in ("V", "S", "D", "U", "C", "R")
            }
        elif lambda_weights is not None:
            raw = {
                "V": float(lambda_weights.get("lambda1", lambda_weights.get("V", 1.0))),
                "S": float(lambda_weights.get("lambda2", lambda_weights.get("S", 0.3))),
                "D": float(lambda_weights.get("lambda3", lambda_weights.get("D", 0.2))),
                "U": float(lambda_weights.get("lambda4", lambda_weights.get("U", 0.2))),
                "C": float(lambda_weights.get("C", lambda_weights.get("lambda_C", 0.15))),
                "R": float(lambda_weights.get("lambda5", lambda_weights.get("R", 0.15))),
            }
            total = sum(raw.values()) or 1.0
            self.signal_weights = {k: v / total for k, v in raw.items()}
        else:
            self.signal_weights = {
                "V": 0.25, "S": 0.15, "D": 0.15, "U": 0.15, "C": 0.15, "R": 0.15,
            }

        # Backward-compatible mirror of weights (legacy callers / logs)
        self.lambda_weights = {
            "lambda1": self.signal_weights["V"],
            "lambda2": self.signal_weights["S"],
            "lambda3": self.signal_weights["D"],
            "lambda4": self.signal_weights["U"],
            "lambda5": self.signal_weights["R"],
            "C": self.signal_weights["C"],
        }

        # Dictionary mapping client_id to TrustHistory
        self.trust_histories: Dict[str, TrustHistory] = {}

        # Create storage directory if specified
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
    
    def reset_all_histories(self) -> None:
        """Clear in-memory trust histories (e.g., before TrustFed multi-round phase)."""
        self.trust_histories = {}

    def initialize_client(self, client_id: str, initial_trust: Optional[float] = None) -> None:
        """
        Initialize trust history for a new client.
        
        Args:
            client_id: Unique client identifier
            initial_trust: Initial trust score (None = use default)
        """
        if client_id not in self.trust_histories:
            trust = initial_trust if initial_trust is not None else self.initial_trust
            self.trust_histories[client_id] = TrustHistory(client_id, trust)
            logger.info(f"Initialized trust for new client: {client_id}, Initial trust: {trust:.4f}")
    
    def compute_multi_signal_trust(
        self,
        signals: Dict[str, float]
    ) -> float:
        """
        Master fusion (§III-E): Ti = Σ_k wk xk with Σ wk = 1.

        Signals (normalized, higher = more trustworthy):
          V validation, S stability, D low-drift, U confidence,
          C communication reliability.
          R contextual safety may be included (legacy / B5-R) or weight 0
          (natural partition: R stays in RL/gov state only).

        Optional side channel: rl_trust_calibration_delta (Δτ) applied after fusion.
        """
        V = float(signals.get("V", signals.get("accuracy", 0.0)))
        S = float(signals.get("S", signals.get("stability", 1.0)))
        D = float(signals.get("D", signals.get("drift", 1.0)))
        U = float(signals.get("U", signals.get("uncertainty", 0.5)))
        C = float(signals.get("C", signals.get("communication", 1.0)))

        if "R" in signals:
            R = float(signals["R"])
        elif "contextual_safety" in signals:
            R = float(signals["contextual_safety"])
        else:
            risk = float(
                signals.get(
                    "contextual_risk",
                    signals.get("response_fidelity_penalty", 0.0),
                )
            )
            R = 1.0 - max(0.0, min(1.0, risk))

        # Clamp normalized evidence
        V, S, D, U, C, R = (max(0.0, min(1.0, x)) for x in (V, S, D, U, C, R))

        w = self.signal_weights
        Ti = (
            w["V"] * V
            + w["S"] * S
            + w["D"] * D
            + w["U"] * U
            + w["C"] * C
            + w["R"] * R
        )
        Ti = max(0.0, min(1.0, float(Ti)))

        delta_rl = float(signals.get("rl_trust_calibration_delta", 0.0))
        if delta_rl != 0.0:
            Ti = max(0.0, min(1.0, Ti + delta_rl))
        return Ti
    
    def update_trust(
        self,
        client_id: str,
        round_num: int,
        computed_trust: float,
        performance_metrics: Optional[Dict[str, Any]] = None,
        multi_signal_signals: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Update trust score for a client using adaptive trust formula.
        
        If use_multi_signal=True:
            Formula: S_i^r = λ1*Acc - λ2*Var - λ3*||Δ|| + λ4*(1-Entropy)
            Then: trust_new = α × trust_old + (1-α) × S_normalized
        
        Otherwise (simple mode):
            Formula: trust_new = α × trust_old + (1-α) × computed_trust
        
        Args:
            client_id: Client identifier
            round_num: Current round number
            computed_trust: Computed trust score from simple validation accuracy (0-1)
            performance_metrics: Additional performance metrics (optional)
            multi_signal_signals: Dictionary with multi-signal values (if use_multi_signal=True)
            
        Returns:
            Updated trust score
        """
        # Initialize client if not exists
        if client_id not in self.trust_histories:
            self.initialize_client(client_id)
        
        history = self.trust_histories[client_id]
        
        # Get previous trust score
        old_trust = history.get_latest_trust()
        
        # Use multi-signal trust fusion if enabled
        if self.use_multi_signal and multi_signal_signals is not None:
            # Compute trust using multi-signal fusion
            computed_trust = self.compute_multi_signal_trust(multi_signal_signals)
            logger.debug(
                f"Multi-signal trust - Client: {client_id}, "
                f"Signals: {multi_signal_signals}, Computed: {computed_trust:.4f}"
            )
        
        # For first round (only initial round 0 exists) or very low computed trust, use computed trust directly
        # This ensures compromised clients are immediately identified
        # Check if this is the first update: only round 0 exists (initial trust)
        is_first_update = (len(history.round_numbers) == 1 and history.round_numbers[0] == 0) or round_num == 1
        
        # Debug: Log first update detection
        if round_num == 1:
            logger.info(f"DEBUG First update check - Client: {client_id}, round_num: {round_num}, "
                       f"round_numbers: {history.round_numbers}, is_first_update: {is_first_update}, "
                       f"computed_trust: {computed_trust:.4f}")
        
        if is_first_update or computed_trust < 0.1:
            # First round or very low trust: use computed trust directly (no smoothing)
            new_trust = computed_trust
            if round_num == 1:
                logger.info(f"DEBUG Using computed trust directly - Client: {client_id}, "
                           f"computed_trust: {computed_trust:.4f}, new_trust: {new_trust:.4f}")
        else:
            # Adaptive trust update: weighted moving average of computed trust
            # This smooths the improved trust calculation over rounds
            # For low computed trust (< 0.2), use more aggressive weighting
            if computed_trust < 0.2:
                # Use lower alpha (more weight on computed trust) to quickly filter compromised clients
                effective_alpha = self.alpha * 0.3  # Reduce alpha by 70% for low-trust clients
            else:
                effective_alpha = self.alpha
            
            new_trust = effective_alpha * old_trust + (1 - effective_alpha) * computed_trust
        
        # Apply bounds validation
        new_trust = self._validate_trust_bounds(new_trust)
        
        # Log trust update
        trust_change = new_trust - old_trust
        val_acc = performance_metrics.get('validation_accuracy', computed_trust) if performance_metrics else computed_trust
        logger.info(
            f"Trust update - Client: {client_id}, Round: {round_num}, "
            f"Old: {old_trust:.4f}, New: {new_trust:.4f}, Change: {trust_change:+.4f}, "
            f"ComputedTrust: {computed_trust:.4f}, ValAcc: {val_acc:.4f}, Alpha: {self.alpha}"
        )
        
        # Prepare performance metrics
        if performance_metrics is None:
            performance_metrics = {'validation_accuracy': val_acc}
        else:
            performance_metrics['validation_accuracy'] = val_acc
        
        # Add consistency and trend information
        consistency = history.get_consistency_score()
        trend = history.get_trend()
        performance_metrics['consistency_score'] = consistency
        performance_metrics['trend'] = trend
        
        # Log additional metrics
        logger.debug(
            f"Trust metrics - Client: {client_id}, Consistency: {consistency:.4f}, "
            f"Trend: {trend}"
        )
        
        # Store in history
        history.add_round(round_num, new_trust, performance_metrics)
        
        return new_trust
    
    def get_trust(self, client_id: str) -> float:
        """
        Get current trust score for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Current trust score (or initial trust if client not found)
        """
        if client_id not in self.trust_histories:
            return self.initial_trust
        return self.trust_histories[client_id].get_latest_trust()
    
    def get_all_trust_scores(self) -> Dict[str, float]:
        """
        Get current trust scores for all clients.
        
        Returns:
            Dictionary mapping client_id to trust score
        """
        return {client_id: history.get_latest_trust() 
                for client_id, history in self.trust_histories.items()}
    
    def apply_decay(self, client_id: str, round_num: int) -> float:
        """
        Apply trust decay for inactive or underperforming clients.
        
        Args:
            client_id: Client identifier
            round_num: Current round number
            
        Returns:
            Decayed trust score
        """
        if client_id not in self.trust_histories:
            return self.initial_trust
        
        history = self.trust_histories[client_id]
        current_trust = history.get_latest_trust()
        
        # Apply decay
        decayed_trust = current_trust * self.decay_rate
        decayed_trust = self._validate_trust_bounds(decayed_trust)
        
        # Log decay application
        decay_amount = current_trust - decayed_trust
        logger.warning(
            f"Trust decay applied - Client: {client_id}, Round: {round_num}, "
            f"Before: {current_trust:.4f}, After: {decayed_trust:.4f}, "
            f"Decay: {decay_amount:.4f} ({self.decay_rate} rate)"
        )
        
        # Store decay event
        history.add_round(
            round_num,
            decayed_trust,
            {'decay_applied': True, 'decay_rate': self.decay_rate}
        )
        
        return decayed_trust
    
    def detect_anomaly(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Detect anomalies in trust score changes.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Anomaly information dict if anomaly detected, None otherwise
        """
        if client_id not in self.trust_histories:
            return None
        
        history = self.trust_histories[client_id]
        trust_scores = history.trust_scores
        
        if len(trust_scores) < 2:
            return None
        
        # Check for sudden drop
        recent_drop = trust_scores[-2] - trust_scores[-1]
        if recent_drop > self.anomaly_threshold:
            anomaly_info = {
                'type': 'sudden_drop',
                'magnitude': recent_drop,
                'previous_trust': trust_scores[-2],
                'current_trust': trust_scores[-1],
                'round': history.round_numbers[-1]
            }
            
            # Log anomaly detection
            logger.warning(
                f"Anomaly detected - Client: {client_id}, Type: {anomaly_info['type']}, "
                f"Magnitude: {recent_drop:.4f}, Round: {anomaly_info['round']}, "
                f"Previous: {trust_scores[-2]:.4f}, Current: {trust_scores[-1]:.4f}"
            )
            
            return anomaly_info
        
        return None
    
    def _validate_trust_bounds(self, trust: float) -> float:
        """
        Validate and clamp trust score to [0, 1] range.
        
        Args:
            trust: Trust score to validate
            
        Returns:
            Validated trust score in [0, 1]
        """
        # Handle NaN and infinity
        if not np.isfinite(trust):
            return self.initial_trust
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, trust))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about trust scores across all clients.
        
        Returns:
            Dictionary with mean, std, min, max, median trust scores
        """
        all_trust = [history.get_latest_trust() for history in self.trust_histories.values()]
        
        if not all_trust:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'count': 0
            }
        
        return {
            'mean': float(np.mean(all_trust)),
            'std': float(np.std(all_trust)),
            'min': float(np.min(all_trust)),
            'max': float(np.max(all_trust)),
            'median': float(np.median(all_trust)),
            'count': len(all_trust)
        }
    
    def save_trust_history(self, client_id: Optional[str] = None) -> None:
        """
        Save trust history to disk.
        
        Args:
            client_id: Specific client to save (None = save all)
        """
        if not self.storage_dir:
            logger.debug("No storage directory specified, skipping save")
            return
        
        clients_to_save = [client_id] if client_id else list(self.trust_histories.keys())
        saved_count = 0
        
        for cid in clients_to_save:
            if cid in self.trust_histories:
                history = self.trust_histories[cid]
                file_path = os.path.join(self.storage_dir, f"{cid}_trust_history.json")
                
                try:
                    with open(file_path, 'w') as f:
                        json.dump(history.to_dict(), f, indent=2)
                    saved_count += 1
                    logger.debug(f"Saved trust history for client: {cid} to {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save trust history for {cid}: {e}")
        
        if saved_count > 0:
            logger.info(f"Saved trust history for {saved_count} client(s)")
    
    def load_trust_history(self, client_id: str) -> bool:
        """
        Load trust history from disk.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.storage_dir:
            logger.debug("No storage directory specified, skipping load")
            return False
        
        file_path = os.path.join(self.storage_dir, f"{client_id}_trust_history.json")
        
        if not os.path.exists(file_path):
            logger.debug(f"Trust history file not found for client: {client_id}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.trust_histories[client_id] = TrustHistory.from_dict(data)
            logger.info(f"Loaded trust history for client: {client_id} from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load trust history for {client_id}: {e}")
            return False
    
    def load_all_trust_histories(self) -> int:
        """
        Load all trust histories from disk.
        
        Returns:
            Number of histories loaded
        """
        if not self.storage_dir:
            logger.debug("No storage directory specified, skipping load all")
            return 0
        
        loaded_count = 0
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('_trust_history.json'):
                    client_id = filename.replace('_trust_history.json', '')
                    if self.load_trust_history(client_id):
                        loaded_count += 1
        
        if loaded_count > 0:
            logger.info(f"Loaded trust histories for {loaded_count} client(s)")
        
        return loaded_count
