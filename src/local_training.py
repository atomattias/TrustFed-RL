"""
Local model training module for federated clients.

This module handles training machine learning models at each federated client.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, Tuple, Any, Optional
import pandas as pd
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    # On some macOS setups xgboost can be installed but fail to import at runtime
    # due to missing OpenMP (libomp). Treat as unavailable unless fully usable.
    XGBOOST_AVAILABLE = False


def train_local_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    random_state: int = 42,
    **model_kwargs
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a local model on client data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('random_forest' or 'logistic_regression')
        random_state: Random seed
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Tuple of (trained_model, training_metrics_dict)
    """
    # Defensive sanitization: some feature pipelines can produce NaN/Inf.
    # These can trigger hard floating-point exceptions inside BLAS during sklearn dot-products.
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    if model_type == 'random_forest':
        n_estimators = model_kwargs.get('n_estimators', 350)  # Increased from 250 for better capacity
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=25,  # Increased depth for better learning (with regularization)
            min_samples_split=8,  # Slightly reduced to allow more splits
            min_samples_leaf=4,  # Slightly reduced for better learning
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'mlp':
        # MLP with reasonable architecture for intrusion detection
        # Optimized for small feature space (6 features in CTU-13)
        hidden_layer_sizes = model_kwargs.get('hidden_layer_sizes', (50,))  # Single layer optimized for 6 features
        max_iter = model_kwargs.get('max_iter', 1000)  # More iterations
        early_stopping = model_kwargs.get('early_stopping', True)  # Enable early stopping
        learning_rate_init = model_kwargs.get('learning_rate_init', 0.001)
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
        n_estimators = model_kwargs.get('n_estimators', 100)
        max_depth = model_kwargs.get('max_depth', 6)
        learning_rate = model_kwargs.get('learning_rate', 0.1)
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
    elif model_type == 'logistic_regression':
        # Check for single class - Logistic Regression requires at least 2 classes
        unique_classes = y_train.unique() if hasattr(y_train, 'unique') else pd.Series(y_train).unique()
        if len(unique_classes) < 2:
            # Single class: return dummy model that always predicts the single class
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='constant', constant=unique_classes[0] if len(unique_classes) > 0 else 0)
            model.fit(X_train, y_train)
            # Return metrics indicating poor performance
            train_metrics = {
                'train_accuracy': 0.5,  # Random performance
                'train_f1': 0.0,
                'train_precision': 0.0,
                'train_recall': 0.0
            }
            return model, train_metrics
        
        # Client-side LR training is used mainly for computing per-client validation metrics and trust signals.
        # Some macOS BLAS builds can raise hard floating-point exceptions in sklearn's dot-products.
        # Use a small NumPy implementation that avoids BLAS-backed matrix multiply for stability.

        class _NumpyLogReg:
            def __init__(self, coef: np.ndarray, intercept: float):
                self.coef_ = coef.reshape(1, -1)
                self.intercept_ = np.array([float(intercept)])
                self.classes_ = np.array([0, 1])

            def predict_proba(self, X):
                Xn = np.asarray(X, dtype=float)
                w = self.coef_.ravel()
                z = (Xn * w).sum(axis=1) + float(self.intercept_[0])
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
                return np.vstack([1.0 - p, p]).T

            def predict(self, X):
                proba = self.predict_proba(X)[:, 1]
                return (proba >= 0.5).astype(int)

        Xn = np.asarray(X_train, dtype=float)
        yn = np.asarray(y_train, dtype=int).ravel()
        n, d = Xn.shape

        w = np.zeros(d, dtype=float)
        b = 0.0
        lr = float(model_kwargs.get('lr', 0.05))
        epochs = int(model_kwargs.get('local_epochs', 2))
        batch_size = int(model_kwargs.get('batch_size', 1024))
        rng = np.random.default_rng(random_state)

        for _ in range(max(1, epochs)):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                bi = idx[start:start + batch_size]
                Xb = Xn[bi]
                yb = yn[bi]
                z = (Xb * w).sum(axis=1) + b
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
                err = (p - yb)
                grad_w = (Xb * err[:, None]).mean(axis=0)
                grad_b = float(err.mean())
                w -= lr * grad_w
                b -= lr * grad_b

        model = _NumpyLogReg(w, b)
        # Compute training metrics and return (already trained).
        y_train_pred = model.predict(Xn)
        train_acc = accuracy_score(yn, y_train_pred)
        train_f1 = f1_score(yn, y_train_pred, zero_division=0)
        training_metrics = {
            'train_accuracy': float(train_acc),
            'train_f1': float(train_f1)
        }
        return model, training_metrics
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest', 'logistic_regression', 'mlp', or 'xgboost'")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    training_metrics = {
        'train_accuracy': train_acc,
        'train_f1': train_f1
    }
    
    return model, training_metrics


def evaluate_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate a model on validation data.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
    }
    
    # Calculate false positive rate
    cm = confusion_matrix(y_val, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        metrics['false_positive_rate'] = 0.0
    
    return metrics


def get_model_parameters(model: Any, model_type: str) -> Dict[str, np.ndarray]:
    """
    Extract model parameters for federated aggregation.
    
    Args:
        model: Trained model
        model_type: Type of model ('random_forest' or 'logistic_regression')
        
    Returns:
        Dictionary with model parameters
    """
    from sklearn.dummy import DummyClassifier
    
    params = {}
    
    # Handle DummyClassifier (used for single-class cases)
    if isinstance(model, DummyClassifier):
        # DummyClassifier doesn't have coef_ or feature_importances_
        # Create dummy parameters with zeros
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            # Fallback: try to infer from training data
            n_features = 10  # Default fallback
        
        if model_type == 'logistic_regression':
            # Create zero coefficients and intercept
            params['coef'] = np.zeros((1, n_features))
            params['intercept'] = np.zeros(1)
            params['classes'] = model.classes_ if hasattr(model, 'classes_') else np.array([0, 1])
        elif model_type == 'mlp':
            # For MLP, create dummy weights/biases (simplified)
            params['coefs'] = [np.zeros((n_features, 10)), np.zeros((10, 1))]  # Simplified structure
            params['intercepts'] = [np.zeros(10), np.zeros(1)]
            params['n_layers'] = 2
            params['n_outputs'] = 1
            params['classes'] = model.classes_ if hasattr(model, 'classes_') else np.array([0, 1])
            params['n_features'] = n_features
        elif model_type in ['random_forest', 'xgboost']:
            # For random forest and xgboost, create zero feature importances
            params['feature_importances'] = np.zeros(n_features)
            params['n_features'] = n_features
            params['classes'] = model.classes_ if hasattr(model, 'classes_') else np.array([0, 1])
        else:
            # Fallback
            params['feature_importances'] = np.zeros(n_features)
            params['n_features'] = n_features
            params['classes'] = model.classes_ if hasattr(model, 'classes_') else np.array([0, 1])
        return params
    
    if model_type == 'random_forest':
        # For Random Forest, we aggregate feature importances
        params['feature_importances'] = model.feature_importances_
        params['n_features'] = len(model.feature_importances_)
        params['classes'] = model.classes_
    elif model_type == 'mlp':
        # For MLP, we aggregate weights and biases from all layers
        params['coefs'] = [layer.copy() for layer in model.coefs_]
        params['intercepts'] = [layer.copy() for layer in model.intercepts_]
        params['n_layers'] = len(model.coefs_)
        params['n_outputs'] = model.n_outputs_
        params['classes'] = model.classes_
        params['n_features'] = model.n_features_in_
    elif model_type == 'xgboost':
        # For XGBoost, we aggregate feature importances (similar to Random Forest)
        params['feature_importances'] = model.feature_importances_
        params['n_features'] = len(model.feature_importances_)
        params['classes'] = model.classes_
    elif model_type == 'logistic_regression':
        # For Logistic Regression, we aggregate coefficients and intercept
        params['coef'] = model.coef_
        params['intercept'] = model.intercept_
        params['classes'] = model.classes_
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return params


def apply_logistic_weight_vector(
    model: Any,
    w_global: np.ndarray,
    feature_columns: list,
) -> Any:
    """Set coef_/intercept_ from concatenated weight vector [w_feat..., bias]."""
    from sklearn.linear_model import SGDClassifier

    if len(w_global) < 2:
        raise ValueError("w_global must include bias term")
    coef = w_global[:-1].reshape(1, -1)
    intercept = np.array([w_global[-1]], dtype=float)
    if not isinstance(model, SGDClassifier):
        model = SGDClassifier(loss="log_loss", max_iter=1, tol=1e-3, random_state=42)
    model.coef_ = coef
    model.intercept_ = intercept
    if not hasattr(model, "classes_"):
        model.classes_ = np.array([0, 1])
    model.n_features_in_ = len(feature_columns)
    return model
