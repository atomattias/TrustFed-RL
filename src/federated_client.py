"""
Federated client module representing a hospital / IoMT site.

Each client trains a local model and computes its trust score.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter
from preprocessing import load_client_data, prepare_labels, split_data, prepare_features
from local_training import train_local_model, evaluate_model, get_model_parameters


class FederatedClient:
    """
    Represents a single federated learning client (hospital site).
    
    Each client:
    1. Loads its own data
    2. Trains a local model
    3. Evaluates the model to compute trust score
    4. Provides model parameters for aggregation
    """
    
    def __init__(
        self,
        client_id: str,
        data_path: str,
        model_type: str = 'random_forest',
        random_state: int = 42,
        client_quality: str = 'unknown',
        clean_validation_source: Optional[str] = None,
        performance_signal: str = 'accuracy',
        uniform_contextual_prior: bool = False,
        use_shared_val_ref: bool = False,
    ):
        """
        Initialize a federated client.
        
        Args:
            client_id: Unique identifier for this client
            data_path: Path to the client's CSV data file (may be corrupted)
            model_type: Type of model to train ('random_forest' or 'logistic_regression')
            random_state: Random seed for reproducibility
            client_quality: Quality tier ('high', 'medium', 'low', 'compromised', 'unknown')
            clean_validation_source: Optional path to clean source file for validation (if None, uses data_path)
            performance_signal: Val metric for trust ('accuracy', 'f1_score', 'recall')
            uniform_contextual_prior: If True, use b_i=0.10 for all clients (no planted tier lookup)
            use_shared_val_ref: If True with clean_validation_source, use 100% client CSV as
                train and 100% of the shared ref as X_val (no 80/20 re-split). Required for
                iomt_natural (Trusted-plan / NATURAL_PARTITION plan WP2).
        """
        self.client_id = client_id
        self.data_path = data_path
        self.model_type = model_type
        self.random_state = random_state
        self.client_quality = client_quality
        self.clean_validation_source = clean_validation_source
        self.performance_signal = performance_signal
        self.uniform_contextual_prior = bool(uniform_contextual_prior)
        self.use_shared_val_ref = bool(use_shared_val_ref)
        
        # Will be set after training
        self.model = None
        self.trust_score = None
        self.train_metrics = None
        self.val_metrics = None
        
        # Data splits
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_features = None
        
        # Multi-signal trust fusion tracking
        self.accuracy_history = []  # Track validation accuracy across rounds
        self.previous_parameters = None  # Track previous model parameters for drift calculation

        # Communication reliability (signal C): participation / lateness / completeness
        self.comm_rounds_expected = 0
        self.comm_rounds_completed = 0
        self.comm_late_count = 0
        self.comm_incomplete_count = 0

        # Performance history for adaptive trust (multi-round mode)
        self.performance_history: List[Dict[str, Any]] = []
        self.current_round = 0
        
        # Store original data for dynamic trust simulation
        self.X_train_original = None
        self.y_train_original = None
        self.X_val_original = None
        self.y_val_original = None
    
    def load_data(self) -> None:
        """Load and preprocess client data."""
        # Load CSV (may be corrupted for compromised clients)
        df = load_client_data(self.data_path)
        
        # Prepare labels
        df = prepare_labels(df)
        
        # Prepare features
        X = prepare_features(df)
        y = df['label']

        if self.use_shared_val_ref:
            if self.clean_validation_source is None:
                raise ValueError(
                    f"{self.client_id}: use_shared_val_ref=True requires clean_validation_source"
                )
            # Natural partition mode: client CSV is 100% train; shared D_val is 100% val.
            # Never re-split the trusted reference (plan WP2).
            self.X_train = X.copy()
            self.y_train = y.copy()
            try:
                df_clean = load_client_data(self.clean_validation_source)
                df_clean = prepare_labels(df_clean)
                X_clean = prepare_features(df_clean)
                y_clean = df_clean['label']
                # Align feature schema to this client's train columns
                X_clean = X_clean.reindex(columns=self.X_train.columns, fill_value=0.0)
                self.X_val = X_clean
                self.y_val = y_clean
                print(
                    f"\n  ✅ SHARED VAL REF (full-file): {self.client_id} "
                    f"train={len(self.X_train):,} val={len(self.X_val):,} "
                    f"from {Path(self.clean_validation_source).name}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"use_shared_val_ref requires readable clean_validation_source "
                    f"for {self.client_id}: {e}"
                ) from e
        else:
            # Legacy: local 80/20; optional clean source still re-split 80/20 (old behaviour)
            self.X_train, self.X_val, self.y_train, self.y_val = split_data(
                X, y, test_size=0.2, random_state=self.random_state
            )

            if self.clean_validation_source is not None:
                try:
                    df_clean = load_client_data(self.clean_validation_source)
                    df_clean = prepare_labels(df_clean)
                    X_clean = prepare_features(df_clean)
                    y_clean = df_clean['label']

                    _, X_val_clean, _, y_val_clean = split_data(
                        X_clean, y_clean, test_size=0.2, random_state=self.random_state
                    )
                    X_val_clean = X_val_clean.reindex(columns=self.X_train.columns, fill_value=0.0)
                    self.X_val = X_val_clean
                    self.y_val = y_val_clean

                    print(f"\n  ✅ CLEAN VALIDATION: {self.client_id} using clean validation from {Path(self.clean_validation_source).name}")
                    print(f"     This ensures trust scores reflect actual client quality!")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not load clean validation set for {self.client_id}: {e}")
                    print(f"     Using corrupted validation set (trust may be inaccurate)")
        
        # Store original data for dynamic trust simulation
        self.X_train_original = self.X_train.copy()
        self.y_train_original = self.y_train.copy()
        self.X_val_original = self.X_val.copy()
        self.y_val_original = self.y_val.copy()
        
        self.X_features = X

    def restore_training_data_from_original(self) -> bool:
        """
        Restore local training data from snapshots taken at load time.

        Baseline approaches (e.g., FedAvg pre-aggregation poisoning) may mutate
        X_train/y_train; TrustFed must start from the same pristine state as
        trust-only replication runs.
        """
        if self.X_train_original is None or self.y_train_original is None:
            return False
        self.X_train = self.X_train_original.copy()
        self.y_train = self.y_train_original.copy()
        # Drop stale V only; keep round S/D history (warmup clear used to wipe it).
        self.val_metrics = None
        return True
    
    def train(self, **model_kwargs) -> Dict[str, float]:
        """
        Train the local model on client data.
        
        Args:
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Dictionary with training metrics
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Train model
        self.model, self.train_metrics = train_local_model(
            self.X_train,
            self.y_train,
            model_type=self.model_type,
            random_state=self.random_state,
            **model_kwargs
        )
        # Invalidate cached val metrics so V (and trust) reflect this round's model.
        # Without this, post-t_a label-flip never moves V — stale warm-up scores persist.
        self.val_metrics = None

        return self.train_metrics
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the local model on validation data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.val_metrics = evaluate_model(self.model, self.X_val, self.y_val)
        return self.val_metrics

    def get_validation_performance_metric(self) -> float:
        """Primary validation metric for trust (scenario-configurable)."""
        if self.val_metrics is None:
            self.evaluate()
        key = self.performance_signal
        if key not in self.val_metrics:
            key = 'accuracy'
        return float(self.val_metrics.get(key, 0.0))

    def compute_trust(self) -> float:
        """
        Compute trust score based on multiple factors, not just validation accuracy.
        
        Improved trust calculation that accounts for:
        1. Training accuracy (model should learn from data)
        2. F1-score (handles class imbalance better than accuracy)
        3. Train-val gap (large gap indicates corruption or overfitting)
        4. Class imbalance (penalize majority class prediction)
        
        Returns:
            Trust score in [0, 1]
        """
        if self.val_metrics is None:
            self.evaluate()
        
        # Ensure train_metrics is available - compute if missing
        if self.train_metrics is None:
            if self.model is None:
                # Model not trained yet - cannot compute trust properly
                # Use a conservative low trust score
                self.trust_score = 0.1
                return 0.1
            else:
                # Compute training metrics on the fly
                y_train_pred = self.model.predict(self.X_train)
                from sklearn.metrics import accuracy_score, f1_score
                train_acc = accuracy_score(self.y_train, y_train_pred)
                train_f1 = f1_score(self.y_train, y_train_pred)
                self.train_metrics = {
                    'train_accuracy': train_acc,
                    'train_f1': train_f1
                }
        
        val_acc = self.val_metrics['accuracy']
        train_acc = self.train_metrics.get('train_accuracy', 0.0)
        f1_score = self.val_metrics.get('f1_score', 0.0)
        
        # Debug logging for compromised clients
        if 'compromised' in self.client_id.lower():
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"TRUST_CALC_DEBUG {self.client_id[:50]}: val_acc={val_acc:.4f}, "
                       f"train_acc={train_acc:.4f}, f1={f1_score:.4f}")
        
        # SIMPLIFIED TRUST CALCULATION: Use validation accuracy directly
        # As stated in README: "Trust scores are computed as validation accuracy"
        # This is simpler and more transparent than complex penalty systems
        # Validation accuracy on clean validation set correctly reflects client quality:
        # - High-quality clients (clean data) → high validation accuracy → high trust
        # - Low-quality clients (corrupted training, clean validation) → low validation accuracy → low trust
        # - Compromised clients (corrupted training, corrupted validation) → may have misleading high accuracy
        
        # Use the configured validation signal (F1 on iomt_natural, not accuracy)
        base_trust = self.get_validation_performance_metric()
        
        # Only apply minimal penalty for obvious majority class prediction
        # This catches cases where model just predicts the majority class
        if self.y_val is not None:
            val_label_counts = Counter(self.y_val)
            if len(val_label_counts) > 0:
                majority_count = max(val_label_counts.values())
                majority_ratio = majority_count / len(self.y_val)
                
                # If validation set is highly imbalanced (>90% one class) and accuracy is very high (>95%),
                # likely just predicting majority class - apply moderate penalty
                if majority_ratio > 0.9 and val_acc > 0.95:
                    # Moderate penalty: reduce trust by 20-30%
                    penalty_factor = 0.75  # Reduce by 25%
                    base_trust = base_trust * penalty_factor
        
        # Ensure trust score is in valid range
        self.trust_score = max(0.0, min(1.0, base_trust))
        
        # Debug logging for compromised clients
        if 'compromised' in self.client_id.lower():
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"TRUST_CALC_DEBUG {self.client_id[:50]}: FINAL trust_score={self.trust_score:.4f}, "
                       f"base_trust={base_trust:.4f}")
        
        return self.trust_score
    
    def record_round_performance(self, round_num: int) -> None:
        """
        Record performance metrics for the current round.
        
        Args:
            round_num: Current round number
        """
        if self.val_metrics is None:
            self.evaluate()
        
        self.current_round = round_num
        
        # Record performance metrics
        round_metrics = {
            'round': round_num,
            'validation_accuracy': self.val_metrics['accuracy'],
            'validation_f1': self.val_metrics['f1_score'],
            'validation_precision': self.val_metrics['precision'],
            'validation_recall': self.val_metrics['recall'],
            'false_positive_rate': self.val_metrics.get('false_positive_rate', 0.0),
            'trust_score': self.trust_score
        }
        
        # Add training metrics if available
        if self.train_metrics:
            round_metrics.update({
                'train_accuracy': self.train_metrics.get('train_accuracy', 0.0),
                'train_f1': self.train_metrics.get('train_f1', 0.0)
            })
        
        self.performance_history.append(round_metrics)
        
        # Track primary performance metric for multi-signal stability (same as lambda1 signal)
        self.accuracy_history.append(self.get_validation_performance_metric())
        
        # Store current model parameters for drift calculation in next round
        if self.model is not None:
            try:
                self.previous_parameters = get_model_parameters(self.model, self.model_type)
            except Exception:
                # If parameter extraction fails, set to None
                self.previous_parameters = None
    
    def get_consistency_score(self, window_size: int = 5) -> float:
        """
        Calculate consistency score based on variance of recent validation accuracies.
        Lower variance = higher consistency = more reliable.
        
        Args:
            window_size: Number of recent rounds to consider
            
        Returns:
            Consistency score (1.0 = perfectly consistent, 0.0 = highly variable)
        """
        if len(self.performance_history) < 2:
            return 1.0  # Not enough data
        
        # Get recent validation accuracies
        recent_accuracies = [
            perf['validation_accuracy'] 
            for perf in self.performance_history[-window_size:]
        ]
        
        if len(recent_accuracies) < 2:
            return 1.0
        
        # Calculate variance
        variance = np.var(recent_accuracies)
        
        # Convert variance to consistency score (inverse relationship)
        # Max variance for binary accuracy is 0.25 (when accuracy alternates 0 and 1)
        consistency = max(0.0, 1.0 - (variance / 0.25))
        
        return consistency
    
    def get_performance_trend(self, window_size: int = 5) -> str:
        """
        Analyze trend in validation accuracy.
        
        Args:
            window_size: Number of recent rounds to consider
            
        Returns:
            'improving', 'declining', or 'stable'
        """
        if len(self.performance_history) < 2:
            return 'stable'
        
        # Get recent validation accuracies
        recent_accuracies = [
            perf['validation_accuracy'] 
            for perf in self.performance_history[-window_size:]
        ]
        
        if len(recent_accuracies) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(recent_accuracies))
        slope = np.polyfit(x, recent_accuracies, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def get_model_update(self, round_num: Optional[int] = None, include_data: bool = True) -> Dict[str, Any]:
        """
        Get model parameters and trust score for federated aggregation.
        
        Args:
            round_num: Current round number (optional, for tracking)
            include_data: If True, include training data for retraining (true FedAvg)
            
        Returns:
            Dictionary with model parameters, trust score, and performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.trust_score is None:
            self.compute_trust()
        
        # Record performance if round number provided (skip duplicate for same round)
        if round_num is not None:
            already = (
                self.performance_history
                and self.performance_history[-1].get("round") == round_num
            )
            if not already:
                self.record_round_performance(round_num)
        
        # Get model parameters
        params = get_model_parameters(self.model, self.model_type)
        
        # Prepare update dictionary
        update = {
            'client_id': self.client_id,
            'model': self.model,
            'parameters': params,
            'trust': self.trust_score,
            'val_accuracy': self.val_metrics['accuracy'] if self.val_metrics else None,
            'val_f1': self.val_metrics['f1_score'] if self.val_metrics else None,
            'round': round_num if round_num is not None else self.current_round
        }
        
        # Include training data for retraining (true FedAvg)
        if include_data and self.X_train is not None and self.y_train is not None:
            update['X_train'] = self.X_train
            update['y_train'] = self.y_train
        
        # Add performance history and consistency metrics if available
        if len(self.performance_history) > 0:
            update['performance_history'] = self.performance_history
            update['consistency_score'] = self.get_consistency_score()
            update['trend'] = self.get_performance_trend()
            update['num_rounds'] = len(self.performance_history)
        
        return update
    
    def inject_dynamic_change(
        self, 
        round_num: int,
        change_type: str = 'degrade',
        severity: float = 0.2
    ) -> None:
        """
        Inject dynamic change into client data to simulate compromise or improvement.
        
        This simulates real-world scenarios where:
        - Clients get compromised mid-training (degradation)
        - Clients improve their data quality (improvement)
        
        Args:
            round_num: Current round number
            change_type: 'degrade' (add noise) or 'improve' (reduce noise, restore original)
            severity: How severe the change is (0.0 to 1.0)
                     For degrade: proportion of data to corrupt
                     For improve: proportion of noise to remove
        """
        if self.X_train_original is None:
            return  # Can't inject changes if original data not stored
        
        if change_type == 'degrade':
            # Add additional noise/corruption to simulate compromise
            import pandas as pd
            
            # Ensure we have DataFrames/Series
            if not isinstance(self.X_train, pd.DataFrame):
                X_train_df = pd.DataFrame(self.X_train)
            else:
                X_train_df = self.X_train.copy()
            
            if not isinstance(self.y_train, pd.Series):
                y_train_df = pd.Series(self.y_train)
            else:
                y_train_df = self.y_train.copy()
            
            # Add label noise
            label_noise_ratio = severity * 0.3  # Up to 30% additional label noise
            if label_noise_ratio > 0:
                # Flip labels
                n_flip = int(len(y_train_df) * label_noise_ratio)
                if n_flip > 0:
                    flip_indices = np.random.choice(len(y_train_df), size=n_flip, replace=False)
                    for idx in flip_indices:
                        y_train_df.iloc[idx] = 1 - y_train_df.iloc[idx]
            
            # Add feature corruption
            feature_corruption_ratio = severity * 0.2  # Up to 20% feature corruption
            if feature_corruption_ratio > 0:
                # Ensure numeric columns can accept additive Gaussian noise.
                # Some datasets store numeric features as int64; adding float noise would error.
                for col in X_train_df.columns:
                    if X_train_df[col].dtype in [np.int64, np.float64]:
                        X_train_df[col] = X_train_df[col].astype(float)

                n_corrupt = int(len(X_train_df) * feature_corruption_ratio)
                if n_corrupt > 0:
                    corrupt_indices = np.random.choice(len(X_train_df), size=n_corrupt, replace=False)
                    for idx in corrupt_indices:
                        # Corrupt random features
                        n_features = X_train_df.shape[1]
                        n_features_to_corrupt = max(1, int(n_features * 0.1))
                        corrupt_feat_indices = np.random.choice(n_features, size=n_features_to_corrupt, replace=False)
                        for feat_idx in corrupt_feat_indices:
                            col = X_train_df.columns[feat_idx]
                            if X_train_df[col].dtype in [np.int64, np.float64]:
                                noise = np.random.normal(0, X_train_df[col].std() * 2)
                                X_train_df.iloc[idx, feat_idx] += noise
            
            # Update stored data
            self.X_train = X_train_df
            self.y_train = y_train_df
            
        elif change_type == 'improve':
            # Restore towards original (simulate data quality improvement)
            restore_ratio = severity  # How much to restore (0.0 to 1.0)
            
            if restore_ratio >= 1.0:
                # Full restoration
                self.X_train = self.X_train_original.copy()
                self.y_train = self.y_train_original.copy()
            else:
                # Partial restoration: blend current with original
                self.X_train = (
                    (1 - restore_ratio) * self.X_train + 
                    restore_ratio * self.X_train_original
                )
                # For labels, restore a portion
                restore_indices = np.random.choice(
                    len(self.y_train), 
                    size=int(len(self.y_train) * restore_ratio), 
                    replace=False
                )
                self.y_train = self.y_train.copy()
                self.y_train.iloc[restore_indices] = self.y_train_original.iloc[restore_indices]
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this client.
        
        Returns:
            Dictionary with client information
        """
        info = {
            'client_id': self.client_id,
            'data_path': self.data_path,
            'model_type': self.model_type,
            'trust_score': self.trust_score
        }
        
        if self.X_train is not None:
            info['train_samples'] = len(self.X_train)
            info['val_samples'] = len(self.X_val)
            info['n_features'] = self.X_train.shape[1]
            info['train_benign_ratio'] = (self.y_train == 0).sum() / len(self.y_train)
            info['val_benign_ratio'] = (self.y_val == 0).sum() / len(self.y_val)
        
        if self.train_metrics:
            info.update({f'train_{k}': v for k, v in self.train_metrics.items()})
        
        if self.val_metrics:
            info.update({f'val_{k}': v for k, v in self.val_metrics.items()})
        
        return info
    
    def record_communication_round(
        self,
        participated: bool = True,
        late: bool = False,
        incomplete: bool = False,
    ) -> None:
        """Record participation evidence for communication reliability (signal C)."""
        self.comm_rounds_expected += 1
        if participated:
            self.comm_rounds_completed += 1
        if late:
            self.comm_late_count += 1
        if incomplete:
            self.comm_incomplete_count += 1

    def get_communication_reliability(self) -> float:
        """
        Communication reliability C ∈ [0, 1].

        Penalizes missed rounds, late submissions, and incomplete updates.
        """
        expected = max(1, self.comm_rounds_expected)
        completion = self.comm_rounds_completed / expected
        late_rate = self.comm_late_count / expected
        incomplete_rate = self.comm_incomplete_count / expected
        C = completion - 0.5 * late_rate - 0.5 * incomplete_rate
        return float(max(0.0, min(1.0, C)))

    def get_contextual_risk(self, operational_penalty: float = 0.0) -> float:
        """
        Contextual / operational risk ∈ [0, 1] (higher = riskier).

        Combines hospital/client risk cues with response-fidelity / operational penalty
        (master maps closed-loop response evidence into contextual risk R).
        """
        # Planted tier prior b_i (simulation oracle) unless uniform ablation is requested.
        if self.uniform_contextual_prior:
            base = 0.1
        else:
            base = 0.1
            cid = self.client_id.lower()
            quality = (self.client_quality or "").lower()
            if "compromised" in cid or quality == "compromised":
                base = 0.35
            elif "low" in cid or quality == "low":
                base = 0.2
        risk = base + 0.65 * float(max(0.0, min(1.0, operational_penalty)))
        return float(max(0.0, min(1.0, risk)))

    def compute_multi_signal_trust_signals(
        self,
        operational_penalty: float = 0.0,
        include_response_fidelity_in_R: bool = True,
    ) -> Dict[str, float]:
        """
        Compute six master trust signals (Table V / §III-E):

        V validation performance, S temporal stability, D low model-update drift,
        U predictive confidence, C communication reliability,
        R contextual safety (= 1 − contextual risk).

        Also emits legacy aliases (accuracy/stability/…) for older callers.
        """
        signals: Dict[str, float] = {}

        # Signal V: always evaluate the current local model on clean val.
        self.evaluate()
        perf_value = self.get_validation_performance_metric()
        signals["V"] = float(perf_value)
        signals["accuracy"] = float(perf_value)
        val_acc = float(self.val_metrics["accuracy"])
        val_f1 = float(self.val_metrics.get("f1_score", 0.0))
        if self.performance_signal == "accuracy" and abs(perf_value - val_acc) > 1e-6:
            raise RuntimeError(
                f"V mismatch for {self.client_id}: performance_signal=accuracy but "
                f"V={perf_value:.4f} != val_accuracy={val_acc:.4f}"
            )
        if self.performance_signal == "f1_score" and abs(perf_value - val_f1) > 1e-6:
            raise RuntimeError(
                f"V mismatch for {self.client_id}: performance_signal=f1_score but "
                f"V={perf_value:.4f} != val_f1={val_f1:.4f}"
            )

        # Signal S: stability (inverse of variance of performance history)
        if len(self.accuracy_history) >= 2:
            variance = np.var(self.accuracy_history)
            stability = max(0.0, 1.0 - (variance / 0.25))
        else:
            stability = 1.0
        signals["S"] = float(stability)
        signals["stability"] = float(stability)

        # Signal D: low drift (inverse of parameter-change norm)
        if self.model is not None and self.previous_parameters is not None:
            try:
                current_params = get_model_parameters(self.model, self.model_type)
                if isinstance(current_params, dict) and isinstance(self.previous_parameters, dict):
                    param_diff_norm = 0.0
                    for key in current_params:
                        if key in self.previous_parameters:
                            if isinstance(current_params[key], np.ndarray):
                                diff = current_params[key] - self.previous_parameters[key]
                                param_diff_norm += np.linalg.norm(diff) ** 2
                            elif isinstance(current_params[key], (int, float)):
                                param_diff_norm += (
                                    current_params[key] - self.previous_parameters[key]
                                ) ** 2
                    param_diff_norm = np.sqrt(param_diff_norm)
                else:
                    param_diff_norm = (
                        abs(current_params - self.previous_parameters)
                        if isinstance(current_params, (int, float))
                        else 1.0
                    )
                drift_penalty = min(1.0, param_diff_norm / 10.0)
                drift_score = 1.0 - drift_penalty
            except Exception:
                drift_score = 1.0
        else:
            drift_score = 1.0
        signals["D"] = float(drift_score)
        signals["drift"] = float(drift_score)

        # Signal U: predictive confidence (inverse entropy)
        if self.model is not None and self.X_val is not None and len(self.X_val) > 0:
            try:
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(self.X_val)
                    epsilon = 1e-10
                    proba = np.clip(proba, epsilon, 1.0 - epsilon)
                    entropy_per_sample = -np.sum(proba * np.log(proba), axis=1)
                    avg_entropy = np.mean(entropy_per_sample)
                else:
                    avg_entropy = 0.5
                max_entropy = np.log(2)
                uncertainty_penalty = min(1.0, avg_entropy / max_entropy)
                uncertainty_score = 1.0 - uncertainty_penalty
            except Exception:
                uncertainty_score = 0.5
        else:
            uncertainty_score = 0.5
        signals["U"] = float(uncertainty_score)
        signals["uncertainty"] = float(uncertainty_score)

        # Signal C: communication reliability
        C = self.get_communication_reliability()
        signals["C"] = float(C)
        signals["communication"] = float(C)

        # Signal R: contextual safety = 1 − risk (response fidelity feeds risk when enabled)
        op = float(operational_penalty) if include_response_fidelity_in_R else 0.0
        risk = self.get_contextual_risk(operational_penalty=op)
        safety = 1.0 - risk
        signals["contextual_risk"] = float(risk)
        signals["R"] = float(safety)
        signals["contextual_safety"] = float(safety)
        signals["response_fidelity_penalty"] = float(operational_penalty)

        return signals