"""
Trust-weighted federated learning experiment engine (B1 baseline).

Used by TrustFed-RL IoMT experiments via `run_experiments.py regression`.
For full-stack / RL / governance runs use: python trustfed_agent_runner.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# For statistical tests
try:
    from scipy import stats
except ImportError:
    stats = None
    print("Warning: scipy not available. Statistical tests will be skipped.")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import load_client_data, prepare_labels, prepare_features, split_data
from local_training import train_local_model, apply_logistic_weight_vector
from federated_client import FederatedClient
from federated_server import FedAvgAggregator, TrustAwareAggregator, EnsembleAggregator, TrustManager, CoordinateMedianAggregator
from evaluation import evaluate_model_on_test, generate_results_summary
from config_loader import (
    load_trust_config,
    apply_trust_config,
    resolve_performance_signal,
    resolve_signal_weights,
    load_privacy_config,
)
from privacy import (
    DPSAGConfig,
    aggregate_clipped_gradients,
    aggregate_trust_weighted_hessian_grad,
    compute_logistic_gradient_stat,
    compute_logistic_irls_stats,
    estimate_epsilon,
    irls_newton_step,
    sample_local_batch,
)

# IMPORTANT: Do not import matplotlib/visualization at module import time.
# Some macOS environments can crash during matplotlib font initialization.
save_all_visualizations = None


class ExperimentRunner:
    """Orchestrates the complete experiment."""
    
    def __init__(
        self,
        data_dir: str = 'data/CSVs',
        model_type: str = 'random_forest',
        random_state: int = 42,
        test_csv: Optional[str] = None,
        num_rounds: int = 1,
        trust_alpha: float = 0.7,
        trust_storage_dir: Optional[str] = None,
        use_multi_signal: bool = False,
        performance_signal: Optional[str] = None,
        trust_use_retraining: bool = True,
        trust_no_retrain_only: bool = False,
        trust_only: bool = False,
        baseline_only: Optional[str] = None,
        dynamic_adversary: bool = True,
        lambda_weights_override: Optional[Dict[str, float]] = None,
        privacy_mode: str = "legacy",
        privacy_config_override: Optional[Dict[str, Any]] = None,
        uniform_contextual_prior: bool = False,
        shared_val_ref: Optional[str] = None,
        use_shared_val_ref: Optional[bool] = None,
        trust_config_path: Optional[str] = None,
    ):
        """
        Initialize the experiment runner.
        
        Args:
            data_dir: Directory containing CSV files
            model_type: Type of model ('random_forest' or 'logistic_regression')
            random_state: Random seed
            test_csv: Optional path to test CSV (if None, will use one of the client CSVs)
            num_rounds: Number of federated learning rounds (1 = single round, >1 = multi-round with adaptive trust).
                        Note: Multi-round mode (recommended: 10 rounds) typically yields better Trust-Aware performance
                        than single-round mode, often outperforming Centralized learning.
            trust_alpha: History weight for adaptive trust (0.7 = 70% old, 30% new)
            trust_storage_dir: Directory to store trust history (None = in-memory only)
            baseline_only: If 'fedavg' or 'median', run only that baseline for B0 / SOTA.
            uniform_contextual_prior: If True, use b_i=0.10 for all clients (B1-U ablation).
            shared_val_ref: Path to shared D_val CSV (iomt_natural). Auto-detected under iomt_natural/.
            use_shared_val_ref: Force full-file shared val; default True when path looks like iomt_natural.
            trust_config_path: Optional trust JSON (e.g. V-only B1 natural config).
        """
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.random_state = random_state
        self.test_csv = test_csv
        self.num_rounds = num_rounds
        self.uniform_contextual_prior = bool(uniform_contextual_prior)
        # Natural / ECU constructed partitions: auto-wire shared D_val
        if shared_val_ref is not None:
            self.shared_val_ref = Path(shared_val_ref)
        elif "iomt_natural" in str(self.data_dir) or "ecu_ioht" in str(self.data_dir):
            self.shared_val_ref = None
            for name in ("iomt_val_ref.csv", "ecu_val_ref.csv"):
                cand = self.data_dir.parent / name
                if cand.exists():
                    self.shared_val_ref = cand
                    break
        else:
            self.shared_val_ref = None
        if use_shared_val_ref is None:
            self.use_shared_val_ref = self.shared_val_ref is not None
        else:
            self.use_shared_val_ref = bool(use_shared_val_ref)
        if self.use_shared_val_ref and self.shared_val_ref is None:
            raise FileNotFoundError(
                "use_shared_val_ref=True but no shared_val_ref / iomt_val_ref.csv found"
            )
        if self.use_shared_val_ref:
            # Natural clients have no planted tiers — always uniform contextual prior
            self.uniform_contextual_prior = True
            # Legacy planted-tier mid-training adversary does not apply
            self.dynamic_adversary = False
            print(f"Shared D_val (full-file): {self.shared_val_ref}")
        self.late_compromise = None
        if self.use_shared_val_ref:
            from adversary.late_compromise import LateCompromiseConfig, LateCompromiseController
            import json
            adv_path = Path(__file__).resolve().parent / "config" / "iomt_natural_adversary.json"
            raw = json.loads(adv_path.read_text()) if adv_path.exists() else {"enabled": True}
            late_cfg = LateCompromiseConfig.from_dict(raw)
            late_cfg.seed = int(self.random_state)
            # Placeholder; attach after clients exist in setup_clients
            self._pending_late_cfg = late_cfg
        else:
            self._pending_late_cfg = None
        if trust_config_path:
            trust_config = load_trust_config(str(trust_config_path))
            print(f"Trust config: {trust_config_path}")
        else:
            trust_config = load_trust_config()
            if "iomt_natural" in str(self.data_dir):
                natural_cfg = Path(__file__).resolve().parent / "config" / "trust_config_iomt_natural.json"
                if natural_cfg.exists():
                    trust_config = load_trust_config(str(natural_cfg))
        self.trust_config = trust_config
        self.resolved_signal_weights = None
        self.round_logs: List[Dict[str, Any]] = []
        self.performance_signal = resolve_performance_signal(
            str(self.data_dir), trust_config, performance_signal
        )
        if use_multi_signal:
            print(f"Trust fusion performance signal (λ₁): {self.performance_signal}")
        self.trust_use_retraining = trust_use_retraining
        self.trust_no_retrain_only = trust_no_retrain_only
        self.trust_only = trust_only
        self.baseline_only = (baseline_only or "").strip().lower() or None
        # Natural / shared-val runs must keep dynamic_adversary=False so late-compromise
        # sync runs (do not overwrite the flag set above when use_shared_val_ref).
        if self.use_shared_val_ref:
            self.dynamic_adversary = False
        else:
            self.dynamic_adversary = dynamic_adversary
        self.lambda_weights_override = lambda_weights_override
        self.privacy_mode = privacy_mode
        privacy_cfg = load_privacy_config()
        self.privacy_strong_v1 = dict(privacy_cfg.get("strong_v1", {}))
        if privacy_config_override:
            self.privacy_strong_v1.update(privacy_config_override)
        if privacy_mode == "strong_v1":
            if model_type != "logistic_regression":
                raise ValueError("privacy_mode=strong_v1 requires logistic_regression")
            print(
                "Privacy mode: strong_v1 (gradient sufficient statistics + DP-SAG; no row upload)"
            )
        if not trust_use_retraining:
            print("Trust-Aware aggregation: parameter averaging (use_retraining=False)")
        if trust_no_retrain_only:
            print("Mode: trust-no-retrain-only (skipping Centralized/FedAvg/robust baselines)")
        if trust_only:
            print("Mode: trust-only (TrustFed retraining only; skipping baseline aggregators)")
        if self.baseline_only:
            print(f"Mode: baseline-only ({self.baseline_only})")
        
        self.clients = []
        self.client_updates = []
        self.test_data = None
        
        # Initialize TrustManager for adaptive trust (if multi-round)
        if num_rounds > 1:
            # Reuse trust_config already resolved above (iomt_natural → R∉T JSON)
            
            # IMPORTANT: do NOT load persistent trust histories by default.
            # Persistent trust is useful for long-lived deployments, but it is a confounder for experiments
            # because it leaks state across trials/datasets. Only enable storage if explicitly requested.
            if trust_storage_dir is None:
                trust_storage_dir = None
            
            # Create TrustManager with config or defaults
            trust_mgr_config = trust_config.get('trust_manager', {})
            
            # Master six-signal weights (legacy λ override still supported)
            signal_weights = None
            if use_multi_signal:
                legacy_override = None
                if lambda_weights_override is not None:
                    legacy_override = dict(lambda_weights_override)
                    print(
                        f"Trust fusion legacy λ override: "
                        f"λ₂={legacy_override.get('lambda2')}, "
                        f"λ₃={legacy_override.get('lambda3')}, "
                        f"λ₄={legacy_override.get('lambda4')}"
                    )
                include_r = None if not self.use_shared_val_ref else False
                signal_weights = resolve_signal_weights(
                    trust_config,
                    legacy_lambda_override=legacy_override,
                    include_R_in_T=include_r,
                )
                self.resolved_signal_weights = dict(signal_weights)
                print(f"Trust signal weights (Σ=1): {signal_weights}")
                if self.use_shared_val_ref and signal_weights.get("R", 0) > 1e-12:
                    raise RuntimeError("iomt_natural requires w_R=0 (R∉T); got R>0")

            # Use trust_alpha from argument if provided, otherwise use config
            final_alpha = trust_alpha if trust_alpha != 0.7 else trust_mgr_config.get('alpha', 0.7)

            self.trust_manager = TrustManager(
                alpha=final_alpha,
                decay_rate=trust_mgr_config.get('decay_rate', 0.95),
                anomaly_threshold=trust_mgr_config.get('anomaly_threshold', 0.2),
                initial_trust=trust_mgr_config.get('initial_trust', 0.5),
                storage_dir=trust_storage_dir,
                use_multi_signal=use_multi_signal,
                signal_weights=signal_weights,
            )
            
            # Store use_multi_signal flag
            self.use_multi_signal = use_multi_signal
            
            # Apply any additional config (but preserve alpha if explicitly set)
            apply_trust_config(self.trust_manager, trust_config)
            # Override alpha again after apply_trust_config to ensure command-line arg takes precedence
            if trust_alpha != 0.7:
                self.trust_manager.alpha = trust_alpha
            
            # Load existing trust histories only if storage_dir was explicitly provided
            if trust_storage_dir:
                self.trust_manager.load_all_trust_histories()
        else:
            self.trust_manager = None
            self.use_multi_signal = False
        
    def discover_client_files(self) -> List[str]:
        """
        Discover all CSV files in the data directory.
        Prefers heterogeneous clients if they exist, then mixed files, then original files.
        
        Returns:
            List of CSV file paths
        """
        csv_files = list(self.data_dir.glob('*.csv'))
        
        # Filter out benign files for client selection (we'll use them for mixing)
        # Also exclude reserved / evaluation-only files (test sets) from ever becoming clients.
        all_attack_files = [
            f for f in csv_files
            if 'benign' not in f.name.lower()
            and 'test_set' not in f.name.lower()
            and 'heterogeneous_test_set' not in f.name.lower()
        ]
        benign_files = [f for f in csv_files if 'benign' in f.name.lower()]
        
        # Check for heterogeneous clients (created by create_heterogeneous_clients.py)
        heterogeneous_files = [f for f in all_attack_files if any(
            tier in f.name.lower() for tier in ['high_quality', 'medium_quality', 'low_quality', 'compromised']
        )]

        # CICIoMT hospital clients (hospital_*_iomt.csv)
        iomt_files = [
            f for f in all_attack_files
            if f.name.lower().endswith('_iomt.csv') or f.name.lower().startswith('hospital_')
        ]
        
        # Prefer mixed files if they exist (created by prepare_realistic_data.py)
        mixed_files = [f for f in all_attack_files if f.name.startswith('mixed_') and f not in heterogeneous_files]
        original_files = [
            f for f in all_attack_files
            if not f.name.startswith('mixed_')
            and f not in heterogeneous_files
            and f not in iomt_files
        ]
        
        if iomt_files:
            attack_files = iomt_files
            print(f"Found {len(iomt_files)} IoMT hospital client files")
        elif heterogeneous_files:
            # Use heterogeneous clients (they have varying quality)
            attack_files = heterogeneous_files
            print(f"Found {len(heterogeneous_files)} heterogeneous client files (varying quality)")
            print(f"  This will demonstrate weaknesses of Centralized and FedAvg!")
            print(f"  High-quality, medium-quality, low-quality, and compromised clients detected")
        elif mixed_files:
            # Use mixed files (they contain both benign and attack samples)
            attack_files = mixed_files
            print(f"Found {len(mixed_files)} mixed CSV files (with benign samples)")
            print(f"Found {len(original_files)} original attack CSV files (ignored)")
        else:
            # Fall back to original files
            attack_files = original_files
            print(f"Found {len(original_files)} attack CSV files")
            if len(mixed_files) == 0:
                print("  (Tip: Run prepare_realistic_data.py to create mixed datasets)")
                print("  (Tip: Run create_heterogeneous_clients.py to create heterogeneous clients)")
        
        print(f"Found {len(benign_files)} benign CSV files")
        
        return [str(f) for f in attack_files], [str(f) for f in benign_files]
    
    def prepare_test_data(self, test_csv_path: str) -> tuple:
        """
        Prepare test dataset.
        
        Args:
            test_csv_path: Path to test CSV file
            
        Returns:
            X_test, y_test
        """
        print(f"\nLoading test data from {test_csv_path}")
        df_test = load_client_data(test_csv_path)
        df_test = prepare_labels(df_test)
        X_test = prepare_features(df_test)
        y_test = df_test['label']
        
        print(f"Test set: {len(X_test)} samples, {X_test.shape[1]} features")
        print(f"Test set class distribution: {(y_test == 0).sum()} benign, {(y_test == 1).sum()} attack")
        
        return X_test, y_test
    
    def setup_clients(self, attack_files: List[str], benign_files: List[str], num_clients: Optional[int] = None) -> None:
        """
        Set up federated clients from CSV files.
        
        Args:
            attack_files: List of attack CSV file paths
            benign_files: List of benign CSV file paths (for mixing)
            num_clients: Number of clients to use (None = use all)
        """
        if num_clients is not None:
            attack_files = attack_files[:num_clients]
        
        # Exclude problematic clients (e.g., those with 100% accuracy despite corruption)
        excluded_clients = ['heartbleed']  # Exclude heartbleed compromised client with 100% accuracy
        original_count = len(attack_files)
        attack_files = [f for f in attack_files if not any(excluded in Path(f).name.lower() for excluded in excluded_clients)]
        if len(attack_files) < original_count:
            print(f"\n⚠️  Excluding {original_count - len(attack_files)} problematic client(s): {excluded_clients}")
        
        print(f"\nSetting up {len(attack_files)} federated clients...")
        
        for idx, attack_file in enumerate(attack_files):
            stem = Path(attack_file).stem
            if self.use_shared_val_ref:
                client_id = stem  # client_01 … client_12
            else:
                client_id = f"client_{idx+1}_{stem}"
            print(f"  Setting up {client_id}...")
            
            # Extract quality from filename if present (e.g., 'high_quality_client_1.csv')
            import re
            quality_match = re.search(r'(high|medium|low|compromised)_quality', Path(attack_file).name)
            client_quality = quality_match.group(1) if quality_match else 'unknown'
            
            # Debug: Check if quality was extracted
            if 'compromised' in Path(attack_file).name.lower() and client_quality != 'compromised':
                # Try alternative pattern: client_X_compromised_...
                alt_match = re.search(r'_compromised_', Path(attack_file).name)
                if alt_match:
                    client_quality = 'compromised'
            
            # CRITICAL FIX: Find original clean source file for validation
            # For heterogeneous clients (compromised/low/medium), use clean validation set
            clean_validation_source = None
            use_shared_val_ref = False
            if self.use_shared_val_ref and self.shared_val_ref is not None:
                # Natural partition (iomt_natural): shared D_val, no planted quality tiers
                client_quality = "unknown"
                clean_validation_source = str(self.shared_val_ref)
                use_shared_val_ref = True
            # FIX: Only use clean validation for medium/low clients, NOT compromised
            # Compromised clients should use CORRUPTED validation to get low trust scores
            elif client_quality in ['low', 'medium']:
                # Extract source file name from heterogeneous client filename
                # Pattern: client_X_compromised_mixed_Y.csv -> mixed_Y.csv
                # Handle cases like: mixed_ssh_patator-new.csv (with hyphens)
                filename = Path(attack_file).name
                # Find "mixed_" followed by everything until ".csv"
                source_match = re.search(r'mixed_[^.]+\.csv', filename)
                if source_match:
                    source_filename = source_match.group(0)
                    # Look for original source in data/CSVs directory
                    original_source = Path('data/CSVs') / source_filename
                    if original_source.exists():
                        clean_validation_source = str(original_source)
                        print(f"\n    ✅ FOUND CLEAN VALIDATION SOURCE for {client_id}: {source_filename}")
                    else:
                        print(f"\n    ⚠️  Clean source not found: {source_filename} (looking in data/CSVs/)")
                else:
                    # IoMT hospital CSVs (hospital_*_*_quality_iomt.csv) have no CICIDS-style
                    # mixed_*.csv clean source — validation uses the client file as-is. Not an error.
                    if re.search(r'hospital_.*_(high|medium|low)_quality_(iomt|wustl_ehms)', filename):
                        pass
                    else:
                        print(f"\n    ⚠️  Could not extract source filename from: {filename}")
            elif client_quality == 'compromised':
                # CRITICAL: Compromised clients MUST use CORRUPTED validation
                # Training data is corrupted → model learns wrong patterns
                # Corrupted validation → model performs poorly (even on corrupted data) → LOW trust score
                # Using clean validation can give high trust if model somehow generalizes
                # Using corrupted validation ensures trust reflects the corrupted quality
                clean_validation_source = None  # Use corrupted validation data
                print(f"\n    ✅ COMPROMISED CLIENT: Using CORRUPTED validation (no clean source)")
                print(f"        Training: CORRUPTED → Model learns wrong patterns")
                print(f"        Validation: CORRUPTED → Model performs poorly → LOW trust")
            else:
                print(f"    (High-quality client - no clean validation needed)")
            
            client = FederatedClient(
                client_id=client_id,
                data_path=attack_file,
                model_type=self.model_type,
                random_state=self.random_state,
                client_quality=client_quality,
                clean_validation_source=clean_validation_source,
                use_shared_val_ref=use_shared_val_ref,
                performance_signal=self.performance_signal,
                uniform_contextual_prior=self.uniform_contextual_prior,
            )
            
            # Load and prepare data
            client.load_data()
            
            # Optionally mix in some benign data (realistic hospital mix)
            # For simplicity, we'll use the attack data as-is
            # In a real scenario, you might sample benign data and mix it
            
            self.clients.append(client)
        
        print(f"Successfully set up {len(self.clients)} clients")
        if self.use_shared_val_ref:
            fingerprints = {
                (
                    len(c.y_val),
                    float(c.y_val.mean()),
                    tuple(c.X_val.columns.tolist()),
                )
                for c in self.clients
            }
            if len(fingerprints) != 1:
                raise RuntimeError(f"Shared D_val mismatch across clients: {fingerprints}")
            n_val, rate, _ = next(iter(fingerprints))
            print(f"  ✅ Identical full-file D_val for all clients (n={n_val}, attack_rate={rate:.4f})")
            # Train must be 100% of client CSV (no local holdout)
            for c in self.clients:
                n_csv = len(load_client_data(c.data_path))
                if len(c.X_train) != n_csv:
                    raise RuntimeError(
                        f"{c.client_id}: expected full-file train ({n_csv}), got {len(c.X_train)}"
                    )
            if getattr(self, "_pending_late_cfg", None) is not None:
                from adversary.late_compromise import LateCompromiseController
                self.late_compromise = LateCompromiseController(
                    self._pending_late_cfg,
                    [c.client_id for c in self.clients],
                )
                print(f"  Late compromise: {self.late_compromise.summarize()}")
    
    def train_clients(self) -> None:
        """Train all client models."""
        print("\n" + "="*60)
        print("Training Local Models at Each Client")
        print("="*60)
        
        for client in self.clients:
            print(f"\nTraining {client.client_id}...")
            client.train()
            client.evaluate()
            client.compute_trust()
            
            info = client.get_info()
            print(f"  Train samples: {info['train_samples']}")
            print(f"  Val samples: {info['val_samples']}")
            print(f"  Val Accuracy: {info['val_accuracy']:.4f}")
            print(f"  Trust Score: {info['trust_score']:.4f}")
        
        # Collect model updates
        self.client_updates = [client.get_model_update() for client in self.clients]
        
        print(f"\n✓ All {len(self.clients)} clients trained successfully")
    
    def approach_1_centralized(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Approach 1: Centralized Learning.
        
        In heterogeneous client scenarios, this approach is weak because:
        - Cannot filter bad data from low-quality clients
        - All data (good + bad) is combined and trained together
        - Bad clients degrade the global model
        - No concept of dynamic trust (single round only)
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*60)
        print("Approach 1: Centralized Learning")
        print("="*60)
        print("Note: In heterogeneous scenarios, Centralized is weak because")
        print("      it cannot filter bad data from low-quality clients.")
        print("      Also, Centralized has no dynamic trust concept (single round).")
        
        # Combine all training data from all clients
        print("\nCombining all client data (good + bad)...")
        all_X_train = []
        all_y_train = []
        client_qualities = []
        
        for client in self.clients:
            all_X_train.append(client.X_train)
            all_y_train.append(client.y_train)
            
            # Detect client quality from filename
            client_path = Path(client.data_path)
            if 'high_quality' in client_path.name.lower():
                client_qualities.append('high')
            elif 'medium_quality' in client_path.name.lower():
                client_qualities.append('medium')
            elif 'low_quality' in client_path.name.lower():
                client_qualities.append('low')
            elif 'compromised' in client_path.name.lower():
                client_qualities.append('compromised')
            else:
                client_qualities.append('unknown')
        
        # Ensure all DataFrames have the same columns before concatenating
        all_columns = set()
        for df in all_X_train:
            all_columns.update(df.columns)
        all_columns = sorted(list(all_columns))
        
        # Reindex all DataFrames to have the same columns (missing columns will be NaN)
        all_X_train_aligned = [df.reindex(columns=all_columns) for df in all_X_train]
        
        X_combined = pd.concat(all_X_train_aligned, ignore_index=True)
        y_combined = pd.concat(all_y_train, ignore_index=True)
        
        # Fill any NaN values that may have been created by missing columns
        X_combined = X_combined.fillna(0)
        
        print(f"Combined dataset: {len(X_combined)} samples")
        print(f"Class distribution: {(y_combined == 0).sum()} benign, {(y_combined == 1).sum()} attack")
        
        # Report client quality distribution
        if any(q != 'unknown' for q in client_qualities):
            quality_counts = {}
            for q in client_qualities:
                quality_counts[q] = quality_counts.get(q, 0) + 1
            print(f"\nClient quality distribution:")
            for quality, count in sorted(quality_counts.items()):
                print(f"  {quality}: {count} clients")
            print("  ⚠️  Centralized cannot filter bad data - all clients contribute equally!")
            print("  ⚠️  Centralized has no dynamic trust - single round only!")
        
        # Train centralized model
        print("\nTraining centralized model on combined data...")
        centralized_model, train_metrics = train_local_model(
            X_combined,
            y_combined,
            model_type=self.model_type,
            random_state=self.random_state
        )
        
        print(f"Training accuracy: {train_metrics['train_accuracy']:.4f}")
        
        # Align test set features with training features
        print("\nAligning test set features with training features...")
        X_test_aligned = X_test.reindex(columns=X_combined.columns, fill_value=0)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = evaluate_model_on_test(centralized_model, X_test_aligned, y_test)
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        # Add note about weakness in heterogeneous scenarios
        if any(q in ['low', 'compromised'] for q in client_qualities):
            print("\n⚠️  Centralized weakness: Bad clients degrade model performance")
            print("    Cannot filter or weight clients by quality")
            print("    No dynamic trust - single round only (cannot adapt)")
        
        return results
    
    def approach_2_federated_equal_weight(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Approach 2: Standard Federated Learning (equal weights - FedAvg).
        
        In heterogeneous client scenarios, this approach is weak because:
        - All clients have equal weight (1/N)
        - Cannot differentiate good vs bad clients
        - Low-quality clients have same influence as high-quality
        - Weights stay constant (no adaptation)
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*60)
        print("Approach 2: Standard Federated Learning (Equal Weights - FedAvg)")
        print("="*60)
        print("Note: In heterogeneous scenarios, FedAvg is weak because")
        print("      all clients have equal weight - cannot prioritize good clients.")

        if self.dynamic_adversary and self.num_rounds <= 1:
            self._apply_static_adversary_before_aggregation()
            for client in self.clients:
                client.train()
                client.evaluate()
                client.compute_trust()
            self.client_updates = [client.get_model_update() for client in self.clients]

        aggregator = FedAvgAggregator(model_type=self.model_type)
        global_model = None
        rounds = max(1, int(self.num_rounds))
        self.round_logs = []
        # Upload poison (e.g. sign_flip) is invisible under sample retraining; use params.
        use_retrain = not self._upload_poison_requires_param_agg()
        if not use_retrain:
            print("FedAvg: parameter averaging (sign_flip active; skip sample retrain)")

        for r in range(1, rounds + 1):
            if rounds > 1:
                print(f"\n--- FedAvg round {r}/{rounds} ---")
                self._apply_mid_training_adversary(r)
            skipped_ids = set()
            for client in self.clients:
                if (
                    getattr(self, "late_compromise", None) is not None
                    and self.late_compromise.should_skip_participation(
                        client.client_id, r
                    )
                ):
                    skipped_ids.add(client.client_id)
                    continue
                client.train()
                client.evaluate()
                client.compute_trust()
            self.client_updates = [
                u
                for u in self._collect_client_updates(
                    round_num=r, include_data=use_retrain
                )
                if u.get("client_id") not in skipped_ids
            ]
            # Report client trust scores to show heterogeneity (first/last round)
            if r == 1 or r == rounds:
                trust_scores = [update.get("trust", 0) for update in self.client_updates]
                if trust_scores:
                    print(f"Client trust (val) range: {min(trust_scores):.4f}–{max(trust_scores):.4f}")
                    print(f"  FedAvg weight per client: {1.0/len(self.client_updates):.4f}")
            print(
                "Aggregating with equal weights "
                f"({'retraining' if use_retrain else 'parameter average'})..."
            )
            global_model = aggregator.aggregate(
                self.client_updates, use_retraining=use_retrain
            )
            n_up = max(len(self.client_updates), 1)
            equal_w = 1.0 / float(n_up)
            if self.use_shared_val_ref:
                attackers = set()
                if getattr(self, "late_compromise", None) is not None:
                    attackers = set(self.late_compromise.attacker_ids)
                client_trust = {}
                up_ids = {u.get("client_id") for u in self.client_updates}
                for client in self.clients:
                    cid = client.client_id
                    try:
                        v_i = float(client.get_validation_performance_metric())
                    except Exception:
                        v_i = None
                    client_trust[cid] = self._client_trust_log_row(
                        client,
                        r,
                        alpha=float(equal_w if cid in up_ids else 0.0),
                        attackers=attackers,
                        trust_value=float(client.trust_score or 0.0),
                        v_i=v_i,
                    )
                self.round_logs.append({
                    "round": r,
                    "client_trust": client_trust,
                    "compromise_active": bool(
                        self.late_compromise
                        and self.late_compromise.is_compromised_round(r)
                    ),
                    "attack_active": bool(
                        self.late_compromise
                        and self.late_compromise.is_attack_active(r)
                    ),
                    "aggregator": "fedavg",
                })

        trust_scores = [update.get("trust", 0) for update in self.client_updates]
        trust_range = (max(trust_scores) - min(trust_scores)) if trust_scores else 0.0

        # Align test set features with training features (use aggregator's training columns)
        if hasattr(aggregator, "training_feature_columns") and aggregator.training_feature_columns:
            X_test_aligned = X_test.reindex(columns=aggregator.training_feature_columns, fill_value=0)
        elif len(self.clients) > 0:
            all_train_columns = set()
            for client in self.clients:
                all_train_columns.update(client.X_train.columns)
            all_train_columns = sorted(list(all_train_columns))
            X_test_aligned = X_test.reindex(columns=all_train_columns, fill_value=0)
        else:
            X_test_aligned = X_test

        results = evaluate_model_on_test(global_model, X_test_aligned, y_test)
        results["round_logs"] = list(self.round_logs)

        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")

        if trust_scores and trust_range > 0.3:
            print("\n⚠️  FedAvg weakness: Equal weighting doesn't differentiate client quality")

        return results
    
    def _collect_client_updates(self, round_num: Optional[int] = None, include_data: bool = False):
        """Gather model updates; apply late-compromise sign-flip when configured."""
        updates = []
        for client in self.clients:
            if round_num is None:
                u = client.get_model_update(include_data=include_data)
            else:
                u = client.get_model_update(round_num=round_num, include_data=include_data)
            if getattr(self, "late_compromise", None) is not None and round_num is not None:
                u = self.late_compromise.maybe_sign_flip_update(
                    client.client_id, u, round_num
                )
            updates.append(u)
        return updates

    def _upload_poison_requires_param_agg(self) -> bool:
        """True when upload-borne poison would be erased by sample retraining / raw-model ensemble.

        Path AGG / sign_flip: flipped values live in update['parameters'], not in client.model
        and not in clean X_train. B0/B2/Rm must all aggregate in parameter space.
        """
        lc = getattr(self, "late_compromise", None)
        return bool(lc is not None and lc.uses_sign_flip())

    def _client_trust_log_row(
        self,
        client,
        round_num: int,
        *,
        alpha: float,
        attackers: set,
        trust_value: float,
        v_i,
    ) -> dict:
        """Per-client round-log fields including Path STRAG skip audit."""
        lc = getattr(self, "late_compromise", None)
        reason = (
            lc.participation_skip_reason(client.client_id, round_num)
            if lc is not None
            else None
        )
        stragglers = set(getattr(lc, "straggler_ids", set()) or set()) if lc else set()
        return {
            "T": float(trust_value),
            "V": v_i,
            "alpha": float(alpha),
            "is_attacker": client.client_id in attackers,
            "is_straggler": client.client_id in stragglers,
            "skip_reason": reason,
        }

    def _apply_mid_training_adversary(self, round_num: int) -> None:
        """
        Shared mid-training adversary schedule for multi-round runs.

        Natural partition (iomt_natural): LateCompromiseController always wins when set.
        Legacy planted-tier schedule: delayed poisoning / gradual corruption / recovery.
        """
        # Always sync late-compromise when configured — independent of dynamic_adversary.
        if getattr(self, "late_compromise", None) is not None:
            report = self.late_compromise.sync_round(round_num, self.clients)
            if report.get("attack_active") and round_num == self.late_compromise.t_a:
                print(
                    f"[LateCompromise] t_a={self.late_compromise.t_a} "
                    f"mode={self.late_compromise.cfg.poison_mode}: {report.get('flipped')}"
                )
            return

        if not self.dynamic_adversary:
            return

        if round_num == 4:
            print("\n[Adversary] Round 4: delayed poisoning on low/compromised clients")
            compromised_clients = [
                c for c in self.clients
                if 'low' in c.client_quality.lower() or 'compromised' in c.client_quality.lower()
            ][:3]
            for client in compromised_clients:
                client.inject_dynamic_change(round_num, change_type='degrade', severity=0.4)

        if round_num in (5, 6, 7):
            severity = 0.12 * (round_num - 4)  # 0.12, 0.24, 0.36 gradual escalation
            gradual_targets = [
                c for c in self.clients if 'medium' in c.client_quality.lower()
            ][:2]
            if gradual_targets:
                print(
                    f"\n[Adversary] Round {round_num}: gradual corruption "
                    f"(severity={severity:.2f}) on medium-tier clients"
                )
            for client in gradual_targets:
                client.inject_dynamic_change(round_num, change_type='degrade', severity=severity)

        if round_num == 6:
            print("\n[Adversary] Round 6: partial quality recovery on medium clients (evasion mimic)")
            improving_clients = [
                c for c in self.clients if 'medium' in c.client_quality.lower()
            ][:2]
            for client in improving_clients:
                client.inject_dynamic_change(round_num, change_type='improve', severity=0.35)

    def _apply_static_adversary_before_aggregation(self) -> None:
        """One-shot delayed poisoning for single-round FedAvg (round-4 equivalent)."""
        if not self.dynamic_adversary:
            return
        print("\n[Adversary] Pre-aggregation: delayed poisoning (FedAvg single-round schedule)")
        for client in self.clients:
            if 'low' in client.client_quality.lower() or 'compromised' in client.client_quality.lower():
                client.inject_dynamic_change(4, change_type='degrade', severity=0.4)

    def _prepare_trust_phase(self) -> None:
        """
        Reset client training data and trust histories before TrustFed.

        Centralized/FedAvg baselines may call inject_dynamic_change on client data;
        without restoration, TrustFed would inherit a different initial state than
        trust-only drivers (e.g., lambda sweeps), breaking reproducibility.
        """
        restored = 0
        for client in self.clients:
            if client.restore_training_data_from_original():
                restored += 1
        if self.trust_manager is not None:
            self.trust_manager.reset_all_histories()
        # FedAvg/baseline paths use np.random without a local Generator; re-seed so
        # TrustFed's mid-training adversary matches trust-only replication runs.
        np.random.seed(self.random_state + 10_000)
        # Natural: restore wiped poison — re-apply if we are already past t_a.
        # Callers should sync_round each round; here we re-apply for round==t_a as safe default
        # when switching phases mid-experiment after FedAvg mutated data.
        if getattr(self, "late_compromise", None) is not None:
            # Re-apply for "current" compromise state using t_a as active marker once past warm-up
            # (trust phase typically starts from round 1 again; sync_round(1) clears poison).
            self.late_compromise.sync_round(1, self.clients)
        print(
            f"\n[Trust phase] Restored training data for {restored}/{len(self.clients)} clients; "
            "cleared trust histories; reset RNG before TrustFed."
        )

    def approach_3_trust_aware(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Approach 3: Trust-Aware Federated Learning (proposed method).
        
        Supports both single-round (static trust) and multi-round (adaptive trust) modes.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Results dictionary
        """
        self._prepare_trust_phase()
        if self.privacy_mode == "strong_v1":
            if self.num_rounds <= 1:
                print("Warning: strong_v1 expects num_rounds > 1; using multi-round DP-SAG path")
            return self._approach_3_multi_round_strong_v1(X_test, y_test)
        if self.num_rounds > 1:
            return self._approach_3_multi_round(X_test, y_test)
        else:
            return self._approach_3_single_round(X_test, y_test)

    def approach_4_fedprox(self, X_test: pd.DataFrame, y_test: pd.Series, mu: float = 1e-3, local_epochs: int = 1) -> Dict[str, Any]:
        """
        Approach 4: FedProx-style baseline (logistic regression only).

        Notes:
        - This codebase's primary FL baseline uses server-side retraining on aggregated samples.
          FedProx is natively defined for parameter-update-based FL under non-IID data.
        - Here we provide a parameter-update-based logistic regression baseline that (a) initializes
          clients from the global model each round and (b) increases proximal-like regularization
          via L2 penalty strength (mu) during local updates.
        """
        if self.model_type != 'logistic_regression':
            raise ValueError("FedProx baseline is implemented only for logistic_regression in this project.")

        import numpy as np
        # Import directly to avoid triggering src/__init__.py (which imports visualization).
        from evaluation import compute_metrics

        print("\n" + "=" * 60)
        print("Approach 4: FedProx-Style Federated Learning (Logistic Regression)")
        print("=" * 60)
        print(f"Config: rounds={self.num_rounds}, local_epochs={local_epochs}, mu={mu}")

        # Canonical feature schema (union across clients)
        all_cols = set()
        for c in self.clients:
            all_cols.update(c.X_train.columns)
        all_cols = sorted(list(all_cols))

        # Align test
        X_test_aligned = X_test.reindex(columns=all_cols, fill_value=0)

        # Helper: sigmoid with numeric stability
        def _sigmoid(z: np.ndarray) -> np.ndarray:
            z = np.clip(z, -50, 50)
            return 1.0 / (1.0 + np.exp(-z))

        # Convert test to numpy and add bias column
        X_test_np = X_test_aligned.to_numpy(dtype=float, copy=False)
        y_test_np = y_test.to_numpy(dtype=int, copy=False)
        X_test_b = np.hstack([X_test_np, np.ones((X_test_np.shape[0], 1), dtype=float)])

        # Initialize global weights (including bias as last component)
        d = len(all_cols)
        w_global = np.zeros((d + 1,), dtype=float)

        # Multi-round FedProx loop (pure numpy SGD with proximal term)
        for r in range(1, max(1, self.num_rounds) + 1):
            self._apply_mid_training_adversary(r)
            w_locals = []
            weights = []

            for client in self.clients:
                Xc_df = client.X_train.reindex(columns=all_cols, fill_value=0)
                yc_s = client.y_train

                # Cap per-client samples for runtime stability
                max_samples_per_client = 15000
                if len(Xc_df) > max_samples_per_client:
                    from sklearn.utils import resample
                    Xc_df, yc_s = resample(
                        Xc_df, yc_s,
                        n_samples=max_samples_per_client,
                        random_state=self.random_state + r,
                        stratify=yc_s if len(np.unique(yc_s)) > 1 else None
                    )

                Xc = Xc_df.to_numpy(dtype=float, copy=False)
                yc = yc_s.to_numpy(dtype=int, copy=False)
                Xc_b = np.hstack([Xc, np.ones((Xc.shape[0], 1), dtype=float)])

                # Skip degenerate single-class clients for this baseline
                if len(np.unique(yc)) < 2:
                    continue

                w = w_global.copy()
                lr = 0.05
                batch_size = 1024

                rng = np.random.default_rng(self.random_state + r)
                for _ in range(max(1, local_epochs)):
                    idx = rng.permutation(Xc_b.shape[0])
                    for start in range(0, len(idx), batch_size):
                        batch_idx = idx[start:start + batch_size]
                        Xb = Xc_b[batch_idx]
                        yb = yc[batch_idx]

                        # Avoid BLAS-backed matmul (can raise hard FPE on some macOS builds)
                        z = (Xb * w).sum(axis=1)
                        p = _sigmoid(z)
                        err = (p - yb)
                        grad = (Xb * err[:, None]).mean(axis=0)

                        # Proximal term towards global weights (do not penalize bias)
                        prox = w - w_global
                        prox[-1] = 0.0
                        grad = grad + mu * prox

                        w = w - lr * grad

                w_locals.append(w)
                weights.append(len(Xc_b))

            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

            if w_locals:
                w_global = np.sum([w_i * w_l for w_i, w_l in zip(weights, w_locals)], axis=0)

        # Evaluate
        y_proba = _sigmoid((X_test_b * w_global).sum(axis=1))
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_test_np, y_pred)
        metrics['y_pred_proba'] = y_proba.tolist()
        metrics['y_pred'] = y_pred.tolist()
        metrics['y_true'] = y_test_np.tolist()
        return metrics

    def approach_5_coordinate_median(self, X_test: pd.DataFrame, y_test: pd.Series, local_epochs: int = 1) -> Dict[str, Any]:
        """
        Path AGG Rm: coordinate-wise median on logistic LR uploads.

        Uses the same client.train / late-compromise / sign_flip path as FedAvg
        (not the legacy standalone SGD loop), so upload poison is visible.
        Equal α among uploaders is logged for α_mal context (not trust weights).
        """
        if self.model_type != 'logistic_regression':
            raise ValueError("Coordinate-wise median baseline is implemented only for logistic_regression in this project.")

        from evaluation import evaluate_model_on_test

        print("\n" + "=" * 60)
        print("Approach 5: Robust Aggregation (Coordinate-wise Median, Logistic Regression)")
        print("=" * 60)
        print(f"Config: rounds={self.num_rounds} (local_epochs arg ignored; uses client.train)")

        self.round_logs = []
        aggregator = CoordinateMedianAggregator(model_type=self.model_type)
        global_model = None
        rounds = max(1, int(self.num_rounds))

        for r in range(1, rounds + 1):
            print(f"\n--- Median round {r}/{rounds} ---")
            self._apply_mid_training_adversary(r)

            skipped_ids = set()
            for client in self.clients:
                if (
                    getattr(self, "late_compromise", None) is not None
                    and self.late_compromise.should_skip_participation(
                        client.client_id, r
                    )
                ):
                    skipped_ids.add(client.client_id)
                    reason = self.late_compromise.participation_skip_reason(
                        client.client_id, r
                    )
                    print(f"  [{reason or 'skip'}] {client.client_id} withholds this round")
                    continue
                client.train()
                client.evaluate()
                client.compute_trust()

            updates = [
                u
                for u in self._collect_client_updates(
                    round_num=r, include_data=True
                )
                if u.get("client_id") not in skipped_ids
            ]
            if not updates:
                print("  Warning: no uploads this round; keeping previous global model")
                continue

            self.client_updates = updates
            print(f"Aggregating {len(updates)} uploads via coordinate-wise median...")
            global_model = aggregator.aggregate(updates)
            weights = aggregator.get_client_weights(updates)

            # Natural partition: per-round T/V/α (α = equal among uploaders)
            if self.use_shared_val_ref:
                attackers = set()
                if getattr(self, "late_compromise", None) is not None:
                    attackers = set(self.late_compromise.attacker_ids)
                client_trust = {}
                for client in self.clients:
                    cid = client.client_id
                    try:
                        v_i = float(client.get_validation_performance_metric())
                    except Exception:
                        v_i = None
                    client_trust[cid] = self._client_trust_log_row(
                        client,
                        r,
                        alpha=float(weights.get(cid, 0.0)),
                        attackers=attackers,
                        trust_value=float(client.trust_score or 0.0),
                        v_i=v_i,
                    )
                self.round_logs.append({
                    "round": r,
                    "client_trust": client_trust,
                    "compromise_active": bool(
                        self.late_compromise
                        and self.late_compromise.is_compromised_round(r)
                    ),
                    "attack_active": bool(
                        self.late_compromise
                        and self.late_compromise.is_attack_active(r)
                    ),
                    "aggregator": "coordinate_median",
                })

        if global_model is None:
            raise RuntimeError("Coordinate median produced no global model")

        if getattr(aggregator, "training_feature_columns", None):
            X_test_aligned = X_test.reindex(
                columns=aggregator.training_feature_columns, fill_value=0
            )
        elif len(self.clients) > 0:
            all_train_columns = set()
            for client in self.clients:
                all_train_columns.update(client.X_train.columns)
            X_test_aligned = X_test.reindex(
                columns=sorted(all_train_columns), fill_value=0
            )
        else:
            X_test_aligned = X_test

        results = evaluate_model_on_test(global_model, X_test_aligned, y_test)
        results["round_logs"] = list(self.round_logs)
        print(f"\nTest F1-Score: {results['f1_score']:.4f}")
        return results

    def approach_6_krum(self, X_test: pd.DataFrame, y_test: pd.Series, local_epochs: int = 1, f: Optional[int] = None) -> Dict[str, Any]:
        """
        Approach 6: Krum robust aggregation on client weight vectors (logistic regression only).

        Uses the same local SGD loop as coordinate-wise median but selects one client's
        weight vector per round via the Krum score (Blanchard et al.).
        This is a parameter-robust baseline; it does not address corrupted local data
        when gradients remain plausible (our threat model).
        """
        if self.model_type != 'logistic_regression':
            raise ValueError("Krum baseline is implemented only for logistic_regression in this project.")

        import numpy as np
        from evaluation import compute_metrics

        def _krum_vector(w_locals: list, f_param: Optional[int]) -> np.ndarray:
            W = np.stack(w_locals, axis=0)
            n = W.shape[0]
            if n <= 2:
                return np.mean(W, axis=0)
            if f_param is None:
                f_param = max(0, (n - 3) // 2)
            m = n - f_param - 2
            m = max(1, min(m, n - 1))
            scores = []
            for i in range(n):
                dists = sorted(
                    float(np.linalg.norm(W[i] - W[j]))
                    for j in range(n) if j != i
                )
                scores.append(sum(dists[:m]))
            idx = int(np.argmin(scores))
            return W[idx]

        print("\n" + "=" * 60)
        print("Approach 6: Krum Aggregation (Logistic Regression)")
        print("=" * 60)
        print(f"Config: rounds={self.num_rounds}, local_epochs={local_epochs}")

        all_cols = set()
        for c in self.clients:
            all_cols.update(c.X_train.columns)
        all_cols = sorted(list(all_cols))

        def _sigmoid(z: np.ndarray) -> np.ndarray:
            z = np.clip(z, -50, 50)
            return 1.0 / (1.0 + np.exp(-z))

        X_test_aligned = X_test.reindex(columns=all_cols, fill_value=0)
        X_test_np = X_test_aligned.to_numpy(dtype=float, copy=False)
        y_test_np = y_test.to_numpy(dtype=int, copy=False)
        X_test_b = np.hstack([X_test_np, np.ones((X_test_np.shape[0], 1), dtype=float)])

        d = len(all_cols)
        w_global = np.zeros((d + 1,), dtype=float)

        for r in range(1, max(1, self.num_rounds) + 1):
            self._apply_mid_training_adversary(r)
            w_locals = []

            for client in self.clients:
                Xc_df = client.X_train.reindex(columns=all_cols, fill_value=0)
                yc_s = client.y_train

                max_samples_per_client = 15000
                if len(Xc_df) > max_samples_per_client:
                    from sklearn.utils import resample
                    Xc_df, yc_s = resample(
                        Xc_df, yc_s,
                        n_samples=max_samples_per_client,
                        random_state=self.random_state + r,
                        stratify=yc_s if len(np.unique(yc_s)) > 1 else None
                    )

                Xc = Xc_df.to_numpy(dtype=float, copy=False)
                yc = yc_s.to_numpy(dtype=int, copy=False)
                Xc_b = np.hstack([Xc, np.ones((Xc.shape[0], 1), dtype=float)])

                if len(np.unique(yc)) < 2:
                    continue

                w = w_global.copy()
                lr = 0.05
                batch_size = 1024
                rng = np.random.default_rng(self.random_state + r)
                for _ in range(max(1, local_epochs)):
                    idx = rng.permutation(Xc_b.shape[0])
                    for start in range(0, len(idx), batch_size):
                        batch_idx = idx[start:start + batch_size]
                        Xb = Xc_b[batch_idx]
                        yb = yc[batch_idx]
                        z = (Xb * w).sum(axis=1)
                        p = _sigmoid(z)
                        err = (p - yb)
                        grad = (Xb * err[:, None]).mean(axis=0)
                        w = w - lr * grad

                w_locals.append(w)

            if w_locals:
                w_global = _krum_vector(w_locals, f)

        y_proba = _sigmoid((X_test_b * w_global).sum(axis=1))
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_test_np, y_pred)
        metrics['y_pred_proba'] = y_proba.tolist()
        metrics['y_pred'] = y_pred.tolist()
        metrics['y_true'] = y_test_np.tolist()
        return metrics
    
    def _approach_3_single_round(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Single-round trust-aware federated learning (static trust).
        
        In heterogeneous client scenarios, this approach is strong because:
        - Clients weighted by trust scores (quality-based)
        - High-trust clients contribute more
        - Low-trust clients have minimal influence
        """
        print("\n" + "="*60)
        print("Approach 3: Trust-Aware Federated Learning (Single Round)")
        print("="*60)
        print("Note: Trust-Aware is strong in heterogeneous scenarios because")
        print("      it weights clients by quality (trust scores).")
        
        # Display trust scores
        print("\nClient Trust Scores (validation accuracy):")
        trust_scores = []
        for update in self.client_updates:
            trust = update.get('trust', 0)
            trust_scores.append(trust)
            print(f"  {update['client_id']}: {trust:.4f}")
        
        if trust_scores:
            trust_range = max(trust_scores) - min(trust_scores)
            print(f"\nTrust score range: {min(trust_scores):.4f} - {max(trust_scores):.4f} (range: {trust_range:.4f})")
            if trust_range > 0.3:
                print("  ✅ High heterogeneity - Trust-Aware can leverage quality differences!")
        
        # Use TrustAwareAggregator with retraining (true trust-aware FedAvg)
        print("\nAggregating client models with trust-weighted retraining (true trust-aware FedAvg)...")
        print("  Clients weighted by trust scores (quality-based weighting)")
        # Exclude very low-trust clients (helps under partial compromise)
        aggregator = TrustAwareAggregator(model_type=self.model_type, min_trust=0.6)
        global_model = aggregator.aggregate(
            self.client_updates, use_retraining=self.trust_use_retraining
        )
        
        # Display client weights
        weights = aggregator.get_client_weights(self.client_updates)
        print("\nClient Weights in Aggregation (trust-weighted):")
        for client_id, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            # Find corresponding trust score
            trust = next((u.get('trust', 0) for u in self.client_updates if u.get('client_id') == client_id), 0)
            print(f"  {client_id}: {weight:.4f} (trust: {trust:.4f})")
        
        # Show comparison with equal weights
        equal_weight = 1.0 / len(self.client_updates)
        print(f"\nComparison:")
        print(f"  Equal weight (FedAvg): {equal_weight:.4f} per client")
        print(f"  Trust-weighted: High-trust clients get more, low-trust get less")
        
        # Evaluate global model
        # CRITICAL: Align test set features to the server-side aggregated training columns.
        # Using a single client's columns can cause feature-name mismatches (and invalid comparisons).
        if hasattr(aggregator, 'training_feature_columns') and aggregator.training_feature_columns:
            X_test_aligned = X_test.reindex(columns=aggregator.training_feature_columns, fill_value=0)
        elif len(self.clients) > 0:
            # Fallback: union of all client training columns (stable, avoids missing features)
            all_train_columns = set()
            for client in self.clients:
                all_train_columns.update(client.X_train.columns)
            all_train_columns = sorted(list(all_train_columns))
            X_test_aligned = X_test.reindex(columns=all_train_columns, fill_value=0)
        else:
            X_test_aligned = X_test
        
        results = evaluate_model_on_test(global_model, X_test_aligned, y_test)
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        # Add note about strength
        if trust_scores and trust_range > 0.3:
            print("\n✅ Trust-Aware strength: Quality-based weighting prioritizes good clients")
            print("    High-trust clients contribute more, low-trust clients contribute less")
        
        return results
    
    def _approach_3_multi_round_strong_v1(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Multi-round TrustFed with DP-SAG on gradient statistics (no row upload)."""
        import logging
        logger = logging.getLogger(__name__)

        print("\n" + "=" * 60)
        print(
            f"Approach 3: Trust-Aware FL — Strong v1 DP-SAG "
            f"({self.num_rounds} rounds, no feature rows to server)"
        )
        print("=" * 60)

        cfg = DPSAGConfig(
            clip_norm=float(self.privacy_strong_v1.get("clip_norm", 1.0)),
            noise_multiplier=float(self.privacy_strong_v1.get("noise_multiplier", 0.0)),
            delta=float(self.privacy_strong_v1.get("delta", 1e-5)),
            learning_rate=float(self.privacy_strong_v1.get("learning_rate", 0.05)),
            secure_aggregation=bool(self.privacy_strong_v1.get("secure_aggregation", True)),
            trust_beta=float(self.privacy_strong_v1.get("trust_beta", 0.8)),
        )
        batch_size = int(self.privacy_strong_v1.get("local_batch_size", 1000))
        optimizer = str(self.privacy_strong_v1.get("privacy_optimizer", "gradient")).lower()
        inner_steps = int(self.privacy_strong_v1.get("inner_steps", 1))
        cfg.hessian_reg = float(self.privacy_strong_v1.get("hessian_reg", 0.01))
        cfg.inner_steps = inner_steps
        rng = np.random.default_rng(self.random_state + 77_000)
        print(f"  Optimizer: {optimizer}, inner_steps={inner_steps}, batch={batch_size}, lr={cfg.learning_rate}")

        all_cols = sorted({c for client in self.clients for c in client.X_train.columns})
        w_global = np.zeros(len(all_cols) + 1, dtype=float)

        for client in self.clients:
            self.trust_manager.initialize_client(client.client_id)

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'=' * 60}\nRound {round_num}/{self.num_rounds}\n{'=' * 60}")
            self._apply_mid_training_adversary(round_num)

            print("\nPhase 1: Local Training")
            for client in self.clients:
                client.train()
                client.evaluate()
                client.compute_trust()

            print("\nPhase 2: Trust Update")
            for client in self.clients:
                computed_trust = client.trust_score
                val_acc = client.val_metrics["accuracy"]
                perf_metrics = {
                    "validation_accuracy": val_acc,
                    "val_f1": client.val_metrics["f1_score"],
                    "val_precision": client.val_metrics["precision"],
                    "val_recall": client.val_metrics["recall"],
                    "train_accuracy": (
                        client.train_metrics.get("train_accuracy", 0.0)
                        if client.train_metrics
                        else 0.0
                    ),
                }
                multi_signal_signals = None
                if self.use_multi_signal:
                    try:
                        client.record_communication_round(participated=True)
                        multi_signal_signals = client.compute_multi_signal_trust_signals()
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute multi-signal trust for {client.client_id}: {e}"
                        )
                updated_trust = self.trust_manager.update_trust(
                    client.client_id,
                    round_num,
                    computed_trust,
                    perf_metrics,
                    multi_signal_signals=multi_signal_signals,
                )
                client.trust_score = updated_trust

            all_trust = self.trust_manager.get_all_trust_scores()
            trust_list = [all_trust[c.client_id] for c in self.clients]

            print("\nPhase 3: Client-side sufficient statistics (no row upload)")
            for inner in range(max(1, inner_steps)):
                client_grads = []
                client_hessians = []
                trust_scores = []
                for client in self.clients:
                    X_b, y_b = sample_local_batch(
                        client.X_train,
                        client.y_train,
                        batch_size,
                        self.random_state + round_num * 1000 + hash(client.client_id) % 1000 + inner,
                    )
                    if optimizer == "irls":
                        g_i, h_i = compute_logistic_irls_stats(
                            X_b, y_b, w_global, all_cols
                        )
                        client_grads.append(g_i)
                        client_hessians.append(h_i)
                    else:
                        g_i = compute_logistic_gradient_stat(
                            X_b, y_b, w_global, all_cols
                        )
                        client_grads.append(g_i)
                    trust_scores.append(all_trust.get(client.client_id, client.trust_score))

                if optimizer == "irls":
                    agg = aggregate_trust_weighted_hessian_grad(
                        client_grads,
                        client_hessians,
                        trust_scores,
                        cfg,
                        rng,
                        equal_weight=False,
                    )
                    step = irls_newton_step(
                        agg["noisy_grad"],
                        agg["mean_hessian"],
                        cfg.hessian_reg,
                        cfg.learning_rate,
                    )
                    w_global = w_global - step
                    step_norm = float(np.linalg.norm(step))
                else:
                    agg = aggregate_clipped_gradients(
                        client_grads,
                        trust_scores,
                        cfg,
                        rng,
                        equal_weight=False,
                    )
                    w_global = w_global - cfg.learning_rate * agg["noisy_grad"]
                    step_norm = float(np.linalg.norm(agg["mean_grad"]))

                if inner == inner_steps - 1:
                    print(
                        f"  DP-SAG step ({optimizer}): ||step||={step_norm:.4f}, "
                        f"noise_multiplier={cfg.noise_multiplier}"
                    )

            if round_num % 5 == 0 or round_num == self.num_rounds:
                self.trust_manager.save_trust_history()

        print(f"\n{'=' * 60}\nFinal Evaluation (Strong v1 global model)\n{'=' * 60}")
        from sklearn.linear_model import SGDClassifier

        global_model = SGDClassifier(loss="log_loss", max_iter=1, tol=1e-3, random_state=self.random_state)
        global_model = apply_logistic_weight_vector(global_model, w_global, all_cols)

        X_test_aligned = X_test.reindex(columns=all_cols, fill_value=0)
        results = evaluate_model_on_test(global_model, X_test_aligned, y_test)

        eps = estimate_epsilon(
            cfg.noise_multiplier,
            self.num_rounds,
            len(self.clients),
            cfg.delta,
            cfg.clip_norm,
        )
        results["privacy_mode"] = "strong_v1"
        results["privacy_config"] = {
            "clip_norm": cfg.clip_norm,
            "noise_multiplier": cfg.noise_multiplier,
            "delta": cfg.delta,
            "learning_rate": cfg.learning_rate,
            "local_batch_size": batch_size,
            "secure_aggregation": cfg.secure_aggregation,
            "privacy_optimizer": optimizer,
            "inner_steps": inner_steps,
            "hessian_reg": cfg.hessian_reg,
        }
        results["epsilon_estimate"] = eps
        results["trust_statistics"] = self.trust_manager.get_statistics()
        results["feature_columns"] = all_cols
        results["no_row_upload"] = True

        print(f"\nTest F1-Score: {results['f1_score']:.4f}")
        if eps is not None:
            print(f"Privacy ε estimate (heuristic): {eps:.3f} (δ={cfg.delta})")

        return results

    def _approach_3_multi_round(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Multi-round trust-aware federated learning with adaptive trust."""
        print("\n" + "="*60)
        print(f"Approach 3: Trust-Aware Federated Learning (Multi-Round: {self.num_rounds} rounds)")
        print("="*60)
        
        # Initialize clients in TrustManager
        self.round_logs = []
        for client in self.clients:
            self.trust_manager.initialize_client(client.client_id)
        
        global_model = None
        
        # Multi-round federated learning loop
        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*60}")
            
            self._apply_mid_training_adversary(round_num)

            # Phase 1: Local Training (optional comm_skip withhold)
            print("\nPhase 1: Local Training")
            skipped_ids = set()
            for client in self.clients:
                if (
                    getattr(self, "late_compromise", None) is not None
                    and self.late_compromise.should_skip_participation(
                        client.client_id, round_num
                    )
                ):
                    skipped_ids.add(client.client_id)
                    reason = self.late_compromise.participation_skip_reason(
                        client.client_id, round_num
                    )
                    print(f"  [{reason or 'skip'}] {client.client_id} withholds this round")
                    continue
                client.train()
                client.evaluate()
                client.compute_trust()
            
            # Phase 2: Trust Update
            print("\nPhase 2: Trust Update")
            for client in self.clients:
                # Get the computed trust score (includes F1, gap penalty, train_acc penalty, etc.)
                computed_trust = client.trust_score
                
                # Debug: Print computed trust for compromised clients in round 1
                if round_num == 1 and 'compromised' in client.client_id.lower():
                    train_acc = client.train_metrics.get('train_accuracy', 0.0) if client.train_metrics else 0.0
                    val_acc = client.val_metrics['accuracy'] if client.val_metrics else 0.0
                    print(f"  DEBUG {client.client_id[:50]}: computed_trust={computed_trust:.4f}, "
                          f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
                
                # Get validation accuracy for logging
                if client.val_metrics:
                    val_acc = client.val_metrics['accuracy']
                    perf_metrics = {
                        'validation_accuracy': val_acc,
                        'val_f1': client.val_metrics['f1_score'],
                        'val_precision': client.val_metrics['precision'],
                        'val_recall': client.val_metrics['recall'],
                        'train_accuracy': client.train_metrics.get('train_accuracy', 0.0) if client.train_metrics else 0.0
                    }
                else:
                    perf_metrics = {
                        'validation_accuracy': 0.0,
                        'val_f1': 0.0,
                        'val_precision': 0.0,
                        'val_recall': 0.0,
                        'train_accuracy': 0.0,
                    }
                
                # Use the computed trust (from improved calculation) for adaptive trust update
                # The adaptive formula smooths the improved trust over rounds, not just validation accuracy
                # Get multi-signal signals if enabled
                multi_signal_signals = None
                participated = client.client_id not in skipped_ids
                if self.use_multi_signal:
                    try:
                        client.record_communication_round(participated=participated)
                        multi_signal_signals = client.compute_multi_signal_trust_signals()
                    except Exception as e:
                        logger.warning(f"Failed to compute multi-signal trust for {client.client_id}: {e}")
                        # Fallback to simple trust
                        multi_signal_signals = None
                elif not participated:
                    # V-only path still records the miss for logs; trust stays V-driven
                    try:
                        client.record_communication_round(participated=False)
                    except Exception:
                        pass
                
                updated_trust = self.trust_manager.update_trust(
                    client.client_id,
                    round_num,
                    computed_trust,  # Use computed trust instead of raw validation accuracy
                    perf_metrics,
                    multi_signal_signals=multi_signal_signals
                )
                
                # Update client's trust score
                client.trust_score = updated_trust
                
                # Check for anomalies
                anomaly = self.trust_manager.detect_anomaly(client.client_id)
                if anomaly:
                    print(f"  ⚠️  Anomaly detected for {client.client_id}: {anomaly['type']} (drop: {anomaly['magnitude']:.4f})")
            
            # Display updated trust scores and evolution
            print(f"\nUpdated Trust Scores (Round {round_num}):")
            all_trust = self.trust_manager.get_all_trust_scores()
            trust_list = []
            for client_id, trust in sorted(all_trust.items(), key=lambda x: x[1], reverse=True):
                trust_list.append(trust)
                # Get trust history to show change
                if client_id in self.trust_manager.trust_histories:
                    history = self.trust_manager.trust_histories[client_id]
                    if history and len(history.trust_scores) > 1:
                        prev_trust = history.trust_scores[-2] if len(history.trust_scores) > 1 else trust
                        change = trust - prev_trust
                        change_str = f" ({change:+.4f})" if abs(change) > 0.001 else " (stable)"
                        print(f"  {client_id}: {trust:.4f}{change_str}")
                    else:
                        print(f"  {client_id}: {trust:.4f}")
                else:
                    print(f"  {client_id}: {trust:.4f}")
            
            # Show trust evolution summary
            if round_num > 1 and trust_list:
                trust_range = max(trust_list) - min(trust_list)
                print(f"\nTrust Statistics (Round {round_num}):")
                print(f"  Range: {min(trust_list):.4f} - {max(trust_list):.4f} (range: {trust_range:.4f})")
                print(f"  Mean: {np.mean(trust_list):.4f}, Std: {np.std(trust_list):.4f}")
                if trust_range > 0.3:
                    print("  ✅ High heterogeneity - Trust-Aware adapts dynamically!")
                print("  📊 Trust scores change over rounds (dynamic) - FedAvg weights stay constant!")
            
            # Phase 3: Get Model Updates (exclude comm_skip clients this round)
            param_agg = self._upload_poison_requires_param_agg()
            self.client_updates = [
                u for u in self._collect_client_updates(
                    round_num=round_num, include_data=not param_agg
                )
                if u.get("client_id") not in skipped_ids
            ]
            
            # Phase 4: Aggregation
            print("\nPhase 3: Aggregation")
            
            # Update trust scores in client updates for aggregator
            all_trust = self.trust_manager.get_all_trust_scores()
            for update in self.client_updates:
                update['trust'] = all_trust.get(update['client_id'], update.get('trust', 0.5))
            
            # β=1 → α_i ∝ T_i (paper Alg. 1); iomt_natural governance uses the same.
            _beta = 1.0 if "iomt_natural" in str(self.data_dir) else 0.8
            _is_natural = "iomt_natural" in str(self.data_dir)
            if param_agg:
                # Upload poison (sign_flip): trust-weighted parameter average.
                # Ensemble on client.model would ignore flipped parameters;
                # sample retrain would ignore them too (clean labels).
                print(
                    "  Path AGG / sign_flip: trust-weighted parameter aggregation "
                    "(not ensemble / not sample retrain)"
                )
                round_agg = TrustAwareAggregator(
                    model_type=self.model_type,
                    trust_manager=self.trust_manager,
                    beta=_beta,
                    min_trust=0.0 if _is_natural else 0.6,
                )
                global_model = round_agg.aggregate(
                    self.client_updates, use_retraining=False
                )
                weights = round_agg.get_client_weights(self.client_updates)
            else:
                ensemble = EnsembleAggregator(
                    model_type=self.model_type,
                    aggregation_method='weighted_voting',
                    beta=_beta,
                )
                ensemble.aggregate(self.client_updates)
                global_model = ensemble
                weights = ensemble.get_client_weights(self.client_updates)
            
            # Display client weights
            print("\nClient Weights in Aggregation:")
            for client_id, weight in weights.items():
                print(f"  {client_id}: {weight:.4f}")

            # Natural partition: per-round T/V/α for AUROC / delay analysis
            if self.use_shared_val_ref:
                attackers = set()
                if getattr(self, "late_compromise", None) is not None:
                    attackers = set(self.late_compromise.attacker_ids)
                client_trust = {}
                for client in self.clients:
                    cid = client.client_id
                    try:
                        v_i = float(client.get_validation_performance_metric())
                    except Exception:
                        v_i = None
                    client_trust[cid] = self._client_trust_log_row(
                        client,
                        round_num,
                        alpha=float(weights.get(cid, 0.0)),
                        attackers=attackers,
                        trust_value=float(all_trust.get(cid, client.trust_score or 0.0)),
                        v_i=v_i,
                    )
                self.round_logs.append({
                    "round": round_num,
                    "client_trust": client_trust,
                    "compromise_active": bool(
                        self.late_compromise
                        and self.late_compromise.is_compromised_round(round_num)
                    ),
                    "attack_active": bool(
                        self.late_compromise
                        and self.late_compromise.is_attack_active(round_num)
                    ),
                    "aggregator": (
                        "trust_weighted_param" if param_agg else "ensemble_weighted"
                    ),
                })
            
            # Save trust history periodically
            if round_num % 5 == 0 or round_num == self.num_rounds:
                self.trust_manager.save_trust_history()
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        
        all_trust = self.trust_manager.get_all_trust_scores()
        param_agg = self._upload_poison_requires_param_agg()
        # Ensure final updates carry current trust (iomt_natural: β=1, no hard 0.6 cut)
        self.client_updates = self._collect_client_updates(
            round_num=self.num_rounds, include_data=not param_agg
        )
        for update in self.client_updates:
            update['trust'] = all_trust.get(update['client_id'], update.get('trust', 0.5))
        
        _is_natural = "iomt_natural" in str(self.data_dir)
        use_retrain = bool(self.trust_use_retraining) and not param_agg
        if param_agg:
            print(
                "Final: trust-weighted parameter aggregation "
                "(sign_flip / upload poison visible)"
            )
        final_aggregator = TrustAwareAggregator(
            model_type=self.model_type,
            trust_manager=self.trust_manager,
            beta=1.0 if _is_natural else 0.8,
            min_trust=0.0 if _is_natural else 0.6,
        )
        final_global_model = final_aggregator.aggregate(
            self.client_updates, use_retraining=use_retrain
        )
        
        # Align test set features with training features (use aggregator's training columns)
        if hasattr(final_aggregator, 'training_feature_columns') and final_aggregator.training_feature_columns:
            X_test_aligned = X_test.reindex(columns=final_aggregator.training_feature_columns, fill_value=0)
        elif len(self.clients) > 0:
            # Fallback: use all unique columns from all clients
            all_train_columns = set()
            for client in self.clients:
                all_train_columns.update(client.X_train.columns)
            all_train_columns = sorted(list(all_train_columns))
            X_test_aligned = X_test.reindex(columns=all_train_columns, fill_value=0)
        else:
            X_test_aligned = X_test
        
        results = evaluate_model_on_test(final_global_model, X_test_aligned, y_test)
        
        # Add trust statistics to results
        trust_stats = self.trust_manager.get_statistics()
        results['trust_statistics'] = trust_stats
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        print(f"\nTrust Statistics:")
        print(f"  Mean: {trust_stats['mean']:.4f}")
        print(f"  Std: {trust_stats['std']:.4f}")
        print(f"  Min: {trust_stats['min']:.4f}")
        print(f"  Max: {trust_stats['max']:.4f}")

        results["round_logs"] = list(self.round_logs)
        if self.resolved_signal_weights is not None:
            results["signal_weights"] = dict(self.resolved_signal_weights)
        
        return results
    
    def run_experiment(self, num_clients: Optional[int] = None, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Args:
            num_clients: Number of clients to use (None = use all)
            num_rounds: Override num_rounds from initialization (optional)
            
        Returns:
            Complete results dictionary
        """
        if num_rounds is not None:
            self.num_rounds = num_rounds
        
        print("="*60)
        print("TrustFed-RL Federated Learning Experiment")
        if self.num_rounds > 1:
            print(f"Mode: Multi-Round ({self.num_rounds} rounds with adaptive trust)")
        else:
            print("Mode: Single-Round (static trust)")
        print("="*60)
        
        # Discover data files
        attack_files, benign_files = self.discover_client_files()
        
        if not attack_files:
            raise ValueError(f"No attack CSV files found in {self.data_dir}")
        
        # CRITICAL FIX: Reserve test file BEFORE setting up clients
        # This ensures test set is completely separate (no data leakage)
        # IMPORTANT: Use heterogeneous test set (matches training distribution)
        if self.test_csv is None and len(attack_files) > 1:
            # First, try to find pre-created heterogeneous test set
            heterogeneous_test_file = Path('data/CSVs/heterogeneous_test_set.csv')
            
            if heterogeneous_test_file.exists():
                # Use pre-created heterogeneous test set (matches training distribution)
                self.reserved_test_file = str(heterogeneous_test_file)
                attack_files_for_clients = attack_files  # Use all heterogeneous files for clients
                print(f"\n✅ RESERVED HETEROGENEOUS TEST SET (No Data Leakage):")
                print(f"   Test file: {Path(self.reserved_test_file).name}")
                print(f"   Matches training distribution (heterogeneous clients)")
                print(f"   Contains BOTH benign and attack samples")
                print(f"   This is realistic for IoMT evaluation")
            else:
                # Fallback: Try to find a mixed file (has both benign and attack samples)
                mixed_test_dir = Path('data/CSVs')
                mixed_files = sorted([f for f in mixed_test_dir.glob('mixed_*.csv') 
                                     if 'benign' not in f.name.lower()])
                
                if mixed_files:
                    # Use first mixed file as test set (has both classes)
                    self.reserved_test_file = str(mixed_files[0])
                    attack_files_for_clients = attack_files  # Use all heterogeneous files for clients
                    print(f"\n⚠️  RESERVED MIXED TEST FILE (No Data Leakage):")
                    print(f"   Test file: {Path(self.reserved_test_file).name}")
                    print(f"   ⚠️  WARNING: Mixed file may have different distribution than training")
                    print(f"   ⚠️  Consider creating heterogeneous_test_set.csv for better matching")
                else:
                    # Last resort: Prefer high-quality file for test set
                    high_quality_files = [f for f in attack_files if 'high_quality' in Path(f).name.lower()]
                    medium_quality_files = [f for f in attack_files if 'medium_quality' in Path(f).name.lower()]
                    
                    if high_quality_files:
                        self.reserved_test_file = high_quality_files[0]
                        attack_files_for_clients = [f for f in attack_files if f != self.reserved_test_file]
                        print(f"\n⚠️  RESERVED HIGH-QUALITY TEST FILE (No Data Leakage):")
                        print(f"   Test file: {Path(self.reserved_test_file).name}")
                        print(f"   ⚠️  WARNING: This file may only contain attack samples")
                        print(f"   ⚠️  Consider creating heterogeneous_test_set.csv for realistic evaluation")
                    elif medium_quality_files:
                        self.reserved_test_file = medium_quality_files[0]
                        attack_files_for_clients = [f for f in attack_files if f != self.reserved_test_file]
                        print(f"\n⚠️  RESERVED MEDIUM-QUALITY TEST FILE (No Data Leakage):")
                        print(f"   Test file: {Path(self.reserved_test_file).name}")
                        print(f"   ⚠️  WARNING: This file may only contain attack samples")
                    else:
                        self.reserved_test_file = attack_files[-1]
                        attack_files_for_clients = attack_files[:-1]
                        print(f"\n⚠️  RESERVED TEST FILE (No Data Leakage):")
                        print(f"   Test file: {Path(self.reserved_test_file).name}")
                        print(f"   ⚠️  WARNING: This file may only contain attack samples")
            
            print(f"   This file will NOT be used for any client training")
            print(f"   Total files: {len(attack_files)}, Clients: {len(attack_files_for_clients)}, Test: 1")
        else:
            self.reserved_test_file = None
            attack_files_for_clients = attack_files
        
        # Set up clients (using files that are NOT reserved for test)
        self.setup_clients(attack_files_for_clients, benign_files, num_clients=num_clients)
        
        # Train all clients
        self.train_clients()
        
        # Prepare test data - CRITICAL FIX: Use completely separate test set
        # Test set was reserved BEFORE training to avoid data leakage
        if self.test_csv is None:
            print("\n" + "="*60)
            print("PREPARING PROPER TEST SET (No Data Leakage)")
            print("="*60)
            
            if hasattr(self, 'reserved_test_file') and self.reserved_test_file:
                # Use the reserved test file (completely separate from training)
                print(f"\n✅ Using reserved test file: {Path(self.reserved_test_file).name}")
                print(f"   This file was NOT used for any client training")
                print(f"   ✅ NO DATA LEAKAGE - test set is completely unseen")
                X_test, y_test = self.prepare_test_data(self.reserved_test_file)
            else:
                # Fallback: Try to find a file not used by clients
                all_attack_files = sorted([str(f) for f in self.data_dir.glob('*.csv') 
                                         if 'benign' not in f.name.lower()])
                client_files = [c.data_path for c in self.clients]
                test_candidates = [f for f in all_attack_files if f not in client_files]
                
                if test_candidates:
                    test_file = test_candidates[0]
                    print(f"\n✅ Using separate file as test set: {Path(test_file).name}")
                    print(f"   This file was NOT used for any client training")
                    X_test, y_test = self.prepare_test_data(test_file)
                else:
                    # Last resort: Use original source data
                    print(f"\n⚠️  No separate test file available - using original source data")
                    source_dir = Path('~/Papers/CSVs').expanduser()
                    if source_dir.exists():
                        source_files = sorted([f for f in source_dir.glob('mixed_*.csv') 
                                             if 'benign' not in f.name.lower()])
                        if source_files:
                            test_file = source_files[0]
                            print(f"   Using original source: {test_file.name}")
                            X_test, y_test = self.prepare_test_data(str(test_file))
                        else:
                            print(f"   ⚠️  WARNING: Using validation data (not ideal)")
                            X_test = self.clients[0].X_val
                            y_test = self.clients[0].y_val
                    else:
                        print(f"   ⚠️  WARNING: Using validation data (not ideal)")
                        X_test = self.clients[0].X_val
                        y_test = self.clients[0].y_val
        else:
            # User provided test CSV
            X_test, y_test = self.prepare_test_data(self.test_csv)
        
        if self.trust_no_retrain_only:
            trust_aware_key = 'trust_aware_no_retrain'
        else:
            trust_aware_key = (
                'trust_aware' if self.trust_use_retraining else 'trust_aware_no_retrain'
            )

        if self.baseline_only == "fedavg":
            print("Mode: baseline-only FedAvg (B0)")
            federated_results = self.approach_2_federated_equal_weight(X_test, y_test)
            return {
                "federated_equal_weight": federated_results,
                "approach": "trustfed_fedavg",
            }
        if self.baseline_only == "median":
            print("Mode: baseline-only coordinate-median (Path AGG Rm)")
            median_results = self.approach_5_coordinate_median(X_test, y_test, local_epochs=1)
            return {
                "coordinate_median": median_results,
                "approach": "trustfed_rm",
            }

        if self.trust_no_retrain_only or self.trust_only:
            centralized_results = None
            federated_results = None
            fedprox_results = None
            median_results = None
            krum_results = None
            trust_aware_results = self.approach_3_trust_aware(X_test, y_test)
        else:
            centralized_results = self.approach_1_centralized(X_test, y_test)
            federated_results = self.approach_2_federated_equal_weight(X_test, y_test)
            trust_aware_results = self.approach_3_trust_aware(X_test, y_test)

            fedprox_results = None
            median_results = None
            krum_results = None
            if self.model_type == 'logistic_regression':
                fedprox_results = self.approach_4_fedprox(X_test, y_test, mu=1e-3, local_epochs=1)
                median_results = self.approach_5_coordinate_median(X_test, y_test, local_epochs=1)
                krum_results = self.approach_6_krum(X_test, y_test, local_epochs=1)
        
        # Generate summary
        client_info = [client.get_info() for client in self.clients]
        summary = None
        if centralized_results is not None and federated_results is not None:
            summary = generate_results_summary(
                centralized_results,
                federated_results,
                trust_aware_results,
                client_info,
                trust_manager=self.trust_manager if hasattr(self, 'trust_manager') else None
            )
        
        # Print comparison
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        if self.trust_no_retrain_only:
            print(f"\n{'Metric':<25} {'Trust-Aware (no retrain)':<25}")
            print("-" * 50)
            print(f"{'Accuracy':<25} {trust_aware_results['accuracy']:<25.4f}")
            print(f"{'F1-Score':<25} {trust_aware_results['f1_score']:<25.4f}")
            print(f"{'Precision':<25} {trust_aware_results['precision']:<25.4f}")
            print(f"{'Recall':<25} {trust_aware_results['recall']:<25.4f}")
            print(f"{'False Positive Rate':<25} {trust_aware_results['false_positive_rate']:<25.4f}")
        elif self.trust_only:
            print(f"\n{'Metric':<25} {'Trust-Aware':<25}")
            print("-" * 50)
            print(f"{'Accuracy':<25} {trust_aware_results['accuracy']:<25.4f}")
            print(f"{'F1-Score':<25} {trust_aware_results['f1_score']:<25.4f}")
            print(f"{'Precision':<25} {trust_aware_results['precision']:<25.4f}")
            print(f"{'Recall':<25} {trust_aware_results['recall']:<25.4f}")
            print(f"{'False Positive Rate':<25} {trust_aware_results['false_positive_rate']:<25.4f}")
        elif fedprox_results is not None and median_results is not None and krum_results is not None:
            print(f"\n{'Metric':<25} {'Centralized':<11} {'FedAvg':<11} {'Trust':<11} {'FedProx':<11} {'Median':<11} {'Krum':<11}")
            print("-" * 96)
            print(f"{'Accuracy':<25} {centralized_results['accuracy']:<11.4f} {federated_results['accuracy']:<11.4f} {trust_aware_results['accuracy']:<11.4f} {fedprox_results['accuracy']:<11.4f} {median_results['accuracy']:<11.4f} {krum_results['accuracy']:<11.4f}")
            print(f"{'F1-Score':<25} {centralized_results['f1_score']:<11.4f} {federated_results['f1_score']:<11.4f} {trust_aware_results['f1_score']:<11.4f} {fedprox_results['f1_score']:<11.4f} {median_results['f1_score']:<11.4f} {krum_results['f1_score']:<11.4f}")
            print(f"{'Precision':<25} {centralized_results['precision']:<11.4f} {federated_results['precision']:<11.4f} {trust_aware_results['precision']:<11.4f} {fedprox_results['precision']:<11.4f} {median_results['precision']:<11.4f} {krum_results['precision']:<11.4f}")
            print(f"{'Recall':<25} {centralized_results['recall']:<11.4f} {federated_results['recall']:<11.4f} {trust_aware_results['recall']:<11.4f} {fedprox_results['recall']:<11.4f} {median_results['recall']:<11.4f} {krum_results['recall']:<11.4f}")
            print(f"{'False Positive Rate':<25} {centralized_results['false_positive_rate']:<11.4f} {federated_results['false_positive_rate']:<11.4f} {trust_aware_results['false_positive_rate']:<11.4f} {fedprox_results['false_positive_rate']:<11.4f} {median_results['false_positive_rate']:<11.4f} {krum_results['false_positive_rate']:<11.4f}")
        elif fedprox_results is not None and median_results is not None:
            print(f"\n{'Metric':<25} {'Centralized':<12} {'FedAvg':<12} {'Trust':<12} {'FedProx':<12} {'Median':<12}")
            print("-" * 85)
            print(f"{'Accuracy':<25} {centralized_results['accuracy']:<12.4f} {federated_results['accuracy']:<12.4f} {trust_aware_results['accuracy']:<12.4f} {fedprox_results['accuracy']:<12.4f} {median_results['accuracy']:<12.4f}")
            print(f"{'F1-Score':<25} {centralized_results['f1_score']:<12.4f} {federated_results['f1_score']:<12.4f} {trust_aware_results['f1_score']:<12.4f} {fedprox_results['f1_score']:<12.4f} {median_results['f1_score']:<12.4f}")
            print(f"{'Precision':<25} {centralized_results['precision']:<12.4f} {federated_results['precision']:<12.4f} {trust_aware_results['precision']:<12.4f} {fedprox_results['precision']:<12.4f} {median_results['precision']:<12.4f}")
            print(f"{'Recall':<25} {centralized_results['recall']:<12.4f} {federated_results['recall']:<12.4f} {trust_aware_results['recall']:<12.4f} {fedprox_results['recall']:<12.4f} {median_results['recall']:<12.4f}")
            print(f"{'False Positive Rate':<25} {centralized_results['false_positive_rate']:<12.4f} {federated_results['false_positive_rate']:<12.4f} {trust_aware_results['false_positive_rate']:<12.4f} {fedprox_results['false_positive_rate']:<12.4f} {median_results['false_positive_rate']:<12.4f}")
        else:
            print(f"\n{'Metric':<25} {'Centralized':<15} {'FedAvg':<15} {'Trust-Aware':<15}")
            print("-" * 70)
            print(f"{'Accuracy':<25} {centralized_results['accuracy']:<15.4f} {federated_results['accuracy']:<15.4f} {trust_aware_results['accuracy']:<15.4f}")
            print(f"{'F1-Score':<25} {centralized_results['f1_score']:<15.4f} {federated_results['f1_score']:<15.4f} {trust_aware_results['f1_score']:<15.4f}")
            print(f"{'Precision':<25} {centralized_results['precision']:<15.4f} {federated_results['precision']:<15.4f} {trust_aware_results['precision']:<15.4f}")
            print(f"{'Recall':<25} {centralized_results['recall']:<15.4f} {federated_results['recall']:<15.4f} {trust_aware_results['recall']:<15.4f}")
            print(f"{'False Positive Rate':<25} {centralized_results['false_positive_rate']:<15.4f} {federated_results['false_positive_rate']:<15.4f} {trust_aware_results['false_positive_rate']:<15.4f}")
        
        results_dict = {}
        if centralized_results is not None:
            results_dict['centralized'] = centralized_results
        if federated_results is not None:
            results_dict['federated_equal_weight'] = federated_results
        results_dict[trust_aware_key] = trust_aware_results
        if summary is not None:
            results_dict['summary'] = summary
        if fedprox_results is not None:
            results_dict['fedprox'] = fedprox_results
        if median_results is not None:
            results_dict['coordinate_median'] = median_results
        if krum_results is not None:
            results_dict['krum'] = krum_results
        
        # Save to JSON
        os.makedirs('results/reports', exist_ok=True)
        with open('results/reports/experiment_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\n✓ Results saved to results/reports/experiment_results.json")
        
        # Generate visualizations (optional)
        print("\nGenerating visualizations...")
        viz_results = {
            'centralized': centralized_results,
            'federated_equal_weight': federated_results,
            'trust_aware': trust_aware_results
        }
        if fedprox_results is not None:
            viz_results['fedprox'] = fedprox_results
        if median_results is not None:
            viz_results['coordinate_median'] = median_results
        if krum_results is not None:
            viz_results['krum'] = krum_results

        print("Warning: visualization disabled (matplotlib import avoided on this system).")
        
        return results_dict


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TrustFed-RL Federated Learning Experiment')
    parser.add_argument('--data-dir', type=str, default='../data/CSVs',
                       help='Directory containing CSV files')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'logistic_regression', 'mlp', 'xgboost'],
                       help='Type of model to use')
    parser.add_argument('--num-clients', type=int, default=None,
                       help='Number of clients to use (None = use all)')
    parser.add_argument('--num-rounds', type=int, default=1,
                       help='Number of federated learning rounds (1 = single round, >1 = multi-round with adaptive trust)')
    parser.add_argument('--test-csv', type=str, default=None,
                       help='Path to test CSV file (optional)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--trust-alpha', type=float, default=0.7,
                       help='History weight for adaptive trust (0.7 = 70%% old, 30%% new)')
    parser.add_argument('--trust-storage-dir', type=str, default=None,
                       help='Directory to store trust history (default: results/trust_history)')
    parser.add_argument('--multi-signal-trust', action='store_true',
                       help='Use multi-signal trust fusion (accuracy, stability, drift, uncertainty) instead of simple validation accuracy')
    parser.add_argument('--num-trials', type=int, default=1,
                       help='Number of independent trials for statistical validation (default: 1, recommended: 5-10)')
    parser.add_argument('--trust-performance-signal', type=str, default='auto',
                       choices=['auto', 'accuracy', 'f1_score', 'recall'],
                       help='Validation metric for multi-signal trust λ₁ (auto/iomt→f1_score)')
    parser.add_argument('--trust-no-retrain', action='store_true',
                       help='Trust-Aware uses parameter averaging instead of trust-weighted retraining')
    parser.add_argument('--trust-no-retrain-only', action='store_true',
                       help='Run only Trust-Aware without retraining (fast ablation mode)')
    parser.add_argument('--trust-only', action='store_true',
                       help='Run only Trust-Aware with retraining (skip baselines; for λ sweeps)')
    parser.add_argument('--lambda2', type=float, default=None,
                       help='Override multi-signal λ₂ (stability penalty weight)')
    parser.add_argument('--lambda3', type=float, default=None,
                       help='Override multi-signal λ₃ (drift penalty weight)')
    parser.add_argument('--lambda4', type=float, default=None,
                       help='Override multi-signal λ₄ (uncertainty weight)')
    parser.add_argument('--no-dynamic-adversary', action='store_true',
                       help='Disable mid-training delayed/gradual poisoning schedule')
    parser.add_argument('--summary-suffix', type=str, default='',
                       help='Extra suffix for statistical_summary JSON (e.g., _adaptive_3trial)')
    parser.add_argument('--privacy-mode', type=str, default='legacy',
                       choices=['legacy', 'strong_v1'],
                       help='legacy=server row retrain; strong_v1=DP-SAG gradient stats (no row upload)')
    parser.add_argument('--clip-norm', type=float, default=None,
                       help='Strong v1: gradient clip norm C')
    parser.add_argument('--noise-multiplier', type=float, default=None,
                       help='Strong v1: DP noise multiplier (0=SA only)')
    parser.add_argument('--privacy-delta', type=float, default=None,
                       help='Strong v1: DP delta parameter')
    parser.add_argument('--privacy-optimizer', type=str, default=None,
                       choices=['gradient', 'irls'],
                       help='Strong v1: gradient-only or IRLS moment stack')
    parser.add_argument('--privacy-inner-steps', type=int, default=None,
                       help='Strong v1: inner global steps per FL round')
    parser.add_argument('--privacy-lr', type=float, default=None,
                       help='Strong v1: global learning rate / Newton scale')
    parser.add_argument('--privacy-batch-size', type=int, default=None,
                       help='Strong v1: local batch size m_i')
    parser.add_argument('--hessian-reg', type=float, default=None,
                       help='Strong v1 IRLS: ridge on Hessian')
    
    args = parser.parse_args()
    dynamic_adversary = not args.no_dynamic_adversary
    perf_signal_arg = None if args.trust_performance_signal == 'auto' else args.trust_performance_signal
    if args.trust_no_retrain_only:
        trust_use_retraining = False
    else:
        trust_use_retraining = not args.trust_no_retrain

    lambda_weights_override = None
    if any(v is not None for v in (args.lambda2, args.lambda3, args.lambda4)):
        trust_config = load_trust_config()
        ms = trust_config.get('multi_signal', {})
        lambda_weights_override = {
            'lambda1': ms.get('lambda1', 1.0),
            'lambda2': args.lambda2 if args.lambda2 is not None else ms.get('lambda2', 0.3),
            'lambda3': args.lambda3 if args.lambda3 is not None else ms.get('lambda3', 0.2),
            'lambda4': args.lambda4 if args.lambda4 is not None else ms.get('lambda4', 0.2),
        }

    privacy_config_override = {}
    if args.clip_norm is not None:
        privacy_config_override["clip_norm"] = args.clip_norm
    if args.noise_multiplier is not None:
        privacy_config_override["noise_multiplier"] = args.noise_multiplier
    if args.privacy_delta is not None:
        privacy_config_override["delta"] = args.privacy_delta
    if args.privacy_optimizer is not None:
        privacy_config_override["privacy_optimizer"] = args.privacy_optimizer
    if args.privacy_inner_steps is not None:
        privacy_config_override["inner_steps"] = args.privacy_inner_steps
    if args.privacy_lr is not None:
        privacy_config_override["learning_rate"] = args.privacy_lr
    if args.privacy_batch_size is not None:
        privacy_config_override["local_batch_size"] = args.privacy_batch_size
    if args.hessian_reg is not None:
        privacy_config_override["hessian_reg"] = args.hessian_reg
    if not privacy_config_override:
        privacy_config_override = None

    def _default_approach_names() -> List[str]:
        if args.trust_no_retrain_only:
            return ['trust_aware_no_retrain']
        if args.trust_only:
            return ['trust_aware'] if trust_use_retraining else ['trust_aware_no_retrain']
        trust_key = 'trust_aware' if trust_use_retraining else 'trust_aware_no_retrain'
        names = ['centralized', 'federated_equal_weight', trust_key]
        if args.model_type == 'logistic_regression' and not args.trust_no_retrain_only:
            names.extend(['fedprox', 'coordinate_median', 'krum'])
        return names

    def _metric_value(trial_results: Dict, approach: str, metric: str) -> float:
        if approach not in trial_results:
            return 0.0
        block = trial_results[approach]
        if metric == 'false_negative_rate':
            if 'false_negative_rate' in block:
                return float(block['false_negative_rate'])
            return 1.0 - float(block.get('recall', 0.0))
        return float(block.get(metric, 0.0))

    def _build_stats_summary(all_trial_results: List[Dict], approach_names: List[str]) -> Dict:
        metrics = [
            'accuracy', 'f1_score', 'precision', 'recall',
            'false_positive_rate', 'false_negative_rate',
        ]
        stats_summary = {}
        for approach in approach_names:
            stats_summary[approach] = {}
            for metric in metrics:
                values = [
                    _metric_value(trial_results, approach, metric)
                    for trial_results in all_trial_results
                    if approach in trial_results
                ]
                if values:
                    stats_summary[approach][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'values': values,
                    }
        return stats_summary

    # Run multiple trials if requested
    if args.num_trials > 1:
        print(f"\n{'='*60}")
        print(f"Running {args.num_trials} independent trials for statistical validation")
        print(f"{'='*60}\n")
        
        all_trial_results = []
        
        for trial in range(args.num_trials):
            print(f"\n{'='*60}")
            print(f"Trial {trial + 1}/{args.num_trials} (Seed: {args.random_state + trial})")
            print(f"{'='*60}\n")
            
            # Create runner with different seed for each trial
            runner = ExperimentRunner(
                data_dir=args.data_dir,
                model_type=args.model_type,
                random_state=args.random_state + trial,  # Different seed per trial
                test_csv=args.test_csv,
                num_rounds=args.num_rounds,
                trust_alpha=args.trust_alpha,
                trust_storage_dir=args.trust_storage_dir,
                use_multi_signal=args.multi_signal_trust,
                performance_signal=perf_signal_arg,
                trust_use_retraining=trust_use_retraining,
                trust_no_retrain_only=args.trust_no_retrain_only,
                trust_only=args.trust_only,
                dynamic_adversary=dynamic_adversary,
                lambda_weights_override=lambda_weights_override,
                privacy_mode=args.privacy_mode,
                privacy_config_override=privacy_config_override,
            )
            
            trial_results = runner.run_experiment(num_clients=args.num_clients)
            all_trial_results.append(trial_results)
        
        # Compute statistics across trials
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY (Across All Trials)")
        print("="*60)
        
        metrics = [
            'accuracy', 'f1_score', 'precision', 'recall',
            'false_positive_rate', 'false_negative_rate',
        ]
        approach_names = _default_approach_names()
        stats_summary = _build_stats_summary(all_trial_results, approach_names)

        f1_by_trial: Dict[str, List[float]] = {}
        for approach in approach_names:
            f1_by_trial[approach] = [
                _metric_value(tr, approach, 'f1_score')
                for tr in all_trial_results
                if approach in tr
            ]
        
        header = f"\n{'Metric':<22}" + "".join(f"{a[:18]:<20}" for a in approach_names)
        print(header)
        print("-" * (22 + 20 * len(approach_names)))
        for metric in metrics:
            metric_display = metric.replace('_', ' ').title()
            row = f"{metric_display:<22}"
            for approach in approach_names:
                if approach in stats_summary and metric in stats_summary[approach]:
                    mean_val = stats_summary[approach][metric]['mean']
                    std_val = stats_summary[approach][metric]['std']
                    row += f" {mean_val:.4f}±{std_val:.4f}  "
                else:
                    row += " " * 20
            print(row)
        
        # Save statistical summary
        resolved_signal = resolve_performance_signal(args.data_dir, load_trust_config(), perf_signal_arg)
        stats_dict = {
            'num_trials': args.num_trials,
            'random_seeds': [args.random_state + i for i in range(args.num_trials)],
            'data_dir': args.data_dir,
            'test_csv': args.test_csv,
            'performance_signal': resolved_signal,
            'statistics': stats_summary,
            'f1_by_trial': f1_by_trial,
            'all_trial_results': all_trial_results
        }
        
        os.makedirs('results/reports', exist_ok=True)
        data_tag = 'iomt' if 'iomt' in args.data_dir.lower() else 'experiment'
        suffix = '_no_retrain' if args.trust_no_retrain_only or args.trust_no_retrain else ''
        if args.summary_suffix:
            suffix = f'{suffix}{args.summary_suffix}'
        summary_path = f'results/reports/statistical_summary_{data_tag}{suffix}.json'
        with open(summary_path, 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        with open('results/reports/statistical_summary.json', 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        print(f"\n✓ Statistical summary saved to {summary_path}")
        
        # Perform statistical significance tests (t-test)
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTS (t-test)")
        print("="*60)
        
        if stats is None:
            print("Warning: scipy not available. Skipping statistical tests.")
            print("Install scipy with: pip install scipy")
        else:
            def perform_ttest(values1, values2, name1, name2, metric_name):
                """Perform t-test between two groups."""
                if len(values1) < 2 or len(values2) < 2:
                    return None, None
                t_stat, p_value = stats.ttest_ind(values1, values2)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"\n{metric_name}: {name1} vs {name2}")
                print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f} {significance}")
                return t_stat, p_value
            
            trust_key = 'trust_aware' if 'trust_aware' in stats_summary else 'trust_aware_no_retrain'
            comparisons = [
                (trust_key, 'federated_equal_weight', 'Trust-Aware', 'FedAvg'),
                (trust_key, 'centralized', 'Trust-Aware', 'Centralized'),
                (trust_key, 'coordinate_median', 'Trust-Aware', 'Median'),
                (trust_key, 'fedprox', 'Trust-Aware', 'FedProx'),
                (trust_key, 'krum', 'Trust-Aware', 'Krum'),
            ]
            for metric in metrics:
                if trust_key not in stats_summary or metric not in stats_summary[trust_key]:
                    continue
                trust_values = [
                    _metric_value(tr, trust_key, metric) for tr in all_trial_results if trust_key in tr
                ]
                for a_key, b_key, a_name, b_name in comparisons:
                    if b_key not in stats_summary or metric not in stats_summary[b_key]:
                        continue
                    b_values = [
                        _metric_value(tr, b_key, metric) for tr in all_trial_results if b_key in tr
                    ]
                    perform_ttest(trust_values, b_values, a_name, b_name, metric.replace('_', ' ').title())
                    if len(trust_values) >= 3 and len(b_values) >= 3:
                        try:
                            w_stat, w_p = stats.wilcoxon(trust_values, b_values)
                            sig = "***" if w_p < 0.001 else "**" if w_p < 0.01 else "*" if w_p < 0.05 else "ns"
                            print(f"  Wilcoxon({a_name} vs {b_name}): W={w_stat:.4f}, p={w_p:.4f} {sig}")
                        except Exception as exc:
                            print(f"  Wilcoxon skipped: {exc}")
            
            print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        results = stats_dict
    else:
        # Single trial (original behavior)
        runner = ExperimentRunner(
            data_dir=args.data_dir,
            model_type=args.model_type,
            random_state=args.random_state,
            test_csv=args.test_csv,
            num_rounds=args.num_rounds,
            trust_alpha=args.trust_alpha,
            trust_storage_dir=args.trust_storage_dir,
            use_multi_signal=args.multi_signal_trust,
            performance_signal=perf_signal_arg,
            trust_use_retraining=trust_use_retraining,
            trust_no_retrain_only=args.trust_no_retrain_only,
            trust_only=args.trust_only,
            dynamic_adversary=dynamic_adversary,
            lambda_weights_override=lambda_weights_override,
            privacy_mode=args.privacy_mode,
            privacy_config_override=privacy_config_override,
        )
        
        results = runner.run_experiment(num_clients=args.num_clients)
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
