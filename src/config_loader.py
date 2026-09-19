"""
Configuration loader for trust parameters.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

# Master TrustFed_RL-2 §III-E default weights (legacy six-signal; sum = 1)
DEFAULT_SIGNAL_WEIGHTS = {
    "V": 0.25,
    "S": 0.15,
    "D": 0.15,
    "U": 0.15,
    "C": 0.15,
    "R": 0.15,
}

# Natural-partition behavioural trust (Trusted-plan §5): T=f(V,S,D,U,C), R∉T
BEHAVIOURAL_SIGNAL_WEIGHTS = {
    "V": 0.20,
    "S": 0.20,
    "D": 0.20,
    "U": 0.20,
    "C": 0.20,
    "R": 0.0,
}


def load_trust_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load trust configuration from JSON file.
    
    Args:
        config_path: Path to config file (default: config/trust_config.json)
        
    Returns:
        Dictionary with configuration parameters
    """
    if config_path is None:
        # Default path
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'trust_config.json'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default configuration
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        print("Using default configuration")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default trust configuration.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        "trust_manager": {
            "alpha": 0.7,
            "decay_rate": 0.95,
            "anomaly_threshold": 0.2,
            "initial_trust": 0.5,
            "storage_dir": "results/trust_history"
        },
        "consistency": {
            "window_size": 5
        },
        "trend_analysis": {
            "window_size": 5,
            "improving_threshold": 0.01,
            "declining_threshold": -0.01
        },
        "logging": {
            "level": "INFO",
            "log_file": "results/trust_updates.log"
        }
    }


def resolve_performance_signal(
    data_dir: str,
    config: Optional[Dict[str, Any]] = None,
    override: Optional[str] = None,
) -> str:
    """
    Choose validation metric for multi-signal trust fusion (signal V).

    IoMT hospital partitions use F1 by default to prioritize attack detection
    under imbalanced attack/benign prevalence.
    """
    allowed = {"accuracy", "f1_score", "recall"}
    if override and override != "auto":
        if override not in allowed:
            raise ValueError(f"trust performance_signal must be one of {allowed}, got {override!r}")
        return override

    if config is None:
        config = load_trust_config()
    ms = config.get("multi_signal", {})
    mode = ms.get("performance_signal", "auto")
    if mode != "auto":
        if mode not in allowed:
            raise ValueError(f"Invalid performance_signal in config: {mode!r}")
        return mode

    path = str(data_dir).lower()
    if "iomt" in path:
        return ms.get("performance_signal_iomt", "f1_score")
    return ms.get("performance_signal_iomt", "f1_score")


def resolve_signal_weights(
    config: Optional[Dict[str, Any]] = None,
    disable_signals: Optional[Sequence[str]] = None,
    legacy_lambda_override: Optional[Dict[str, float]] = None,
    include_R_in_T: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Resolve trust fusion weights with Σwk = 1.

    Prefer multi_signal.signal_weights. Fall back to normalized legacy λ₁–λ₅
    (λ₅ → R, C taken from equal share if missing). Zero out disable_signals
    (e.g. C,R for B5−S) and renormalize.

    include_R_in_T:
      - None → use multi_signal.include_R_in_T (default True for legacy configs)
      - False → force w_R=0 and renormalize (natural partition / Trusted-plan §5)
      - True → keep R in fusion (B5-R ablation / legacy paper arms)
    """
    if config is None:
        config = {}
    ms = config.get("multi_signal", {}) if isinstance(config, dict) else {}

    if include_R_in_T is None:
        include_R_in_T = bool(ms.get("include_R_in_T", True))

    base = DEFAULT_SIGNAL_WEIGHTS if include_R_in_T else BEHAVIOURAL_SIGNAL_WEIGHTS
    weights = dict(base)
    configured = ms.get("signal_weights")
    if isinstance(configured, dict) and configured:
        for k in DEFAULT_SIGNAL_WEIGHTS:
            if k in configured:
                weights[k] = float(configured[k])
    elif legacy_lambda_override or ms.get("legacy_lambda") or any(
        k in ms for k in ("lambda1", "lambda2", "lambda3", "lambda4", "lambda5")
    ):
        leg = dict(ms.get("legacy_lambda") or {})
        for k in ("lambda1", "lambda2", "lambda3", "lambda4", "lambda5"):
            if k in ms:
                leg[k] = ms[k]
        if legacy_lambda_override:
            leg.update(legacy_lambda_override)
        raw = {
            "V": float(leg.get("lambda1", 1.0)),
            "S": float(leg.get("lambda2", 0.3)),
            "D": float(leg.get("lambda3", 0.2)),
            "U": float(leg.get("lambda4", 0.2)),
            "C": float(leg.get("lambda_C", leg.get("lambda1", 1.0) * 0.15)),
            "R": float(leg.get("lambda5", 0.25)),
        }
        weights = raw

    if not include_R_in_T:
        weights["R"] = 0.0
    elif float(weights.get("R", 0.0)) <= 0.0:
        # B5-R ablation on a natural config that stores R=0: restore default R share
        weights["R"] = float(DEFAULT_SIGNAL_WEIGHTS["R"])

    disabled: List[str] = list(ms.get("disable_signals") or [])
    if disable_signals:
        disabled.extend(disable_signals)
    disabled_u = {s.strip().upper() for s in disabled if s}
    for key in disabled_u:
        if key in weights:
            weights[key] = 0.0

    total = sum(weights.values())
    if total <= 0:
        weights = dict(BEHAVIOURAL_SIGNAL_WEIGHTS if not include_R_in_T else DEFAULT_SIGNAL_WEIGHTS)
        total = sum(weights.values())
    return {k: float(v) / total for k, v in weights.items()}


def resolve_trust_config_path(
    dataset: str,
    trust_config_path: Optional[str] = None,
    root: Optional[Path] = None,
    trust_profile: Optional[str] = None,
) -> str:
    """Pick trust JSON: explicit path wins; iomt_natural uses R∉T config.

    trust_profile:
      - ``b1`` / ``v_only`` → V-only (natural B1 ablation)
      - ``bc`` / ``c_only`` / ``b1c`` → C-only (Path 1 skip-aware baseline)
      - ``b2`` / ``behavioural`` / None → equal V,S,D,U,C (natural default)
    """
    if root is None:
        root = Path(__file__).parent.parent
    if trust_config_path:
        return str(trust_config_path)
    if dataset == "iomt_natural":
        profile = (trust_profile or "").strip().lower()
        if profile in ("b1", "v_only", "b1v"):
            b1 = root / "config" / "trust_config_iomt_natural_b1.json"
            if b1.exists():
                return str(b1)
        if profile in ("bc", "c_only", "b1c"):
            bc = root / "config" / "trust_config_iomt_natural_c_only.json"
            if bc.exists():
                return str(bc)
        natural = root / "config" / "trust_config_iomt_natural.json"
        if natural.exists():
            return str(natural)
    return str(root / "config" / "trust_config.json")


def resolve_dataset_configs(
    dataset: str,
    governance_config_path: Optional[str] = None,
    rl_config_path: Optional[str] = None,
    root: Optional[Path] = None,
) -> tuple[str, str]:
    """Return governance and RL config paths for a dataset name.

    Explicit paths win when provided. Otherwise IoMT uses the IoMT-specific
    configs; other datasets use the shared defaults.
    """
    if root is None:
        root = Path(__file__).parent.parent
    if dataset in ("iomt", "iomt_natural", "wustl_ehms"):
        gov = governance_config_path or str(root / "config" / "iomt_governance_config.json")
        rl = rl_config_path or str(root / "config" / "iomt_rl_config.json")
        return gov, rl
    gov = governance_config_path or str(root / "config" / "governance_config.json")
    rl = rl_config_path or str(root / "config" / "rl_config.json")
    return gov, rl


def apply_trust_config(trust_manager: Any, config: Dict[str, Any]) -> None:
    """
    Apply configuration to TrustManager instance.
    
    Args:
        trust_manager: TrustManager instance
        config: Configuration dictionary
    """
    trust_config = config.get('trust_manager', {})
    
    if 'alpha' in trust_config:
        trust_manager.alpha = trust_config['alpha']
    if 'decay_rate' in trust_config:
        trust_manager.decay_rate = trust_config['decay_rate']
    if 'anomaly_threshold' in trust_config:
        trust_manager.anomaly_threshold = trust_config['anomaly_threshold']
    if 'initial_trust' in trust_config:
        trust_manager.initial_trust = trust_config['initial_trust']
    if 'storage_dir' in trust_config:
        trust_manager.storage_dir = trust_config['storage_dir']
        if trust_manager.storage_dir:
            os.makedirs(trust_manager.storage_dir, exist_ok=True)


def load_privacy_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load privacy configuration (Strong v1 DP-SAG defaults)."""
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "privacy_config.json"
    config_path = Path(config_path)
    if not config_path.exists():
        return {
            "strong_v1": {
                "local_batch_size": 1000,
                "clip_norm": 1.0,
                "noise_multiplier": 0.0,
                "delta": 1e-5,
                "learning_rate": 0.05,
                "secure_aggregation": True,
                "trust_beta": 0.8,
            }
        }
    with open(config_path, "r") as f:
        return json.load(f)


def load_json_config(config_path: Optional[str], filename: str) -> Dict[str, Any]:
    """Load a JSON config from TrustFed-Agent/config/."""
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / filename
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def load_agent_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return load_json_config(config_path, "agent_config.json")


def load_governance_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return load_json_config(config_path, "governance_config.json")


def load_rl_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return load_json_config(config_path, "rl_config.json")
