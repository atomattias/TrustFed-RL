"""
Evaluation module for computing metrics and comparing approaches.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Any, Dict, List, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }
    
    # Calculate false positive rate
    # Use labels parameter to ensure 2x2 matrix even if only one class present
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    except ValueError:
        # Fallback if labels don't match
        cm = confusion_matrix(y_true, y_pred)
        # Ensure 2x2 shape
        if cm.shape == (1, 1):
            # Only one class present
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            cm_full = np.zeros((2, 2), dtype=int)
            if 1 in unique_true or 1 in unique_pred:
                # Attack class present
                cm_full[1, 1] = cm[0, 0] if len(unique_true) == 1 and len(unique_pred) == 1 else cm[0, 0]
            else:
                # Benign class present
                cm_full[0, 0] = cm[0, 0]
            cm = cm_full
        elif cm.shape != (2, 2):
            # Reshape to 2x2
            cm_full = np.zeros((2, 2), dtype=int)
            min_dim = min(cm.shape[0], 2)
            min_dim2 = min(cm.shape[1], 2)
            cm_full[:min_dim, :min_dim2] = cm[:min_dim, :min_dim2]
            cm = cm_full
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['false_positive_rate'] = 0.0
        metrics['false_negative_rate'] = 0.0
        metrics['true_positive_rate'] = 0.0
        metrics['true_negative_rate'] = 0.0
    
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def evaluate_model_on_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = 'standard'
) -> Dict[str, Any]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model or aggregator
        X_test: Test features
        y_test: Test labels
        model_type: Type of model ('standard', 'ensemble', 'aggregator')
        
    Returns:
        Dictionary with evaluation metrics, including predictions and probabilities
    """
    # Make predictions
    if model_type == 'ensemble':
        # Ensemble aggregator
        y_pred = model.predict(X_test.values)
        # Try to get probabilities
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test.values)[:, 1]  # Probability of positive class
            except:
                y_pred_proba = None
        else:
            y_pred_proba = None
    elif hasattr(model, 'predict'):
        # Standard sklearn model or aggregator.
        #
        # IMPORTANT: On some macOS BLAS/OpenMP builds, sklearn linear-model predict() can trigger
        # a hard floating-point exception inside safe_sparse_dot. If the model exposes linear
        # coefficients, compute predictions manually to avoid BLAS-backed matmul.
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            Xn = np.asarray(X_test, dtype=float)
            w = np.asarray(model.coef_, dtype=float).reshape(-1)
            b = float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0])
            z = (Xn * w).sum(axis=1) + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            y_pred_proba = p
            y_pred = (p >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            # Try to get probabilities
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                except Exception:
                    y_pred_proba = None
            else:
                y_pred_proba = None
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Compute metrics
    metrics = compute_metrics(y_test.values, y_pred)
    
    # Store predictions and probabilities for visualization
    metrics['y_pred'] = y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
    metrics['y_true'] = y_test.values.tolist() if isinstance(y_test.values, np.ndarray) else y_test.values
    if y_pred_proba is not None:
        metrics['y_pred_proba'] = y_pred_proba.tolist() if isinstance(y_pred_proba, np.ndarray) else y_pred_proba
    
    # Add classification report
    metrics['classification_report'] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    
    return metrics


def compare_approaches(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Compare results from different approaches.
    
    Args:
        results: Dictionary mapping approach names to their metrics
        metric: Metric to compare
        
    Returns:
        DataFrame with comparison
    """
    comparison = {}
    for approach, metrics in results.items():
        comparison[approach] = metrics.get(metric, 0.0)
    
    df = pd.DataFrame([comparison]).T
    df.columns = [metric]
    df = df.sort_values(by=metric, ascending=False)
    
    return df


def extract_trust_evolution_metrics(
    client_info: List[Dict[str, Any]],
    trust_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extract trust evolution metrics from client info and trust manager.
    
    Args:
        client_info: List of client information dictionaries
        trust_manager: Optional TrustManager instance with trust histories
        
    Returns:
        Dictionary with trust evolution metrics
    """
    evolution_metrics = {
        'final_trust_statistics': {},
        'trust_evolution': {},
        'client_trends': {},
        'trust_changes': {}
    }
    
    # Extract final trust scores
    trust_scores = [c.get('trust_score', 0) for c in client_info if c.get('trust_score') is not None]
    if trust_scores:
        evolution_metrics['final_trust_statistics'] = {
            'mean': float(np.mean(trust_scores)),
            'std': float(np.std(trust_scores)),
            'min': float(np.min(trust_scores)),
            'max': float(np.max(trust_scores)),
            'median': float(np.median(trust_scores)),
            'count': len(trust_scores)
        }
    
    # Extract trust evolution from performance history if available
    for client in client_info:
        client_id = client.get('client_id', 'unknown')
        perf_history = client.get('performance_history', [])
        
        if perf_history:
            # Extract trust scores over rounds
            trust_over_rounds = [p.get('trust_score', 0) for p in perf_history if 'trust_score' in p]
            if trust_over_rounds:
                evolution_metrics['trust_evolution'][client_id] = {
                    'rounds': list(range(1, len(trust_over_rounds) + 1)),
                    'trust_scores': trust_over_rounds,
                    'initial_trust': trust_over_rounds[0] if trust_over_rounds else 0,
                    'final_trust': trust_over_rounds[-1] if trust_over_rounds else 0,
                    'total_change': trust_over_rounds[-1] - trust_over_rounds[0] if len(trust_over_rounds) > 1 else 0
                }
                
                # Calculate trend
                if len(trust_over_rounds) >= 2:
                    trend = client.get('trend', 'stable')
                    evolution_metrics['client_trends'][client_id] = trend
                    
                    # Calculate total change
                    total_change = trust_over_rounds[-1] - trust_over_rounds[0]
                    evolution_metrics['trust_changes'][client_id] = {
                        'total_change': float(total_change),
                        'percent_change': float((total_change / trust_over_rounds[0] * 100) if trust_over_rounds[0] > 0 else 0),
                        'trend': trend
                    }
    
    # Extract from TrustManager if available
    if trust_manager and hasattr(trust_manager, 'trust_histories'):
        for client_id, history in trust_manager.trust_histories.items():
            if client_id not in evolution_metrics['trust_evolution']:
                trust_scores = history.trust_scores
                if len(trust_scores) > 1:
                    evolution_metrics['trust_evolution'][client_id] = {
                        'rounds': history.round_numbers[1:],  # Skip initial round 0
                        'trust_scores': trust_scores[1:],
                        'initial_trust': trust_scores[0],
                        'final_trust': trust_scores[-1],
                        'total_change': trust_scores[-1] - trust_scores[0]
                    }
                    
                    # Calculate statistics
                    if len(trust_scores) > 1:
                        trust_changes = [trust_scores[i] - trust_scores[i-1] for i in range(1, len(trust_scores))]
                        evolution_metrics['trust_changes'][client_id] = {
                            'total_change': float(trust_scores[-1] - trust_scores[0]),
                            'percent_change': float((trust_scores[-1] - trust_scores[0]) / trust_scores[0] * 100) if trust_scores[0] > 0 else 0,
                            'avg_change_per_round': float(np.mean(trust_changes)),
                            'max_increase': float(np.max(trust_changes)) if trust_changes else 0,
                            'max_decrease': float(np.min(trust_changes)) if trust_changes else 0,
                            'trend': history.get_trend()
                        }
    
    return evolution_metrics


def generate_results_summary(
    centralized_results: Dict[str, Any],
    federated_results: Dict[str, Any],
    trust_aware_results: Dict[str, Any],
    client_info: Optional[List[Dict[str, Any]]] = None,
    trust_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive results summary.
    
    Args:
        centralized_results: Results from centralized learning
        federated_results: Results from standard federated learning
        trust_aware_results: Results from trust-aware federated learning
        client_info: Optional list of client information dictionaries
        trust_manager: Optional TrustManager instance for trust evolution metrics
        
    Returns:
        Dictionary with summary
    """
    summary = {
        'approaches': {
            'centralized': centralized_results,
            'federated_equal_weight': federated_results,
            'trust_aware': trust_aware_results
        },
        'comparison': {
            'accuracy': {
                'centralized': centralized_results.get('accuracy', 0),
                'federated_equal_weight': federated_results.get('accuracy', 0),
                'trust_aware': trust_aware_results.get('accuracy', 0)
            },
            'f1_score': {
                'centralized': centralized_results.get('f1_score', 0),
                'federated_equal_weight': federated_results.get('f1_score', 0),
                'trust_aware': trust_aware_results.get('f1_score', 0)
            },
            'false_positive_rate': {
                'centralized': centralized_results.get('false_positive_rate', 0),
                'federated_equal_weight': federated_results.get('false_positive_rate', 0),
                'trust_aware': trust_aware_results.get('false_positive_rate', 0)
            }
        }
    }
    
    if client_info:
        summary['clients'] = client_info
        # Compute trust statistics
        trust_scores = [c.get('trust_score', 0) for c in client_info if c.get('trust_score') is not None]
        if trust_scores:
            summary['trust_statistics'] = {
                'mean': float(np.mean(trust_scores)),
                'std': float(np.std(trust_scores)),
                'min': float(np.min(trust_scores)),
                'max': float(np.max(trust_scores)),
                'median': float(np.median(trust_scores))
            }
        
        # Extract trust evolution metrics
        trust_evolution = extract_trust_evolution_metrics(client_info, trust_manager)
        if trust_evolution['trust_evolution'] or trust_evolution['final_trust_statistics']:
            summary['trust_evolution'] = trust_evolution
    
    # Add trust statistics from results if available (multi-round mode)
    if 'trust_statistics' in trust_aware_results:
        summary['trust_statistics'] = trust_aware_results['trust_statistics']
    
    return summary


def compute_response_metrics(simulator_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Summarize response simulation history."""
    if not simulator_history:
        return {
            "response_precision": 0.0,
            "fp_response_rate": 0.0,
            "cumulative_defense_utility": 0.0,
            "violation_rate": 0.0,
        }
    executed = [h for h in simulator_history if h.get("executed")]
    correct = sum(
        1 for h in executed
        if h.get("tp_contain") or (not h.get("fp_response") and not h.get("fn_miss"))
    )
    precision = correct / len(executed) if executed else 0.0
    fp_rate = sum(1 for h in simulator_history if h.get("fp_response")) / len(simulator_history)
    utility = sum(h.get("reward", 0.0) for h in simulator_history)
    viol = sum(1 for h in simulator_history if h.get("policy_violation")) / len(simulator_history)
    return {
        "response_precision": float(precision),
        "fp_response_rate": float(fp_rate),
        "cumulative_defense_utility": float(utility),
        "violation_rate": float(viol),
    }


def estimate_param_payload_bytes(parameters: Any) -> int:
    """Approximate on-wire size of FL model parameters (float payloads only)."""
    if parameters is None:
        return 0
    if isinstance(parameters, dict):
        return int(sum(estimate_param_payload_bytes(v) for v in parameters.values()))
    if isinstance(parameters, (list, tuple)):
        return int(sum(estimate_param_payload_bytes(v) for v in parameters))
    arr = np.asarray(parameters)
    if arr.dtype == object:
        return int(sum(estimate_param_payload_bytes(v) for v in arr.ravel()))
    # Prefer float64 nbytes when numeric; fall back to 8 bytes per scalar.
    if np.issubdtype(arr.dtype, np.number):
        return int(arr.astype(np.float64, copy=False).nbytes)
    return 8


def estimate_update_payload_bytes(update: Dict[str, Any]) -> int:
    """Uplink cost for one client update (parameters or masked vector; exclude train data)."""
    if update.get("masked_vector") is not None:
        mv = np.asarray(update["masked_vector"])
        return int(mv.nbytes) + int(update.get("share_blob_bytes") or 0)
    return estimate_param_payload_bytes(update.get("parameters"))


def summarize_resource_metrics(round_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate round latency and communication cost from per-round resource blocks."""
    latencies = []
    uplink = []
    downlink = []
    messages = []
    for log in round_logs:
        res = log.get("resources") or {}
        if res.get("round_latency_sec") is not None:
            latencies.append(float(res["round_latency_sec"]))
        if res.get("uplink_bytes") is not None:
            uplink.append(float(res["uplink_bytes"]))
        if res.get("downlink_bytes") is not None:
            downlink.append(float(res["downlink_bytes"]))
        if res.get("messages") is not None:
            messages.append(int(res["messages"]))
    total_up = float(sum(uplink)) if uplink else 0.0
    total_down = float(sum(downlink)) if downlink else 0.0
    return {
        "rounds_measured": len(latencies),
        "mean_round_latency_sec": float(np.mean(latencies)) if latencies else None,
        "total_round_latency_sec": float(sum(latencies)) if latencies else None,
        "total_uplink_bytes": total_up,
        "total_downlink_bytes": total_down,
        "total_communication_bytes": total_up + total_down,
        "mean_messages_per_round": float(np.mean(messages)) if messages else None,
        "resource_aware_measured": bool(latencies),
    }


def summarize_trust_metrics(trust_manager: Optional[Any] = None) -> Dict[str, Any]:
    """Compact trust summary for metrics JSON (mean/std/min/max of final scores)."""
    if trust_manager is None or not hasattr(trust_manager, "get_all_trust_scores"):
        return {"available": False}
    scores = list(trust_manager.get_all_trust_scores().values())
    if not scores:
        return {"available": False, "count": 0}
    arr = np.asarray(scores, dtype=float)
    return {
        "available": True,
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "scores": {k: float(v) for k, v in trust_manager.get_all_trust_scores().items()},
    }


def compute_natural_trust_discrimination(
    round_logs: List[Dict[str, Any]],
    *,
    compromise_round: int,
    attacker_ids: Optional[List[str]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Natural-partition trust metrics (Trusted-plan §11 / WP5).

    Uses score s_i = 1 − T_i vs attacker ground truth after t_a.
    Detection delay: first t≥t_a where every attacker is in top-k by s_i
    (or min rank of attackers ≤ k — we use: all attackers appear in top-k suspicious).
    """
    out: Dict[str, Any] = {
        "available": False,
        "compromise_round": int(compromise_round),
        "top_k": int(top_k),
    }
    if not round_logs or compromise_round is None:
        return out

    # Infer attackers from logs if not provided
    attackers: set = set(attacker_ids or [])
    if not attackers:
        for log in round_logs:
            for cid, row in (log.get("client_trust") or {}).items():
                if row.get("is_attacker"):
                    attackers.add(cid)
    attackers = set(attackers)
    out["attacker_ids"] = sorted(attackers)
    if not attackers:
        return out

    post = [log for log in round_logs if int(log.get("round", 0)) >= int(compromise_round)]
    if not post:
        return out

    def _snapshot(log: Dict[str, Any]) -> Dict[str, float]:
        ct = log.get("client_trust") or {}
        return {cid: float(row.get("T", 0.0)) for cid, row in ct.items()}

    # AUROC / AUPRC at round 30 if present, else last post-compromise round
    target_round = None
    for prefer in (30, max(int(l.get("round", 0)) for l in post)):
        for log in post:
            if int(log.get("round", -1)) == prefer:
                target_round = log
                break
        if target_round is not None:
            break
    if target_round is None:
        target_round = post[-1]

    trusts = _snapshot(target_round)
    y_true = []
    scores = []
    for cid, t in trusts.items():
        y_true.append(1 if cid in attackers else 0)
        scores.append(1.0 - float(t))
    y_true_a = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores, dtype=float)

    auroc = None
    auprc = None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(np.unique(y_true_a)) > 1:
            auroc = float(roc_auc_score(y_true_a, scores_a))
            auprc = float(average_precision_score(y_true_a, scores_a))
    except Exception as exc:
        out["sklearn_error"] = str(exc)

    # Detection delay
    detect_round = None
    for log in post:
        trusts_r = _snapshot(log)
        if not trusts_r:
            continue
        ranked = sorted(trusts_r.keys(), key=lambda c: 1.0 - trusts_r[c], reverse=True)
        top = set(ranked[:top_k])
        if attackers.issubset(top):
            detect_round = int(log.get("round"))
            break
    delay = None if detect_round is None else int(detect_round - compromise_round)

    # False distrust at evaluation round: benign in top-k
    ranked_eval = sorted(trusts.keys(), key=lambda c: 1.0 - trusts[c], reverse=True)
    top_eval = ranked_eval[:top_k]
    false_distrust_ids = [c for c in top_eval if c not in attackers]
    false_distrust_rate = float(len(false_distrust_ids) / max(top_k, 1))

    # α aggregates
    alphas_mal, alphas_ben = [], []
    ct = target_round.get("client_trust") or {}
    for cid, row in ct.items():
        a = row.get("alpha")
        if a is None:
            continue
        (alphas_mal if cid in attackers else alphas_ben).append(float(a))

    out.update({
        "available": True,
        "eval_round": int(target_round.get("round")),
        "auroc": auroc,
        "auprc": auprc,
        "detection_round": detect_round,
        "detection_delay": delay,
        "false_distrust_ids": false_distrust_ids,
        "false_distrust_count": len(false_distrust_ids),
        "false_distrust_rate": false_distrust_rate,
        "mean_alpha_malicious": float(np.mean(alphas_mal)) if alphas_mal else None,
        "mean_alpha_benign": float(np.mean(alphas_ben)) if alphas_ben else None,
        "trust_eval": {k: float(v) for k, v in trusts.items()},
    })
    return out


def delta_f1_robust(f1_trust: Optional[float], f1_fedavg: Optional[float]) -> Optional[float]:
    """ΔF1_robust = F1_TrustFed − F1_FedAvg under the same poison setting."""
    if f1_trust is None or f1_fedavg is None:
        return None
    return float(f1_trust) - float(f1_fedavg)


def compute_trust_recovery(
    round_logs: List[Dict[str, Any]],
    *,
    compromise_round: int,
    attacker_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    On–off recovery (WP7): compare mean attacker T during attack vs recovery phases.

    recovery_delta = mean(T_attackers | recovery) − mean(T_attackers | attack)
    Positive ⇒ trust rises after attackers return to benign.
    """
    out: Dict[str, Any] = {"available": False, "compromise_round": int(compromise_round)}
    attackers: set = set(attacker_ids or [])
    if not attackers:
        for log in round_logs:
            for cid, row in (log.get("client_trust") or {}).items():
                if row.get("is_attacker"):
                    attackers.add(cid)
    if not attackers or not round_logs:
        return out

    attack_ts: List[float] = []
    recovery_ts: List[float] = []
    warmup_ts: List[float] = []
    for log in round_logs:
        r = int(log.get("round", 0))
        ct = log.get("client_trust") or {}
        mean_att = []
        for cid in attackers:
            if cid in ct and ct[cid].get("T") is not None:
                mean_att.append(float(ct[cid]["T"]))
        if not mean_att:
            continue
        m = float(np.mean(mean_att))
        if r < compromise_round:
            warmup_ts.append(m)
        elif log.get("recovery_phase") or (
            log.get("attack_active") is False and r >= compromise_round
        ):
            recovery_ts.append(m)
        elif log.get("attack_active") is True or (
            log.get("compromise_active") and log.get("attack_active") is not False
        ):
            attack_ts.append(m)

    if not attack_ts or not recovery_ts:
        out["reason"] = "need_both_attack_and_recovery_phases"
        out["n_attack_rounds"] = len(attack_ts)
        out["n_recovery_rounds"] = len(recovery_ts)
        return out

    mean_attack = float(np.mean(attack_ts))
    mean_recovery = float(np.mean(recovery_ts))
    mean_warmup = float(np.mean(warmup_ts)) if warmup_ts else None
    out.update({
        "available": True,
        "attacker_ids": sorted(attackers),
        "mean_T_attackers_warmup": mean_warmup,
        "mean_T_attackers_attack": mean_attack,
        "mean_T_attackers_recovery": mean_recovery,
        "recovery_delta": mean_recovery - mean_attack,
        "recovery_vs_warmup": (
            None if mean_warmup is None else mean_recovery - mean_warmup
        ),
        "n_attack_rounds": len(attack_ts),
        "n_recovery_rounds": len(recovery_ts),
    })
    return out


def summarize_robustness_metrics(
    round_logs: List[Dict[str, Any]],
    *,
    adversary_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detection stability + adversary exposure for Contrib. 6 robustness reporting."""
    f1s = []
    for log in round_logs:
        det = log.get("detection") or {}
        f1 = det.get("f1_score", det.get("f1"))
        if f1 is not None:
            f1s.append(float(f1))
    out: Dict[str, Any] = {
        "detection_f1_mean": float(np.mean(f1s)) if f1s else None,
        "detection_f1_std": float(np.std(f1s)) if f1s else None,
        "detection_f1_final": f1s[-1] if f1s else None,
        "detection_f1_min": float(np.min(f1s)) if f1s else None,
    }
    if adversary_summary:
        out["adversary"] = adversary_summary
    return out


def resolve_metrics_run_id(
    approach: str,
    *,
    enable_lambda5: bool = True,
    adversary_mode: str = "static",
    dataset: str = "iomt",
    disable_cr_signals: bool = False,
    uniform_contextual_prior: bool = False,
    late_compromise_tag: Optional[str] = None,
    include_R_in_T: bool = False,
) -> str:
    """Unique run id for metrics filenames (avoids B5 / B5−S / B5−λ overwrite)."""
    if approach in ("trustfed_b1v", "b1v", "b1_v_only"):
        if uniform_contextual_prior:
            run_id = (
                f"trustfed_b1v_uniform_{dataset}"
                if dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13")
                else "trustfed_b1v_uniform"
            )
        elif dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13"):
            run_id = f"trustfed_b1v_{dataset}"
        else:
            run_id = "trustfed_b1v"
    elif approach in ("trustfed_b2", "b2", "behavioural"):
        if uniform_contextual_prior:
            run_id = (
                f"trustfed_b2_uniform_{dataset}"
                if dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13")
                else "trustfed_b2_uniform"
            )
        elif dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13"):
            run_id = f"trustfed_b2_{dataset}"
        else:
            run_id = "trustfed_b2"
    elif approach in ("trustfed_b1", "b1"):
        if uniform_contextual_prior:
            if dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13"):
                run_id = f"trustfed_b1_uniform_{dataset}"
            else:
                run_id = "trustfed_b1_uniform"
        elif dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13"):
            run_id = f"trustfed_b1_{dataset}"
        else:
            run_id = "trustfed_b1"
    elif approach in ("trustfed_fedavg", "fedavg", "b0"):
        run_id = (
            f"trustfed_fedavg_{dataset}"
            if dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13")
            else "trustfed_fedavg"
        )
    elif approach in ("trustfed_median", "median", "trustfed_rm", "rm"):
        run_id = (
            f"trustfed_rm_{dataset}"
            if dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13")
            else "trustfed_rm"
        )
    elif approach in ("trustfed_bc", "bc"):
        if uniform_contextual_prior:
            run_id = (
                f"trustfed_bc_uniform_{dataset}"
                if dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13")
                else "trustfed_bc_uniform"
            )
        elif dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13"):
            run_id = f"trustfed_bc_{dataset}"
        else:
            run_id = "trustfed_bc"
    else:
        run_id = approach
        if approach == "trustfed_agent" and disable_cr_signals:
            run_id = "trustfed_agent_no_cr"
        elif approach == "trustfed_agent" and not enable_lambda5:
            run_id = "trustfed_agent_no_lambda5"
        elif approach == "trustfed_agent" and include_R_in_T:
            run_id = "trustfed_agent_b5r"
        if uniform_contextual_prior:
            run_id = f"{run_id}_uniform"
        if adversary_mode == "co_adaptive":
            run_id = f"{run_id}_co_adaptive"
        if dataset in ("iomt", "iomt_natural", "wustl_ehms", "ctu13"):
            run_id = f"{run_id}_{dataset}"

    if late_compromise_tag:
        run_id = f"{run_id}_{late_compromise_tag}"
    return run_id


def save_trustfed_agent_metrics(
    metrics: Dict[str, Any],
    out_dir: Path,
    approach: str,
    seed: int,
    run_id: Optional[str] = None,
) -> Path:
    """Write per-run metrics JSON for TrustFed-Agent experiments."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = run_id or metrics.get("run_id") or approach
    path = out_dir / f"run_{key}_seed_{seed}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
