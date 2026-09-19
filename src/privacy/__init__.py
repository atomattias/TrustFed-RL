"""Privacy-preserving aggregation helpers."""

from .dp_sag import (
    DPSAGConfig,
    aggregate_clipped_gradients,
    aggregate_trust_weighted_hessian_grad,
    clip_vector,
    estimate_epsilon,
    irls_newton_step,
)
from .sufficient_stats import (
    compute_logistic_gradient_stat,
    compute_logistic_irls_stats,
    sample_local_batch,
)
from .weighted_secagg import (
    WeightedSecAggRound,
    assert_no_plaintext,
    install_logistic_from_avg,
    privacy_update_from_masked,
    shamir_threshold,
    weighted_payload,
)

__all__ = [
    "DPSAGConfig",
    "aggregate_clipped_gradients",
    "clip_vector",
    "estimate_epsilon",
    "compute_logistic_gradient_stat",
    "sample_local_batch",
    "WeightedSecAggRound",
    "assert_no_plaintext",
    "install_logistic_from_avg",
    "privacy_update_from_masked",
    "shamir_threshold",
    "weighted_payload",
]
