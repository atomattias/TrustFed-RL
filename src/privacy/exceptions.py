"""Errors for weighted update hiding."""


class PrivacyError(Exception):
    """Base error for the weighted SecAgg path."""


class ShareError(PrivacyError):
    """Too few Shamir shares or inconsistent reconstruction."""


class DimensionError(PrivacyError):
    """Masked vector length does not match the round dimension."""


class PlaintextLeakError(PrivacyError):
    """Privacy-mode payload still contains model parameters or training data."""


class EmptyCohortError(PrivacyError):
    """No online clients submitted a masked vector."""
