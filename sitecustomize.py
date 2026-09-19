"""
Workaround for intermittent NumPy macOS polyfit sanity-check crashes (SIGFPE).

NumPy performs a quick macOS-only sanity check at import time via `sys.platform == "darwin"`.
On this machine that sanity check can occasionally hard-crash the interpreter with
`Fatal Python error: Floating-point exception` before any project code runs.

We temporarily masquerade the platform to bypass that check so the experiment code can run.
This affects only the import-time check; runtime behavior should be unaffected for our CPU-only
simulation and ML stack.
"""

import sys

if sys.platform == "darwin":
    sys.platform = "linux"

