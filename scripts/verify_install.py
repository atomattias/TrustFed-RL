#!/usr/bin/env python3
"""Verify TrustFed-Agent imports and core dependencies."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHECKS = []


def check(name: str, fn) -> None:
    try:
        fn()
        CHECKS.append((name, True, ""))
    except Exception as e:
        CHECKS.append((name, False, str(e)))


def main() -> int:
    check("numpy", lambda: __import__("numpy"))
    check("pandas", lambda: __import__("pandas"))
    check("sklearn", lambda: __import__("sklearn"))
    check("scipy", lambda: __import__("scipy"))

    check("agents", lambda: __import__("agents.autonomous_agent"))
    check("governance", lambda: __import__("governance.policy_engine"))
    check("adversary", lambda: __import__("adversary.co_adaptive"))
    check("federated_server", lambda: __import__("federated_server"))
    check("federated_client", lambda: __import__("federated_client"))
    check("agent_experiment_runner", lambda: __import__("agent_experiment_runner"))

    # RL stack (required for B4/B5 PPO experiments)
    for name, mod in [
        ("gymnasium", "gymnasium"),
        ("stable_baselines3", "stable_baselines3"),
        ("sb3_contrib", "sb3_contrib"),
    ]:
        try:
            __import__(mod)
            CHECKS.append((name, True, ""))
        except ImportError as e:
            CHECKS.append((name, False, f"not installed: {e}"))

    failed = [c for c in CHECKS if not c[1]]
    print("TrustFed-Agent install verification\n" + "-" * 40)
    for name, ok, err in CHECKS:
        if ok and err:
            print(f"  {name:<28} SKIP ({err})")
        else:
            status = "OK" if ok else f"FAIL: {err}"
            print(f"  {name:<28} {status}")

    if failed:
        print(f"\n{len(failed)} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
