#!/usr/bin/env bash
# Path STRAG T1.1 smoke (revised): path_strag adversary + short post-t_a window.
# Not confirmatory. Does not overwrite H1 / Path1 / Path AGG / hard adversary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
SEED="${SEED:-42}"
# Short smoke (not confirmatory): t_a=3, rounds=5 — exercises post-t_a skips quickly
ROUNDS="${ROUNDS:-5}"
TA="${TA:-3}"
ADV="${ADV:-config/iomt_natural_adversary_path_strag.json}"
SUFF="${SUFF:-path_strag_smoke}"
METRICS="$ROOT/results/trustfed_agent/metrics"

echo "=== Path STRAG smoke seed=$SEED rounds=$ROUNDS ta=$TA adv=$ADV suffix=$SUFF ==="

# Force re-run: remove prior smoke artifact only (never bare H1 / path_agg)
shopt -s nullglob
OLD=("$METRICS"/run_*${SUFF}_seed_${SEED}.json)
for f in "${OLD[@]:-}"; do
  [[ -n "$f" ]] || continue
  echo "removing prior smoke $f"
  rm -f "$f"
done

"$PY" run_experiments.py regression \
  --dataset iomt_natural \
  --baseline b2 \
  --seed "$SEED" \
  --num-rounds "$ROUNDS" \
  --compromise-round "$TA" \
  --late-compromise-config "$ADV" \
  --uniform-b-prior \
  --metrics-suffix "$SUFF"

FILES=("$METRICS"/run_trustfed_b2_uniform_iomt_natural_*_strag_*${SUFF}_seed_${SEED}.json)
if [[ ${#FILES[@]} -eq 0 ]]; then
  FILES=("$METRICS"/run_*${SUFF}_seed_${SEED}.json)
fi
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: no smoke metrics found under $METRICS" >&2
  exit 1
fi

"$PY" - <<PY
import json
from pathlib import Path
p = Path(r"""${FILES[0]}""")
d = json.loads(p.read_text())
assert "path_strag_smoke" in p.name, p.name
lc = d.get("late_compromise") or {}
bs = lc.get("benign_straggler") or {}
assert bs.get("enabled") is True
strag = set(bs.get("straggler_client_ids") or [])
att = set(lc.get("attacker_client_ids") or [])
assert strag == {"client_09", "client_10", "client_11"}, strag
assert att == {"client_01", "client_06", "client_12"}, att
assert strag.isdisjoint(att)

disc = d.get("trust_discrimination") or {}
assert disc.get("available") is True
assert set(disc.get("attacker_ids") or []) == att
assert set(disc.get("attacker_ids") or []).isdisjoint(strag)

# Post-t_a rounds: attackers always skip; ≥1 benign_straggler skip appears
ta = int(lc.get("compromise_round") or 3)
logs = d.get("round_logs") or []
assert logs, "missing round_logs"
post = [r for r in logs if int(r.get("round", 0)) >= ta]
assert post, "no post-t_a rounds"
attacker_skip_ok = True
benign_skips = 0
for r in post:
    ct = r.get("client_trust") or {}
    for cid, row in ct.items():
        reason = row.get("skip_reason")
        if cid in att:
            if reason != "attacker_comm_skip" or float(row.get("alpha") or 0) != 0.0:
                attacker_skip_ok = False
        if reason == "benign_straggler":
            benign_skips += 1
            assert cid in strag and not row.get("is_attacker")
            assert float(row.get("alpha") or 0) == 0.0
assert attacker_skip_ok, "attacker always-skip failed"
assert benign_skips >= 1, "expected ≥1 benign_straggler skip in smoke window"

print("smoke_file", p.name)
print("auroc", disc.get("auroc"), "benign_skips", benign_skips)
print("T1.1 smoke QA PASS")
PY
echo "=== Path STRAG smoke done ==="
