#!/usr/bin/env bash
# Kubernetes/Kueue experiments on the kind cluster (run k8s/setup.sh first).
# Each run takes roughly horizon + drain (~8-12 min); 3 loads x 6 configs x 10 seeds is ~ 2-3 days,
# so start with SEEDS=3 and LOADS="0.9" and expand where effects appear.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
SEEDS="${SEEDS:-3}"; LOADS="${LOADS:-0.3 0.6 0.9}"; BUDGET="${BUDGET:-0.2}"; Q="${ATTACKER_QUEUES:-4}"
for seed in $(seq 0 $((SEEDS-1))); do
 for load in $LOADS; do
  python experiments/run_k8s.py --condition baseline --load "$load" --seed "$seed"
  python experiments/run_k8s.py --condition naive    --load "$load" --budget "$BUDGET" --seed "$seed"
  python experiments/run_k8s.py --condition crafted --manip reshape  --load "$load" --budget "$BUDGET" --seed "$seed"
  python experiments/run_k8s.py --condition crafted --manip split    --load "$load" --budget "$BUDGET" --attacker-queues "$Q" --seed "$seed"
  python experiments/run_k8s.py --condition crafted --manip combined --load "$load" --budget "$BUDGET" --attacker-queues "$Q" --seed "$seed"
  python experiments/run_k8s.py --condition mitigated --manip combined --load "$load" --budget "$BUDGET" --attacker-queues "$Q" --seed "$seed"
 done
done
python experiments/analyze.py
