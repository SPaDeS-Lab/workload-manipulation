#!/usr/bin/env bash
# Deploy and evaluate Google Gemma 3 12B serving under vLLM with the VTC fair dispatch proxy.
# MODEL=google/gemma-3-12b-it KV=65536 bash scripts/run_gpu_validation.sh
set -euo pipefail
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MODEL="${MODEL:-google/gemma-3-12b-it}"
KV="${KV:-65536}"
vllm serve "$MODEL" --port 8000 --enable-prefix-caching=false > results/vllm.log 2>&1 &
VPID=$!; trap 'kill $VPID ${PPID_PROXY:-} 2>/dev/null || true' EXIT
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
python llm_gpu/calibrate_cost_model.py --model "$MODEL" --kv-capacity "$KV"
python llm_gpu/vllm_vtc_proxy.py --port 9000 --max-inflight 64 & PPID_PROXY=$!; sleep 3
for cond in "baseline none 1" "naive none 1" "crafted combined 8"; do
  set -- $cond
  python llm_gpu/make_trace.py --condition "$1" --manip "$2" --keys "$3" --cost-model llm_gpu/cost_model.json --out "results/trace_$1.csv"
  python llm_gpu/loadgen.py --trace "results/trace_$1.csv" --model "$MODEL" --condition "$1" --manip "$2" --seed 0
done
# Metrics written to results/llm_gpu.csv and raw traces to results/llm_gpu_raw.json
