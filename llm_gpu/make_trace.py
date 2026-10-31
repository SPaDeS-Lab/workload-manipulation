"""Write the exact request trace the simulator uses, so simulator and GPU runs replay the same load.
  python llm_gpu/make_trace.py --condition crafted --manip combined --keys 8 --load 0.9 --budget 0.2 --seed 0 --out trace.csv
"""
import argparse, csv
from fairattack.llm import CostModel, Policy, make_trace, estimate_capacity_rps

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="crafted")
    ap.add_argument("--manip", default="combined")
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--tenants", type=int, default=5)
    ap.add_argument("--load", type=float, default=0.9)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=300)
    ap.add_argument("--capacity-rps", type=float, default=None, help="measured on the GPU; default = Gemma 3 12B estimate")
    ap.add_argument("--cost-model", default="llm_gpu/cost_model.json")
    ap.add_argument("--profile", type=int, nargs=2, default=[3000, 800])
    ap.add_argument("--out", default="trace.csv")
    a = ap.parse_args()
    cm = CostModel.load(a.cost_model)
    cap = a.capacity_rps or estimate_capacity_rps(cm)
    tr = make_trace(a.seed, a.tenants, a.horizon, a.load * cap / a.tenants, condition=a.condition,
                    budget=a.budget, manip=a.manip, n_keys=a.keys, crafted_profile=tuple(a.profile))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "owner", "key", "n_in", "n_out"]); w.writerows(tr)
    print(f"{len(tr)} requests, capacity {cap:.2f} rps -> {a.out}")
