"""Pure functions that turn per-Job records into the common metric set (no cluster needed).

A record is a dict:
  tenant, submit, admit (None if never admitted), start, end (None if unfinished),
  cpu_req, cpu_lim, mem_gi, work_s (useful CPU-seconds), cpu_seconds (measured), queue_idx

Four quantities are kept apart (plan, section 2A):
  requests / limits           -> cpu_req, cpu_lim
  admission accounting        -> accounted CPU-seconds = request x time the quota was held
  runtime consumption         -> measured cpu_seconds, prorated to the observation window
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
from .common import gain, jain, pct


def _ov(a0, a1, w0, w1):
    return max(0.0, min(a1, w1) - max(a0, w0))


def compute_metrics(recs: List[dict], n: int, w0: float, w1: float, cap_cpu: float,
                    attacker_queues: int, extra_jobs: float, budget: float,
                    accounting_mode: str = "declared") -> Dict[str, float]:
    actual = np.zeros(n)
    accounted = np.zeros(n)
    settled = np.zeros(n)
    done = np.zeros(n)
    waits = {i: [] for i in range(n)}
    for r in recs:
        t = r["tenant"]
        if r.get("admit") is not None:
            waits[t].append(r["admit"] - r["submit"])
        if r.get("start") is not None:
            end = r["end"] if r.get("end") is not None else w1
            span = max(end - r["start"], 1e-9)
            cpu = r["cpu_seconds"] if r.get("cpu_seconds") is not None else r["work_s"] * _ov(r["start"], end, w0, w1) / span
            actual[t] += cpu * _ov(r["start"], end, w0, w1) / span
            # Settled accounting: adjusts charge based on measured runtime CPU usage
            actual_rate = (r.get("cpu_seconds") or r.get("work_s") or r.get("cpu_req")) / span
            settled_charge = actual_rate * _ov(r["admit"] if r.get("admit") is not None else r["start"], end, w0, w1)
            settled[t] += settled_charge
        if r.get("admit") is not None:
            end = r["end"] if r.get("end") is not None else w1
            accounted[t] += r["cpu_req"] * _ov(r["admit"], end, w0, w1)
            if r.get("start") is None:
                # Admitted but did not start in observation window: charge request
                settled[t] += r["cpu_req"] * _ov(r["admit"], end, w0, w1)
        if r.get("end") is not None and w0 <= r["end"] <= w1:
            done[t] += 1
    sa = actual / actual.sum() if actual.sum() else np.zeros(n)
    sacc_declared = accounted / accounted.sum() if accounted.sum() else np.zeros(n)
    sacc_settled = settled / settled.sum() if settled.sum() else np.zeros(n)

    # Active accounting selected by mode
    sacc = sacc_settled if accounting_mode == "settled" else sacc_declared

    e = 1.0 / n
    hw = [w for i in range(1, n) for w in waits[i]]
    unfinished_h = [r for r in recs if r["tenant"] != 0 and r.get("end") is None]
    m = dict(u_attacker=sa[0], e_attacker=e, gain=gain(sa[0], e),
             u_attacker_accounted=sacc[0], gain_accounted=gain(sacc[0], e),
             accounting_gap=sa[0] - sacc[0],                       # realised minus accounted share
             u_attacker_accounted_settled=sacc_settled[0],
             gain_accounted_settled=gain(sacc_settled[0], e),
             accounting_gap_settled=sa[0] - sacc_settled[0],
             accounting_gap_declared=sa[0] - sacc_declared[0],
             honest_service_mean=float(sa[1:].mean()) if n > 1 else float("nan"),
             honest_accounted_mean=float(sacc[1:].mean()) if n > 1 else float("nan"),
             honest_wait_p50=pct(hw, 50), honest_wait_p95=pct(hw, 95),
             honest_jobs_done_mean=float(done[1:].mean()) if n > 1 else float("nan"),
             attacker_jobs_done=float(done[0]),
             honest_shortfall=float(max(0.0, e - sa[1:].mean())) if n > 1 else float("nan"),
             utilization_actual=float(actual.sum() / (cap_cpu * (w1 - w0))) if w1 > w0 else float("nan"),
             utilization_requested=float(accounted.sum() / (cap_cpu * (w1 - w0))) if w1 > w0 else float("nan"),
             jain_actual=jain(actual), honest_unfinished=float(len(unfinished_h)))
    att = [r for r in recs if r["tenant"] == 0]
    if att:
        m["attacker_cpu_req_mean"] = float(np.mean([r["cpu_req"] for r in att]))
        m["attacker_cpu_lim_mean"] = float(np.mean([r["cpu_lim"] for r in att]))
    m["attack_cost_submissions"] = float(extra_jobs)
    m["attack_cost_identities"] = float(attacker_queues - 1)
    m["attack_cost"] = float(extra_jobs) / max(1.0, len([r for r in recs if r["tenant"] == 1])) + attacker_queues - 1
    return m
