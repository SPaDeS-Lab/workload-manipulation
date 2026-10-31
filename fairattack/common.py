"""Shared helpers: long-format result writer, statistics, seeding.

Every experiment script writes rows with the same schema so that a single
analysis script can build all figures:

    domain, condition, manipulation, load, budget, n_tenants, param, seed, metric, value

condition    : baseline | naive | crafted | mitigated
manipulation : none | reshape | split | combined | ... (domain specific)
load         : fraction of measured capacity (domain specific, see README)
budget       : attacker extra workload as a fraction of one honest tenant's demand
param        : free-form "k=v;k=v" string for sweep variables (keys, aux, lead_ms, ...)
"""
from __future__ import annotations
import csv, math, os
from typing import Dict, Iterable
import numpy as np

FIELDS = ["domain", "condition", "manipulation", "load", "budget",
          "n_tenants", "param", "seed", "metric", "value"]


class ResultWriter:
    """Append-only CSV writer (header written once)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(FIELDS)

    def emit(self, meta: Dict, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue
            self._w.writerow([meta["domain"], meta["condition"], meta.get("manipulation", "none"),
                              meta.get("load", ""), meta.get("budget", 0.0), meta.get("n_tenants", ""),
                              meta.get("param", ""), meta["seed"], k, float(v)])
        self._f.flush()

    def close(self):
        self._f.close()


def pstr(**kw) -> str:
    """Build the 'param' string deterministically."""
    return ";".join(f"{k}={kw[k]}" for k in sorted(kw))


def parse_param(s: str) -> Dict[str, str]:
    out = {}
    for part in str(s).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def mean_ci(x: Iterable[float], conf: float = 0.95):
    """Mean and half-width of the t confidence interval."""
    a = np.asarray(list(x), dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(a[0]), float("nan")
    from scipy import stats
    h = stats.sem(a) * stats.t.ppf((1 + conf) / 2, n - 1)
    return float(a.mean()), float(h)


def jain(x) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.sum(x ** 2) == 0:
        return float("nan")
    return float(np.sum(x) ** 2 / (len(x) * np.sum(x ** 2)))


def gain(u: float, e: float) -> float:
    """Attacker gain over entitlement, G = (u - e) / e."""
    return (u - e) / e if e > 0 else float("nan")


def pct(x, q):
    x = np.asarray(list(x), dtype=float)
    return float(np.percentile(x, q)) if len(x) else float("nan")


def rng_for(seed: int, *salt: int) -> np.random.Generator:
    """Independent, reproducible stream per (seed, salt...). Paired runs share the same
    honest-workload stream because the attacker uses a different salt."""
    return np.random.default_rng(np.random.SeedSequence([seed, *salt]))


def relative_reduction(m_mitigated: float, m_crafted: float) -> float:
    """Relative reduction Delta_M = 1 - M_mitigated / M_crafted (Section 6.4).
    100% (1.0) indicates full mitigation; negative values indicate degradation.
    """
    if m_crafted == 0:
        return 0.0 if m_mitigated == 0 else -float("inf")
    return float(1.0 - (m_mitigated / m_crafted))

