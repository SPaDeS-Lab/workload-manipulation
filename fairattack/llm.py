"""LLM serving module with Google Gemma 3 12B model support and VTC scheduling.

This module provides:
  * Gemma 3 12B model specifications (architecture, context length, KV cache footprints).
  * Cost model calibrated for Gemma 3 12B (prefill quadratic model + continuous batching decode).
  * Virtual Token Counter (VTC) policy and scheduling rules.
  * Workload trace generator for benchmarking fair scheduling and manipulation attacks.
"""
from __future__ import annotations
import json, math, os
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from .common import rng_for, jain, pct, gain

# -----------------------------------------------------------------------------
# Gemma 3 12B Model Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Gemma3_12B_Config:
    """Google Gemma 3 12B model specifications."""
    model_id: str = "google/gemma-3-12b-it"
    alias: str = "google/gemma-3-12b"
    parameter_count: int = 12_200_000_000   # ~12.2B parameters
    hidden_size: int = 3840
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 16            # Grouped Query Attention (GQA)
    head_dim: int = 128
    vocab_size: int = 256_000
    max_position_embeddings: int = 131_072   # 128k context window
    default_serving_max_context: int = 8192
    bytes_per_parameter: int = 2             # bfloat16: ~24.4 GB weights
    weight_memory_gb: float = 24.4

    @staticmethod
    def kv_cache_bytes_per_token() -> int:
        """KV cache bytes per token: 2 (bfloat16) * 2 (K & V) * num_layers * num_kv_heads * head_dim."""
        return 2 * 2 * 40 * 16 * 128  # 327,680 bytes/token (~320 KB/token)


# -----------------------------------------------------------------------------
# Gemma 3 12B Cost Model
# -----------------------------------------------------------------------------
@dataclass
class CostModel:
    """GPU execution cost model calibrated for Google Gemma 3 12B on 80GB VRAM (e.g. A100/H100).

    Formulae:
      prefill_time(n) = c_pf * n + c_pf2 * n^2
      decode_step(B, K) = c0 + c_seq * B + c_kv * K
        B: concurrent sequences in batch
        K: sum of context lengths across active sequences
    """
    model_name: str = "google/gemma-3-12b-it"
    # Latencies in seconds:
    c0: float = 1.2e-2          # 12 ms kernel launch & base decode overhead
    c_seq: float = 1.8e-4       # 0.18 ms per sequence decode step
    c_kv: float = 4.5e-7        # GQA attention memory bandwidth per active KV token
    c_pf: float = 1.8e-4        # Linear prefill scaling (~5500 tokens/sec single stream)
    c_pf2: float = 5.0e-9       # Quadratic attention prefill penalty for long contexts
    kv_capacity: int = 65_536   # KV cache capacity in tokens (~21 GB on top of ~25 GB weights)
    max_batch: int = 128        # Max batch size for continuous batching
    max_prefill_tokens: int = 8192

    @staticmethod
    def load(path: Optional[str]) -> "CostModel":
        if not path or not os.path.exists(path):
            return CostModel()
        with open(path) as f:
            d = json.load(f)
        fields = CostModel.__dataclass_fields__
        return CostModel(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def prefill_time(self, n_tokens: int) -> float:
        """Time in seconds to prefill n_tokens."""
        return self.c_pf * n_tokens + self.c_pf2 * (n_tokens ** 2)

    def prefill(self, n_tokens: int) -> float:
        return self.prefill_time(n_tokens)

    def decode_step_time(self, batch_size: int, total_kv_tokens: int) -> float:
        """Time in seconds for a single decode iteration."""
        return self.c0 + self.c_seq * batch_size + self.c_kv * total_kv_tokens

    def estimate_request_gpu_time(self, n_in: int, n_out: int, avg_batch: int = 32) -> float:
        """Estimated total GPU execution time for a request."""
        t_pf = self.prefill_time(n_in)
        # Average context per decode step is n_in + n_out/2
        avg_kv = avg_batch * (n_in + n_out / 2)
        step_cost = self.c0 / max(avg_batch, 1) + self.c_seq + self.c_kv * avg_kv
        t_decode = n_out * step_cost
        return t_pf + t_decode


# -----------------------------------------------------------------------------
# VTC Policy & Mitigations (Cost-Priced Tokens & Owner-Collapsed Metering)
# -----------------------------------------------------------------------------
@dataclass
class Policy:
    """Virtual Token Counter policy configuration.

    meter_by: 'key' (standard VTC) or 'owner' (mitigation: owner-collapsed metering).
    charge: 'tokens' (linear token pricing) or 'cost_priced' (mitigation: cost-priced tokens).
    """
    meter_by: str = "key"
    charge: str = "tokens"
    w_in: float = 1.0
    w_out: float = 2.0
    gpu_scale: float = 1e4      # Scale factor to make GPU-seconds comparable to token counts


def charged_units(n_in: int, n_out: int, pol: Policy = Policy(), cm: Optional[CostModel] = None) -> float:
    """Compute charged units according to the policy (tokens or cost-priced)."""
    if pol.charge in ("cost_priced", "gpu"):
        model = cm or CostModel()
        return model.estimate_request_gpu_time(n_in, n_out) * pol.gpu_scale
    return pol.w_in * n_in + pol.w_out * n_out


def cost_priced_units(n_in: int, n_out: int, cm: Optional[CostModel] = None, gpu_scale: float = 1e4) -> float:
    """Compute cost-priced units based on estimated GPU time (Section 6.3)."""
    model = cm or CostModel()
    return model.estimate_request_gpu_time(n_in, n_out) * gpu_scale



# -----------------------------------------------------------------------------
# Trace Generation for Workload Evaluation
# -----------------------------------------------------------------------------
def sample_lengths(rng: np.random.Generator, n: int, mean_in: float, mean_out: float,
                   sigma: float = 0.5, cap_in: int = 16000, cap_out: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """Sample prompt and output lengths from a lognormal distribution."""
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    mu_i = np.log(mean_in) - sigma ** 2 / 2
    mu_o = np.log(mean_out) - sigma ** 2 / 2
    li = np.clip(rng.lognormal(mu_i, sigma, n), 8, cap_in).astype(int)
    lo = np.clip(rng.lognormal(mu_o, sigma, n), 1, cap_out).astype(int)
    return li, lo


def poisson_arrivals(rng: np.random.Generator, rate: float, horizon: float) -> np.ndarray:
    if rate <= 0 or horizon <= 0:
        return np.array([], dtype=float)
    n = rng.poisson(rate * horizon)
    return np.sort(rng.uniform(0, horizon, n))


def make_trace(seed: int, n_tenants: int, horizon: float, rate_honest: float, *,
               condition: str = "baseline", budget: float = 0.2, manip: str = "none",
               n_keys: int = 1, honest_profile: Tuple[int, int] = (500, 200),
               crafted_profile: Tuple[int, int] = (3000, 800),
               pol: Policy = Policy()) -> List[Tuple[float, int, str, int, int]]:
    """Generate a time-sorted list of requests: (t, owner, key, n_in, n_out).

    Tenant 0 represents the evaluation/attacker slot.
      * baseline: tenant 0 submits honest workload with honest shape and 1 key.
      * naive: tenant 0 submits (1+budget)x honest request rate with honest shape, 1 key.
      * crafted: tenant 0 submits manipulated workload (reshape/split/combined) with equal
                 offered charged units to the naive attacker.
    """
    rows = []
    for o in range(n_tenants):
        rng = rng_for(seed, 1, o)
        ts = poisson_arrivals(rng, rate_honest, horizon)
        li, lo = sample_lengths(rng, len(ts), *honest_profile)
        if o != 0:
            rows += [(float(t), o, f"{o}:0", int(a), int(b)) for t, a, b in zip(ts, li, lo)]
            continue
        if condition == "baseline":
            rows += [(float(t), 0, "0:0", int(a), int(b)) for t, a, b in zip(ts, li, lo)]
            continue

        arng = rng_for(seed, 2, 0)
        rate_naive = rate_honest * (1 + budget)
        if condition == "naive":
            ts_n = poisson_arrivals(arng, rate_naive, horizon)
            li_n, lo_n = sample_lengths(arng, len(ts_n), *honest_profile)
            rows += [(float(t), 0, "0:0", int(a), int(b)) for t, a, b in zip(ts_n, li_n, lo_n)]
            continue

        reshape = manip in ("reshape", "combined")
        split = manip in ("split", "combined")
        prof = crafted_profile if reshape else honest_profile
        ch_h = charged_units(*honest_profile, pol)
        ch_c = charged_units(*prof, pol)
        rate_c = rate_naive * ch_h / ch_c

        ts_c = poisson_arrivals(arng, rate_c, horizon)
        if reshape:
            li_c, _ = sample_lengths(arng, len(ts_c), prof[0], prof[1], sigma=0.1)
            _, lo_c = sample_lengths(arng, len(ts_c), prof[0], prof[1], sigma=0.1)
        else:
            li_c, lo_c = sample_lengths(arng, len(ts_c), *honest_profile)
        k = max(n_keys if split else 1, 1)
        rows += [(float(t), 0, f"0:{i % k}", int(a), int(b)) for i, (t, a, b) in enumerate(zip(ts_c, li_c, lo_c))]

    rows.sort(key=lambda r: r[0])
    return rows


def estimate_capacity_rps(cm: CostModel, profile: Tuple[int, int] = (500, 200),
                          target_concurrency: int = 32) -> float:
    """Estimate throughput capacity (requests per second) for Gemma 3 12B under continuous batching."""
    n_in, n_out = profile
    single_req_time = cm.estimate_request_gpu_time(n_in, n_out, avg_batch=target_concurrency)
    # Effective throughput with concurrent streams
    effective_rps = target_concurrency / max(single_req_time, 1e-4)
    return float(effective_rps)
