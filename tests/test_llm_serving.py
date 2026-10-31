import os
from fairattack.llm import (
    Gemma3_12B_Config, CostModel, Policy, make_trace, estimate_capacity_rps,
    charged_units, cost_priced_units
)
from llm_gpu.vllm_vtc_proxy import VTC


def test_gemma_3_12b_config():
    cfg = Gemma3_12B_Config()
    assert cfg.model_id == "google/gemma-3-12b-it"
    assert cfg.parameter_count > 12_000_000_000
    assert cfg.max_position_embeddings == 131_072
    assert cfg.num_key_value_heads == 16
    assert cfg.num_attention_heads == 32
    assert cfg.weight_memory_gb > 20.0
    # KV cache per token calculation
    kv_per_tok = cfg.kv_cache_bytes_per_token()
    assert kv_per_tok > 0


def test_gemma_3_12b_cost_model():
    cost_file = os.path.join(os.path.dirname(__file__), "..", "llm_gpu", "cost_model.json")
    cm = CostModel.load(cost_file)
    assert cm.model_name == "google/gemma-3-12b-it"
    assert cm.kv_capacity == 65536

    # Test prefill calculation
    t_pf_short = cm.prefill(512)
    t_pf_long = cm.prefill(4096)
    assert t_pf_long > t_pf_short > 0

    # Test decode step calculation
    t_dec_small = cm.decode_step_time(batch_size=4, total_kv_tokens=4000)
    t_dec_large = cm.decode_step_time(batch_size=32, total_kv_tokens=32000)
    assert t_dec_large > t_dec_small > 0

    # Test capacity estimation
    cap = estimate_capacity_rps(cm, profile=(500, 200), target_concurrency=16)
    assert cap > 0


def test_gemma_3_12b_trace_generation():
    cm = CostModel.load(None)
    trace = make_trace(seed=42, n_tenants=3, horizon=30.0, rate_honest=2.0, condition="crafted",
                       manip="combined", n_keys=4, budget=0.2)
    assert len(trace) > 0
    # Verify tuple structure: (t, owner, key, n_in, n_out)
    for r in trace:
        assert len(r) == 5
        t, owner, key, n_in, n_out = r
        assert 0 <= owner < 3
        assert n_in > 0 and n_out > 0


def test_vtc_scheduler_proxy_logic():
    class Args:
        meter_by = "key"
        charge = "tokens"
        max_inflight = 2
        cost_model = "llm_gpu/cost_model.json"
        gpu_scale = 1e4

    vtc = VTC(Args())
    c1, c2 = "client1", "client2"

    # Enqueue items
    item1 = dict(n_in=100)
    item2 = dict(n_in=200)
    vtc.enqueue(c1, item1)
    vtc.enqueue(c2, item2)

    # Initial counters
    assert vtc.cnt[c1] == 0.0
    assert vtc.cnt[c2] == 0.0

    # Counter lift on arrival when other clients backlogged
    vtc.cnt[c1] = 50.0
    item3 = dict(n_in=50)
    vtc.enqueue("client3", item3)
    assert vtc.cnt["client3"] >= 0.0


def test_owner_collapsed_metering_mitigation():
    """Verify Section 6.3 Owner-Collapsed Metering: All keys of a single tenant collapse into one meter."""
    class ArgsKey:
        meter_by = "key"
        charge = "tokens"
        max_inflight = 4
        cost_model = "llm_gpu/cost_model.json"
        gpu_scale = 1e4

    class ArgsOwner:
        meter_by = "owner"
        charge = "tokens"
        max_inflight = 4
        cost_model = "llm_gpu/cost_model.json"
        gpu_scale = 1e4

    vtc_key = VTC(ArgsKey())
    vtc_owner = VTC(ArgsOwner())

    # Keys belonging to owner 0
    assert vtc_key.client("key_0", "0") != vtc_key.client("key_1", "0")
    # Owner-collapsed metering maps all keys to owner ID
    assert vtc_owner.client("key_0", "0") == vtc_owner.client("key_1", "0") == "o0"


def test_cost_priced_tokens_mitigation():
    """Verify Section 6.3 Cost-Priced Tokens: quadratic prefill & KV-cache memory costs are charged."""
    cm = CostModel.load("llm_gpu/cost_model.json")
    pol_linear = Policy(charge="tokens")
    pol_cost = Policy(charge="cost_priced")

    # Short request vs Long context request
    short_in, short_out = 500, 200
    long_in, long_out = 4000, 200

    # Linear token pricing ratio
    ratio_linear = charged_units(long_in, long_out, pol_linear) / charged_units(short_in, short_out, pol_linear)
    # Cost-priced ratio includes quadratic prefill and KV-cache footprint
    ratio_cost = charged_units(long_in, long_out, pol_cost, cm) / charged_units(short_in, short_out, pol_cost, cm)

    # Cost-priced ratio charges long prompts significantly more than linear tokens
    assert ratio_cost > ratio_linear
