"""Fit the simulator's GPU cost model against a running vLLM OpenAI-compatible server.

  vllm serve <model> --port 8000 --enable-prefix-caching=false   # prefix caching OFF, or prompts repeat for free
  python llm_gpu/calibrate_cost_model.py --url http://localhost:8000 --model <model> --kv-capacity 100000

Measures
  prefill:  TTFT of a single request with n_in tokens, max_tokens=1      -> c_pf, c_pf2
  decode :  mean inter-token latency with B concurrent streams, context ~n_in -> c0, c_seq, c_kv
Writes llm_gpu/cost_model.json; pass it to experiments/run_llm.py --cost-model.
"""
import argparse, asyncio, json, time
import aiohttp
import numpy as np


async def stream(session, url, model, n_in, n_out):
    body = dict(model=model, prompt="hello " * n_in, max_tokens=n_out, stream=True,
                temperature=0.0, ignore_eos=True)
    t0 = time.time()
    ts = []
    async with session.post(f"{url}/v1/completions", json=body) as r:
        async for line in r.content:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                ts.append(time.time())
    return t0, ts


async def prefill_points(session, a):
    xs, ys = [], []
    for n in (128, 512, 1024, 2048, 4096, 8192):
        for _ in range(3):
            t0, ts = await stream(session, a.url, a.model, n, 1)
            xs.append(n); ys.append(ts[0] - t0)
    A = np.c_[xs, np.square(xs)]
    coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return max(coef[0], 1e-7), max(coef[1], 0.0)


async def decode_points(session, a):
    rows = []
    for B in (1, 4, 16, 32, 64):
        for n_in in (128, 1024, 4096):
            res = await asyncio.gather(*[stream(session, a.url, a.model, n_in, 128) for _ in range(B)])
            itl = np.mean([np.diff(ts[len(ts) // 4:]).mean() for _, ts in res if len(ts) > 8])
            rows.append((B, B * (n_in + 64), itl))
    A = np.array([[1, B, K] for B, K, _ in rows])
    y = np.array([r[2] for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [max(c, 0.0) for c in coef], rows


async def main(a):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        await stream(s, a.url, a.model, 64, 8)                 # warm-up
        c_pf, c_pf2 = await prefill_points(s, a)
        (c0, c_seq, c_kv), rows = await decode_points(s, a)
    out = dict(c0=c0, c_seq=c_seq, c_kv=c_kv, c_pf=c_pf, c_pf2=c_pf2,
               kv_capacity=a.kv_capacity, max_batch=a.max_batch, max_prefill_tokens=a.max_prefill_tokens)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    # fit quality
    pred = np.array([c0 + c_seq * B + c_kv * K for B, K, _ in rows])
    obs = np.array([r[2] for r in rows])
    print("decode-step fit: median relative error", float(np.median(np.abs(pred - obs) / obs)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="google/gemma-3-12b-it", help="Model name (default: google/gemma-3-12b-it)")
    ap.add_argument("--kv-capacity", type=int, default=65536, help="Tokens of KV cache (default: 65536 for Gemma 3 12B on 80GB GPU)")
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--max-prefill-tokens", type=int, default=8192)
    ap.add_argument("--out", default="llm_gpu/cost_model.json")
    asyncio.run(main(ap.parse_args()))
