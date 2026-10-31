"""Replay a trace against the VTC proxy, sample GPU/KV metrics, write results in the common schema.

  python llm_gpu/loadgen.py --trace trace.csv --proxy http://localhost:9000 --vllm http://localhost:8000 \
         --model <model> --condition crafted --manip combined --load 0.9 --budget 0.2 --seed 0 --tenants 5
"""
import argparse, asyncio, csv, json, re, subprocess, time
import aiohttp
import numpy as np
from fairattack.common import ResultWriter, pstr, pct, gain, jain
from fairattack.llm import CostModel


async def one(sess, a, row, t0, out):
    t, owner, key, n_in, n_out = row
    await asyncio.sleep(max(0.0, t0 + t - time.time()))
    body = dict(model=a.model, prompt="hello " * n_in, max_tokens=n_out, stream=True, ignore_eos=True, temperature=0.0)
    ts0, first, n = time.time(), None, 0
    try:
        async with sess.post(f"{a.proxy}/v1/completions", json=body,
                             headers={"X-Key": key, "X-Owner": str(owner), "X-Input-Tokens": str(n_in)}) as r:
            async for line in r.content:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    n += 1
                    first = first or time.time()
    except Exception:
        pass
    out.append(dict(owner=owner, key=key, n_in=n_in, n_out=n_out, t=t, ttft=(first - ts0) if first else None,
                    e2e=time.time() - ts0, toks=n))


async def sampler(a, stop, rows):
    async with aiohttp.ClientSession() as s:
        while not stop.is_set():
            row = dict(t=time.time(), util=None, mem=None, kv=None, running=None)
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
                row["util"], row["mem"] = float(o[0]), float(o[1])
                txt = await (await s.get(f"{a.vllm}/metrics")).text()
                for name, k in (("gpu_cache_usage_perc", "kv"), ("num_requests_running", "running")):
                    m = re.search(rf"vllm:{name}[^ ]* ([0-9.e+-]+)", txt)
                    row[k] = float(m.group(1)) if m else None
            except Exception:
                pass
            rows.append(row)
            await asyncio.sleep(1.0)


async def main(a):
    trace = [(float(r["t"]), int(r["owner"]), r["key"], int(r["n_in"]), int(r["n_out"]))
             for r in csv.DictReader(open(a.trace))]
    out, samples, stop = [], [], asyncio.Event()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        smp = asyncio.create_task(sampler(a, stop, samples))
        t0 = time.time()
        await asyncio.gather(*[one(s, a, r, t0, out) for r in trace])
        stop.set(); await smp
    warm = a.warmup
    cm = CostModel.load(a.cost_model)
    n = a.tenants
    ch = np.zeros(n); gpu_model = np.zeros(n)
    for r in out:
        if warm <= r["t"] <= a.horizon:
            ch[r["owner"]] += r["n_in"] + 2 * r["toks"]
            # model-based GPU estimate (cannot be measured per client on a shared GPU)
            gpu_model[r["owner"]] += cm.prefill(r["n_in"]) + r["toks"] * (cm.c0 / 32 + cm.c_seq + cm.c_kv * (r["n_in"] + r["toks"] / 2))
    sc, sg = ch / max(ch.sum(), 1), gpu_model / max(gpu_model.sum(), 1e-9)
    hon = [r for r in out if r["owner"] != 0 and warm <= r["t"] <= a.horizon]
    tt = [r["ttft"] if r["ttft"] is not None else 1e3 for r in hon]
    e2 = [r["e2e"] for r in hon]
    ok = lambda k: [s[k] for s in samples if s[k] is not None and warm <= s["t"] - samples[0]["t"] <= a.horizon]
    e = 1.0 / n
    m = dict(u_attacker=sc[0], e_attacker=e, gain=gain(sc[0], e), u_attacker_gpu_model=sg[0], gain_gpu_model=gain(sg[0], e),
             honest_service_mean=float(sc[1:].mean()), honest_ttft_p50=pct(tt, 50), honest_ttft_p95=pct(tt, 95),
             honest_e2e_p50=pct(e2, 50), honest_e2e_p95=pct(e2, 95), jain_policy=jain(sc),
             gpu_util_mean=float(np.mean(ok("util"))) if ok("util") else float("nan"),
             gpu_mem_mean_mib=float(np.mean(ok("mem"))) if ok("mem") else float("nan"),
             kv_usage_mean=float(np.mean(ok("kv"))) if ok("kv") else float("nan"),
             attack_cost=float(len({r["key"] for r in out if r["owner"] == 0}) - 1) + a.budget)
    w = ResultWriter(a.out)
    w.emit(dict(domain="llm_gpu", condition=a.condition, manipulation=a.manip, load=a.load, budget=a.budget,
                n_tenants=n, seed=a.seed, param=pstr(meter=a.meter_by)), m)
    w.close()
    json.dump(dict(requests=out, samples=samples), open(a.raw, "w"))
    print(json.dumps({k: round(v, 3) for k, v in m.items() if v == v}))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True); ap.add_argument("--proxy", default="http://localhost:9000")
    ap.add_argument("--vllm", default="http://localhost:8000"); ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--condition", required=True); ap.add_argument("--manip", default="none")
    ap.add_argument("--load", type=float, default=0.9); ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--tenants", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=300); ap.add_argument("--warmup", type=float, default=30)
    ap.add_argument("--meter-by", default="key"); ap.add_argument("--cost-model", default="llm_gpu/cost_model.json")
    ap.add_argument("--out", default="results/llm_gpu.csv"); ap.add_argument("--raw", default="results/llm_gpu_raw.json")
    asyncio.run(main(ap.parse_args()))
