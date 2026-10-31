"""VTC scheduler as a dispatch proxy in front of vLLM (OpenAI-compatible).

The proxy keeps at most --max-inflight requests inside vLLM and chooses which waiting request to
forward next by the VTC rule (counter lift on arrival, lowest counter first).

Mitigations implemented (Section 6.3):
  1. Cost-Priced Tokens (--charge cost_priced): Replaces linear token charges with estimated GPU
     costs from the calibrated CostModel (accounting for quadratic prefill, decode batching, and KV cache).
  2. Owner-Collapsed Metering (--meter-by owner): Maintains one shared meter across all API keys
     belonging to the same owner, preventing identity-splitting advantage.
  3. Combined Defense: Applying both mitigations simultaneously restores fair 1/n allocation (Lemma 3).

  python llm_gpu/vllm_vtc_proxy.py --upstream http://localhost:8000 --port 9000 --max-inflight 64 \
         --meter-by owner --charge cost_priced
Clients send POST /v1/completions with headers  X-Key, X-Owner, X-Input-Tokens.
"""
import argparse, asyncio, json
from collections import defaultdict, deque
import aiohttp
from aiohttp import web
from fairattack.llm import CostModel

W_IN, W_OUT = 1.0, 2.0


class VTC:
    def __init__(self, a):
        self.a = a
        self.q = defaultdict(deque)
        self.cnt = defaultdict(float)
        self.last = 0.0
        self.inflight = 0
        self.wake = asyncio.Event()
        self.cm = CostModel.load(a.cost_model) if a.charge == "cost_priced" else CostModel()

    def client(self, key, owner):
        # Owner-collapsed metering (Section 6.3): all keys of an owner share one meter
        return f"o{owner}" if self.a.meter_by == "owner" else key

    def enqueue(self, c, item):
        if not self.q[c]:
            back = [self.cnt[k] for k, d in self.q.items() if d]
            self.cnt[c] = max(self.cnt[c], min(back)) if back else max(self.cnt[c], self.last)
        self.q[c].append(item)
        self.wake.set()

    async def dispatcher(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            while self.inflight < self.a.max_inflight:
                cands = sorted((self.cnt[k], k) for k, d in self.q.items() if d)
                if not cands:
                    break
                _, c = cands[0]
                item = self.q[c].popleft()
                n_in = item["n_in"]
                if self.a.charge == "cost_priced":
                    # Cost-priced prefill charge (targets distortion kappa_a)
                    dispatch_cost = self.cm.prefill_time(n_in) * self.a.gpu_scale
                else:
                    dispatch_cost = W_IN * n_in
                self.cnt[c] += dispatch_cost
                self.last = self.cnt[c]
                self.inflight += 1
                item["go"].set()


async def handle(req: web.Request):
    st: VTC = req.app["vtc"]
    body = await req.read()
    key, owner = req.headers.get("X-Key", "k"), req.headers.get("X-Owner", "0")
    c = st.client(key, owner)
    n_in = int(req.headers.get("X-Input-Tokens", "0"))
    item = dict(n_in=n_in, go=asyncio.Event())
    st.enqueue(c, item)
    await item["go"].wait()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    out_tokens = 0
    try:
        async with req.app["sess"].post(f"{st.a.upstream}/v1/completions", data=body,
                                        headers={"Content-Type": "application/json"}) as up:
            async for line in up.content:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    out_tokens += 1
                    if st.a.charge == "cost_priced":
                        # Cost-priced decode token charging accounting for KV memory & batching
                        step_cost = (st.cm.c0 / max(st.inflight, 1) + st.cm.c_seq + st.cm.c_kv * (n_in + out_tokens)) * st.a.gpu_scale
                        st.cnt[c] += step_cost
                    else:
                        st.cnt[c] += W_OUT          # one streamed chunk ~ one output token
                await resp.write(line)
    finally:
        st.inflight -= 1
        st.wake.set()
    return resp


async def metrics(req):
    st: VTC = req.app["vtc"]
    return web.json_response(dict(counters=st.cnt, inflight=st.inflight,
                                  queued={k: len(d) for k, d in st.q.items()},
                                  meter_by=st.a.meter_by, charge=st.a.charge))


async def on_start(app):
    app["sess"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    app["task"] = asyncio.create_task(app["vtc"].dispatcher())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--max-inflight", type=int, default=64)
    ap.add_argument("--meter-by", choices=["key", "owner"], default="key",
                    help="Metering mode: 'key' (per-key) or 'owner' (owner-collapsed metering)")
    ap.add_argument("--charge", choices=["tokens", "cost_priced"], default="tokens",
                    help="Charging mode: 'tokens' (linear pricing) or 'cost_priced' (GPU cost-priced tokens)")
    ap.add_argument("--cost-model", default="llm_gpu/cost_model.json", help="Path to cost model JSON")
    ap.add_argument("--gpu-scale", type=float, default=1e4, help="Scale factor for GPU cost pricing")
    a = ap.parse_args()
    app = web.Application()
    app["vtc"] = VTC(a)
    app.on_startup.append(on_start)
    app.add_routes([web.post("/v1/completions", handle), web.get("/vtc", metrics)])
    web.run_app(app, port=a.port)
