"""Run one Kueue experiment on a live cluster and append metrics to results/k8s.csv.

  python experiments/run_k8s.py --condition crafted --manip combined --load 0.9 --budget 0.2 \
         --attacker-queues 4 --seed 0

Conditions
  baseline  : all tenants honest (attacker slot behaves honestly)
  naive     : attacker submits (1+budget)x the honest job rate with honest-shaped Jobs
  crafted   : same job count as naive, Jobs request cpu 0.5 but limit 4 (reshape),
              and/or spread over --attacker-queues ClusterQueues (split)
  mitigated : crafted workload, but requests := limits (emulated charge-the-limit policy)
Offered load = total accounted CPU of honest Jobs / cohort CPU quota (so load 0.9 ~ 90% busy by requests).
Use scripts/run_k8s_suite.sh to sweep seeds/loads.
"""
import argparse, json, subprocess, threading, time, yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
from fairattack.common import ResultWriter, pstr, rng_for
from fairattack import k8s_manifests as km
from fairattack.k8s_metrics import compute_metrics


def kc(*args, inp=None, check=True):
    r = subprocess.run(["kubectl", *args], input=inp, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr}")
    return r.stdout


def reset(n):
    kc("delete", "ns", *[km.ns(i) for i in range(n)], "--ignore-not-found", "--wait=true", check=False)
    kc("delete", "clusterqueue", "--all", "--ignore-not-found", check=False)
    kc("delete", "resourceflavor", "--all", "--ignore-not-found", check=False)


def apply(docs):
    # two passes: CRDs/objects that others reference first
    kc("apply", "-f", "-", inp=km.dump(docs))
    time.sleep(2)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def arrivals(seed, n, load, budget, cond, cpu_quota, work_s, horizon):
    """Poisson arrival times per tenant. Honest job accounts cpu_req x (work/workers) CPU-seconds."""
    h = km.SHAPES["honest"]
    job_cpu_s = h["cpu_req"] * (work_s / h["workers"])          # accounted CPU-seconds per honest job
    rate_h = load * (n * cpu_quota) / (n * job_cpu_s)           # jobs/s per honest tenant
    out = {}
    for t in range(n):
        rng = rng_for(seed, 21, t) if t else rng_for(seed, 22, 0)
        rate = rate_h * (1 + budget) if (t == 0 and cond != "baseline") else rate_h
        if t == 0 and cond == "baseline":
            rng = rng_for(seed, 21, 0)
        k = rng.poisson(rate * horizon)
        out[t] = np.sort(rng.uniform(0, horizon, k))
    return out


def submit_all(arr, cond, manip, mitigation, attacker_queues, work_s, tag):
    t0 = time.time()
    plan = []
    for t, ts in arr.items():
        for k, at in enumerate(ts):
            crafted = t == 0 and cond in ("crafted", "mitigated") and manip in ("reshape", "combined")
            split = t == 0 and cond in ("crafted", "mitigated") and manip in ("split", "combined")
            shape = "crafted" if crafted else "honest"
            q = (k % attacker_queues) if split else 0
            mit = "charge_limits" if cond == "mitigated" else "none"
            plan.append((at, km.job_manifest(f"{tag}-t{t}-{k}", t, q, shape, work_s, mit)))
    plan.sort(key=lambda p: p[0])
    submit_time = {}
    lock = threading.Lock()

    def one(item):
        at, job = item
        delay = t0 + at - time.time()
        if delay > 0:
            time.sleep(delay)
        now = time.time()
        kc("create", "-f", "-", inp=json.dumps(job))
        with lock:
            submit_time[(job["metadata"]["namespace"], job["metadata"]["name"])] = now

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, plan))
    return t0, submit_time


def collect(n, submit_time, work_s):
    jobs = json.loads(kc("get", "jobs", "-A", "-l", "fairattack=1", "-o", "json"))["items"]
    wls = json.loads(kc("get", "workloads.kueue.x-k8s.io", "-A", "-o", "json"))["items"]
    admit = {}
    for w in wls:
        for o in w["metadata"].get("ownerReferences", []):
            if o["kind"] == "Job":
                for c in w.get("status", {}).get("conditions", []):
                    if c["type"] == "Admitted" and c["status"] == "True":
                        admit[(w["metadata"]["namespace"], o["name"])] = parse_ts(c["lastTransitionTime"])
    recs = []
    for j in jobs:
        key = (j["metadata"]["namespace"], j["metadata"]["name"])
        a = j["metadata"]["annotations"]
        res = None
        if j.get("status", {}).get("succeeded"):
            log = kc("logs", "-n", key[0], f"job/{key[1]}", check=False)
            for line in log.splitlines():
                if line.startswith("RESULT "):
                    res = json.loads(line[7:])
        recs.append(dict(tenant=int(j["metadata"]["labels"]["tenant"]),
                         submit=submit_time.get(key, parse_ts(j["metadata"]["creationTimestamp"])),
                         admit=admit.get(key),
                         start=res["start"] if res else parse_ts(j.get("status", {}).get("startTime")),
                         end=res["end"] if res else None,
                         cpu_req=float(a["cpu_req"]), cpu_lim=float(a["cpu_lim"]), mem_gi=float(a["mem_gi"]),
                         work_s=float(a["work_s"]), cpu_seconds=res["cpu_seconds"] if res else None))
    return recs


def sample_cq(stop, rows):
    """Admission accounting as Kueue sees it: ClusterQueue fairSharing share and cpu usage."""
    while not stop.is_set():
        try:
            for cq in json.loads(kc("get", "clusterqueue", "-o", "json"))["items"]:
                st = cq.get("status", {})
                use = 0.0
                for f in st.get("flavorsUsage", []):
                    for r in f.get("resources", []):
                        if r["name"] == "cpu":
                            v = r["total"]
                            use += float(v[:-1]) / 1000 if v.endswith("m") else float(v)
                rows.append(dict(t=time.time(), cq=cq["metadata"]["name"], cpu=use,
                                 share=float(st.get("fairSharing", {}).get("weightedShare", 0) or 0)))
        except Exception:
            pass
        stop.wait(3.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["baseline", "naive", "crafted", "mitigated"])
    ap.add_argument("--manip", default="none", choices=["none", "reshape", "split", "combined"])
    ap.add_argument("--load", type=float, default=0.9)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--tenants", type=int, default=3)
    ap.add_argument("--attacker-queues", type=int, default=1)
    ap.add_argument("--cpu-quota", type=float, default=4.0, help="nominal CPU per tenant")
    ap.add_argument("--mem-quota-gi", type=float, default=16.0)
    ap.add_argument("--work", type=float, default=30.0, help="useful CPU-seconds per Job")
    ap.add_argument("--horizon", type=float, default=300.0, help="arrival window (s)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/k8s.csv")
    ap.add_argument("--keep", action="store_true", help="do not delete namespaces afterwards")
    a = ap.parse_args()

    n = a.tenants
    aq = a.attacker_queues if (a.manip in ("split", "combined") and a.condition in ("crafted", "mitigated")) else 1
    tag = f"s{a.seed}"
    reset(n)
    apply(km.cluster_manifests(n, a.cpu_quota, a.mem_quota_gi, aq))
    arr = arrivals(a.seed, n, a.load, a.budget, a.condition, a.cpu_quota, a.work, a.horizon)
    stop, cq_rows = threading.Event(), []
    th = threading.Thread(target=sample_cq, args=(stop, cq_rows), daemon=True)
    th.start()
    t0, submit_time = submit_all(arr, a.condition, a.manip, "", aq, a.work, tag)
    deadline = time.time() + a.timeout
    while time.time() < deadline:                       # wait for all Jobs to finish
        js = json.loads(kc("get", "jobs", "-A", "-l", "fairattack=1", "-o", "json"))["items"]
        if js and all(j.get("status", {}).get("succeeded") or j.get("status", {}).get("failed") for j in js):
            break
        time.sleep(5)
    stop.set()
    recs = collect(n, submit_time, a.work)
    w0, w1 = t0, t0 + a.horizon
    n_honest_jobs = len(arr[1]) if n > 1 else len(arr[0])
    extra = len(arr[0]) - n_honest_jobs if a.condition != "baseline" else 0
    nodes = json.loads(kc("get", "nodes", "-o", "json"))["items"]
    cap = sum(float(x["status"]["allocatable"]["cpu"]) for x in nodes
              if "node-role.kubernetes.io/control-plane" not in x["metadata"].get("labels", {}))
    m = compute_metrics(recs, n, w0, w1, cap, aq, extra, a.budget)
    # Kueue's own view of the shares during the window (admission accounting)
    for i in range(n):
        names = km.queue_names(i, aq)
        sh = [r["share"] for r in cq_rows if r["cq"] in names and w0 <= r["t"] <= w1]
        m[f"kueue_share_mean_t{i}"] = float(np.mean(sh)) if sh else float("nan")
    json.dump(dict(recs=recs, cq=cq_rows), open(f"results/k8s_raw_{a.condition}_{a.manip}_{a.seed}.json", "w"))
    meta = dict(domain="k8s", condition=a.condition, manipulation=a.manip, load=a.load, budget=a.budget,
                n_tenants=n, seed=a.seed, param=pstr(queues=aq))
    w = ResultWriter(a.out)
    w.emit(meta, m)
    w.close()
    print(json.dumps({k: round(v, 3) for k, v in m.items() if k in ("gain", "accounting_gap", "honest_wait_p95")}))
    if not a.keep:
        reset(n)
