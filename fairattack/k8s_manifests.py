"""Generate Kueue/Kubernetes manifests for the fair-sharing experiments.

Cohort of `n` tenants. Tenant 0 is the attacker slot. Equal nominal quotas, equal weights.
Kueue's cohort fair sharing ranks ClusterQueues by their share of *borrowed quota computed from
admitted workloads' resource requests* (not runtime CPU use). API version below is v1beta1;
if your Kueue release serves v1beta2, change KUEUE_API.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional
import yaml

KUEUE_API = "kueue.x-k8s.io/v1beta1"
COHORT = "fair-cohort"
HERE = os.path.dirname(os.path.abspath(__file__))
BURN_PATH = os.path.join(HERE, "..", "k8s", "burn.py")

# job shapes: (cpu_request, cpu_limit, mem_request_gi, workers)
SHAPES = {
    "honest": dict(cpu_req=2.0, cpu_lim=2.0, mem_gi=1.0, workers=2),
    # reshape: tiny CPU request (cheap in Kueue's accounting) but high limit (bursts at run time)
    "crafted": dict(cpu_req=0.5, cpu_lim=4.0, mem_gi=1.0, workers=4),
}


def ns(i: int) -> str:
    return f"tenant-{i}"


def queue_names(i: int, attacker_queues: int) -> List[str]:
    """ClusterQueue names owned by tenant i."""
    if i == 0 and attacker_queues > 1:
        return [f"cq-t0-q{j}" for j in range(attacker_queues)]
    return [f"cq-t{i}"]


def cluster_manifests(n: int, cpu_quota: float, mem_quota_gi: float, attacker_queues: int = 1,
                      weight: int = 1, mitigation: str = "none") -> List[dict]:
    docs: List[dict] = [dict(apiVersion="kueue.x-k8s.io/v1beta1" if False else KUEUE_API, kind="ResourceFlavor",
                             metadata=dict(name="default-flavor"))]
    for i in range(n):
        docs.append(dict(apiVersion="v1", kind="Namespace", metadata=dict(name=ns(i), labels=dict(tenant=str(i)))))
        with open(BURN_PATH) as f:
            docs.append(dict(apiVersion="v1", kind="ConfigMap",
                             metadata=dict(name="burn", namespace=ns(i)), data={"burn.py": f.read()}))
        qs = queue_names(i, attacker_queues)
        for j, cq in enumerate(qs):
            # an attacker that splits across k ClusterQueues divides its tenant quota evenly
            c, m = cpu_quota / len(qs), mem_quota_gi / len(qs)
            docs.append(dict(
                apiVersion=KUEUE_API, kind="ClusterQueue", metadata=dict(name=cq),
                spec=dict(cohort=COHORT, namespaceSelector=dict(matchLabels=dict(tenant=str(i))),
                          fairSharing=dict(weight=weight),
                          resourceGroups=[dict(coveredResources=["cpu", "memory"], flavors=[dict(
                              name="default-flavor",
                              resources=[dict(name="cpu", nominalQuota=_q(c)),
                                         dict(name="memory", nominalQuota=f"{m:g}Gi")])])])))
            docs.append(dict(apiVersion=KUEUE_API, kind="LocalQueue",
                             metadata=dict(name=f"lq-{j}", namespace=ns(i)), spec=dict(clusterQueue=cq)))
    return docs


def _q(x: float) -> str:
    return f"{int(x*1000)}m" if x != int(x) else str(int(x))


def job_manifest(name: str, tenant: int, queue_idx: int, shape: str, work_s: float,
                 mitigation: str = "none", image: str = "python:3.12-slim", seed_label: str = "0") -> dict:
    s = dict(SHAPES[shape])
    if mitigation in ("charge_limits", "settled_accounting"):
        # emulates settled accounting / mutating policy that charges actual limit: requests := limits
        s["cpu_req"] = s["cpu_lim"]
    res = dict(requests=dict(cpu=_q(s["cpu_req"]), memory=f"{s['mem_gi']:g}Gi"),
               limits=dict(cpu=_q(s["cpu_lim"]), memory=f"{s['mem_gi']:g}Gi"))
    return dict(
        apiVersion="batch/v1", kind="Job",
        metadata=dict(name=name, namespace=ns(tenant),
                      labels={"kueue.x-k8s.io/queue-name": f"lq-{queue_idx}", "fairattack": "1",
                              "tenant": str(tenant), "shape": shape},
                      annotations=dict(cpu_req=str(s["cpu_req"]), cpu_lim=str(s["cpu_lim"]),
                                       mem_gi=str(s["mem_gi"]), work_s=str(work_s))),
        spec=dict(suspend=True, backoffLimit=0, ttlSecondsAfterFinished=3600,
                  template=dict(spec=dict(restartPolicy="Never", containers=[dict(
                      name="burn", image=image, command=["python", "/opt/burn.py"],
                      args=["--work", str(work_s), "--workers", str(s["workers"]), "--mem-mb", "256"],
                      env=[dict(name="NODE_NAME", valueFrom=dict(fieldRef=dict(fieldPath="spec.nodeName")))],
                      volumeMounts=[dict(name="burn", mountPath="/opt")], resources=res)],
                      volumes=[dict(name="burn", configMap=dict(name="burn"))]))))


def dump(docs: List[dict]) -> str:
    return yaml.safe_dump_all(docs, sort_keys=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, default=3)
    ap.add_argument("--cpu-quota", type=float, default=4)
    ap.add_argument("--mem-quota-gi", type=float, default=16)
    ap.add_argument("--attacker-queues", type=int, default=1)
    a = ap.parse_args()
    print(dump(cluster_manifests(a.tenants, a.cpu_quota, a.mem_quota_gi, a.attacker_queues)))
