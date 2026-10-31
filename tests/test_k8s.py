import yaml
from fairattack.k8s_manifests import cluster_manifests, job_manifest, dump
from fairattack.k8s_metrics import compute_metrics
from fairattack.common import relative_reduction


def test_manifests_roundtrip_and_split_queues():
    docs = cluster_manifests(3, 4, 16, attacker_queues=4)
    kinds = [d["kind"] for d in docs]
    assert kinds.count("ClusterQueue") == 2 + 4
    cqs = [d for d in docs if d["kind"] == "ClusterQueue" and d["metadata"]["name"].startswith("cq-t0")]
    assert all(c["spec"]["resourceGroups"][0]["flavors"][0]["resources"][0]["nominalQuota"] == "1" for c in cqs)
    assert len(list(yaml.safe_load_all(dump(docs)))) == len(docs)


def test_mitigation_charges_limits_and_settled():
    j1 = job_manifest("x1", 0, 0, "crafted", 30, mitigation="charge_limits")
    r1 = j1["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert r1["requests"]["cpu"] == r1["limits"]["cpu"] == "4"

    j2 = job_manifest("x2", 0, 0, "crafted", 30, mitigation="settled_accounting")
    r2 = j2["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert r2["requests"]["cpu"] == r2["limits"]["cpu"] == "4"


def test_metrics_synthetic():
    recs = []
    for t, share in ((0, 3), (1, 1), (2, 1)):
        for k in range(share):
            recs.append(dict(tenant=t, submit=0, admit=1, start=1, end=11, cpu_req=2 if t else 0.5,
                             cpu_lim=2, mem_gi=1, work_s=20, cpu_seconds=20))
    m = compute_metrics(recs, 3, 0, 20, 12, 1, 0, 0.2)
    assert m["u_attacker"] > 0.5 and m["u_attacker_accounted"] < m["u_attacker"]


def test_settled_accounting_mitigation():
    """Verify Lemma 1 (Settled meters are faithful): Settled accounting eliminates the accounting gap."""
    recs = []
    # Attacker requests 0.5 CPU but consumes 4.0 CPU; Honest tenants request 2.0 and use 2.0
    for t, count, req, actual_cpu in ((0, 4, 0.5, 4.0), (1, 2, 2.0, 2.0), (2, 2, 2.0, 2.0)):
        for k in range(count):
            recs.append(dict(tenant=t, submit=0, admit=1, start=1, end=11, cpu_req=req,
                             cpu_lim=4.0, mem_gi=1, work_s=20, cpu_seconds=actual_cpu * 10))

    # Declared accounting
    m_decl = compute_metrics(recs, 3, 0, 20, 12, 1, 0, 0.2, accounting_mode="declared")
    # Settled accounting (Section 6.1)
    m_settled = compute_metrics(recs, 3, 0, 20, 12, 1, 0, 0.2, accounting_mode="settled")

    # Under declared accounting, attacker has a large accounting gap
    assert m_decl["accounting_gap"] > 0.15
    # Under settled accounting, accounting gap drops to 0 (faithful meter)
    assert abs(m_settled["accounting_gap"]) < 1e-6
    # Relative reduction Delta_M = 1 - M_mitigated / M_crafted approaches 100%
    red = relative_reduction(abs(m_settled["accounting_gap"]), abs(m_decl["accounting_gap"]))
    assert red > 0.99
