# Exploiting Scheduling Fairness: Adversarial Attacks on Distributed Workloads



## Overview

This repository contains the implementation, test suites, and evaluation artifacts for the paper across three distributed systems:

| System | Domain | Mechanism | Attack Vector |
| :--- | :--- | :--- | :--- |
| **Kueue** | Cloud Batch Computing | Kubernetes cohort fair-sharing | Request-to-limit ratio reshaping & queue splitting |
| **Themis** | Blockchain Consensus | HotStuff BFT with order-fair batching | Condorcet cycle insertion & nonce tie-break manipulation |
| **VTC** | Multi-Agent LLM Serving | Continuous batching with Google Gemma 3 12B | Prefill/decode token pricing mismatch & API key splitting |

### Attack Strategies
* **Workload Reshaping**: Modifying the structural properties of tasks (e.g., CPU request vs. limit, prompt length vs. decode length, or transaction arrival timestamps) to minimize declared cost while maximizing consumed runtime service.
* **Workload Splitting**: Distributing equivalent workload across multiple identities, queues, or API keys to exploit per-entity fair-share allocation.

### Evaluated Conditions
* **Baseline**: Honest workload across all tenants.
* **Naive**: Attacker submits extra work volume using an honest workload profile.
* **Crafted**: Attacker offers the exact same policy-charged units as naive, but applies reshaping and/or splitting.
* **Mitigated**: Targeted defenses implemented for each domain:
  * **Kubernetes (Settled Accounting)**: Adjusts each Job's charge based on measured runtime CPU usage rather than initial declaration, eliminating the accounting gap (Lemma 1).
  * **Blockchain (Attribute-Blind Lottery Ordering)**: Uniformly randomizes transaction order within each cycle using an unpredictable seed, ensuring $\Pr[t_a \prec t_v] = 1/2$ (Lemma 2).
  * **LLM Serving (Cost-Priced Tokens & Owner-Collapsed Metering)**: Replaces linear token pricing with calibrated GPU execution costs ($\kappa_a \to \kappa_h$) and collapses all API keys of an owner to one meter ($m \to 1$), restoring fair $1/n$ allocation (Lemma 3).
  * **Mitigation Metric**: Relative reduction $\Delta_M = 1 - M_{\text{mitigated}} / M_{\text{crafted}}$ (100% indicates full mitigation).

---

## Repository Structure

```
fairattack/
  blockchain/          HotStuff BFT consensus + Themis order-fairness engine
    crypto.py          Cryptographic keypairs, HMAC-SHA256 signatures, and digests
    models.py          Transaction, Vote, QuorumCertificate (QC), and Block definitions
    themis.py          Themis precedence graph, Tarjan SCC batching & condensation DAG
    consensus.py       3-phase HotStuff BFT consensus (Prepare, Pre-Commit, Commit, Decide)
    ledger.py          Append-only blockchain ledger, account balances, and state transitions
    node.py            Validator node with mempool, arrival tracking, and proposal engine
    network.py         Multi-node async transport coordinator with fault-injection
    cli.py             CLI demo to launch validator clusters and submit transactions
  llm.py               Google Gemma 3 12B model specs, cost model, VTC policy & traces
  k8s_manifests.py     Kueue ClusterQueue, LocalQueue, ResourceFlavor, and Job manifests
  k8s_metrics.py       Runtime consumption vs. admission accounting metrics
  common.py            Shared metrics (Jain's index, advantage gain, confidence intervals)

experiments/
  run_k8s.py           Runs live Kueue experiments on a Kubernetes cluster

k8s/
  kind-config.yaml     Kind 3-worker cluster configuration
  setup.sh             Automated installer for Kind and Kueue with fair-sharing
  burn.py              CPU/memory burner payload for Job execution

llm_gpu/
  cost_model.json      Calibrated GPU cost model for Google Gemma 3 12B
  calibrate_cost_model.py  Profiles TTFT and decode throughput on vLLM
  vllm_vtc_proxy.py    VTC fair dispatch proxy in front of vLLM
  loadgen.py           Replays traces through VTC proxy and logs GPU/KV metrics
  make_trace.py        Generates Gemma 3 12B workload traces for attack evaluation

scripts/
  run_k8s_suite.sh     Automated sweep of loads and attack conditions on Kubernetes
  run_gpu_validation.sh End-to-end vLLM GPU validation script for Gemma 3 12B

tests/
  run_tests.py         Lightweight test runner (no external dependencies)
  test_hotstuff_themis.py  Tests for HotStuff BFT, Themis ordering, and Byzantine fault tolerance
  test_k8s.py          Tests for Kueue manifest generation and metric calculations
  test_llm_serving.py  Tests for Gemma 3 12B model configuration, cost model, and VTC proxy
```

---

## Quick Start

### 1. Installation & Sanity Tests
```bash
pip install -r requirements.txt
python tests/run_tests.py
```
Runs all 14 unit and integration tests across HotStuff BFT, Themis order-fairness, Kubernetes Kueue, and Gemma 3 12B LLM serving.

---

### 2. Blockchain: HotStuff BFT with Themis Order-Fairness
Run an interactive 4-node HotStuff validator cluster demonstrating Themis fair ordering:
```bash
python -m fairattack.blockchain.cli --nodes 4 --rounds 3 --txs 5
```
* **Features**: Replicas exchange signed local arrival proposals; leader computes precedence threshold $\theta = \lceil \gamma \cdot n - 2f \rceil$, resolves Condorcet cycles via Tarjan's SCC algorithm, orders batches on the condensation DAG, and commits blocks through 3-phase HotStuff BFT ($2f+1$ Quorum Certificates).

---

### 3. LLM Serving: Google Gemma 3 12B under VTC
Deploy and evaluate Google Gemma 3 12B with continuous batching under the Virtual Token Counter (VTC) scheduler:
```bash
# 1. Start vLLM serving Google Gemma 3 12B (GPU server)
vllm serve google/gemma-3-12b-it --port 8000 --enable-prefix-caching=false

# 2. Launch VTC dispatch proxy
python llm_gpu/vllm_vtc_proxy.py --upstream http://localhost:8000 --port 9000 --max-inflight 64

# 3. Generate crafted workload trace and replay
python llm_gpu/make_trace.py --condition crafted --manip combined --keys 8 --out trace.csv
python llm_gpu/loadgen.py --trace trace.csv --model google/gemma-3-12b-it
```

---

### 4. Kubernetes: Kueue Cohort Fair-Sharing
Deploy a local Kind cluster and run the workload manipulation suite:
```bash
# 1. Create 3-worker cluster and configure Kueue fair-sharing
bash k8s/setup.sh

# 2. Run experiment suite across loads and conditions
SEEDS=3 LOADS="0.9" bash scripts/run_k8s_suite.sh
```
* **Features**: Tests admission accounting vs. runtime consumption gaps, request-to-limit ratio reshaping ($0.5$ CPU request, $4.0$ CPU limit), and queue splitting over multiple ClusterQueues.
