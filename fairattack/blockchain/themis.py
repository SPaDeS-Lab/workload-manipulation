"""Themis Order-Fairness Protocol Engine (Kelkar et al., 2023).

Implements:
  * Local order proposal generation and validation.
  * Fair precedence graph construction with threshold theta = ceil(gamma * n - 2f).
  * Tarjan's Strongly Connected Components (SCC) algorithm for Condorcet cycle batching.
  * Condensation DAG topological ordering with deterministic tie-breaking.
  * Independent verification of Themis fair ordering proofs.
"""
from __future__ import annotations
import heapq, math, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Optional
from .models import Transaction
from .crypto import sha256_hex, canonical_json, KeyPair


@dataclass
class ThemisConfig:
    """Themis order-fairness parameters."""
    n: int = 4
    f: int = 1
    gamma: float = 0.75
    tiebreak: str = "nonce"  # "nonce" | "hash" | "lottery" (Attribute-Blind Lottery Ordering)
    lottery_seed: Optional[int | str] = None

    @property
    def theta(self) -> int:
        """Fair precedence threshold: ceil(gamma * n - 2f)."""
        val = math.ceil(self.gamma * self.n - 2 * self.f)
        return max(1, int(val))

    @staticmethod
    def for_n(n: int, gamma: float = 0.75, **kwargs) -> "ThemisConfig":
        """Compute maximum f satisfying n > 4f / (2*gamma - 1)."""
        denom = 2 * gamma - 1
        if denom <= 0:
            f = 0
        else:
            f = max(0, math.ceil(n * denom / 4) - 1)
        return ThemisConfig(n=n, f=f, gamma=gamma, **kwargs)


@dataclass
class ThemisProposal:
    """A replica's signed local proposal of transaction arrival order."""
    replica_id: str
    view: int
    tx_order: List[str]  # ordered tx_ids
    signature: str = ""

    def digest(self) -> str:
        return f"{self.replica_id}:{self.view}:{canonical_json(self.tx_order)}"

    def sign(self, key_pair: KeyPair):
        self.signature = key_pair.sign(self.digest())

    def verify(self, key_pair: KeyPair) -> bool:
        return key_pair.verify(self.digest(), self.signature)


def tarjan_scc(adj: Dict[str, Set[str]], vertices: List[str]) -> List[List[str]]:
    """Compute Strongly Connected Components (SCCs) using Tarjan's algorithm."""
    index = 0
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, set()):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            scc: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in vertices:
        if v not in indices:
            strongconnect(v)

    return sccs


def themis_order_transactions(
    tx_map: Dict[str, Transaction],
    proposals: List[ThemisProposal],
    cfg: ThemisConfig,
) -> Tuple[List[Transaction], List[int]]:
    """Order transactions using Themis fair ordering rules.

    Returns:
      (ordered_transactions, batch_ids)
    """
    if not proposals:
        # If no proposals provided, order by transaction nonce/hash
        ordered = sorted(tx_map.values(), key=lambda t: (t.nonce, t.tx_id))
        return ordered, [0] * len(ordered)

    # 1. Identify all transactions appearing in the proposals
    tx_ids: Set[str] = set()
    for p in proposals:
        tx_ids.update(p.tx_order)
    # Also include any transactions in tx_map that may not be in proposals
    tx_ids.update(tx_map.keys())

    all_txs = sorted(list(tx_ids))
    m = len(all_txs)
    if m <= 1:
        return [tx_map[t] for t in all_txs if t in tx_map], [0] * len(all_txs)

    tx_idx = {tx_id: i for i, tx_id in enumerate(all_txs)}

    # 2. Build pairwise order counts C[u, v]: # proposals where u precedes v
    # For each proposal, record position of each tx
    order_counts: Dict[Tuple[str, str], int] = {}
    for p in proposals:
        pos = {tx_id: idx for idx, tx_id in enumerate(p.tx_order)}
        for u in p.tx_order:
            for v in p.tx_order:
                if u != v and pos[u] < pos[v]:
                    key = (u, v)
                    order_counts[key] = order_counts.get(key, 0) + 1

    # 3. Directed Precedence Graph E: u -> v iff C[u, v] >= theta
    theta = cfg.theta
    adj: Dict[str, Set[str]] = {tx_id: set() for tx_id in all_txs}
    for (u, v), count in order_counts.items():
        if count >= theta and u in adj and v in adj:
            adj[u].add(v)

    # 4. Compute Strongly Connected Components (batches)
    sccs = tarjan_scc(adj, all_txs)
    num_components = len(sccs)

    # Map each tx_id to its component index
    comp_map: Dict[str, int] = {}
    for c_idx, scc in enumerate(sccs):
        for tx_id in scc:
            comp_map[tx_id] = c_idx

    # 5. Build Condensation DAG among components
    cond_adj: Dict[int, Set[int]] = {c: set() for c in range(num_components)}
    indegree: Dict[int, int] = {c: 0 for c in range(num_components)}

    for u in all_txs:
        c_u = comp_map[u]
        for v in adj[u]:
            c_v = comp_map[v]
            if c_u != c_v and c_v not in cond_adj[c_u]:
                cond_adj[c_u].add(c_v)
                indegree[c_v] += 1

    # 6. Tie-breaking keys & Attribute-Blind Lottery Ordering
    is_lottery = cfg.tiebreak in ("lottery", "random")

    def tx_key(tx_id: str) -> Tuple[int, str]:
        tx = tx_map.get(tx_id)
        if not tx:
            return (0, tx_id)
        if cfg.tiebreak == "nonce":
            return (tx.nonce, tx.tx_id)
        return (0, tx.tx_id)

    def component_key(c_idx: int) -> Tuple[Any, int]:
        if is_lottery:
            # Attribute-blind pseudorandom rank derived from unpredictable lottery seed
            seed_str = f"lottery_comp:{cfg.lottery_seed if cfg.lottery_seed is not None else '0'}:{c_idx}"
            return (sha256_hex(seed_str), c_idx)
        members = sccs[c_idx]
        return (min(tx_key(t) for t in members), c_idx)

    # 7. Topological sort over Condensation DAG using min-heap for deterministic tie-breaking
    heap: List[Tuple[Any, int]] = []
    for c in range(num_components):
        if indegree[c] == 0:
            heapq.heappush(heap, (component_key(c), c))

    ordered_txs: List[Transaction] = []
    batch_ids: List[int] = []
    curr_batch = 0

    while heap:
        _, c = heapq.heappop(heap)
        if is_lottery and len(sccs[c]) > 1:
            # Attribute-Blind Lottery Ordering (Lemma 2):
            # Uniform random permutation independent of transaction attributes / attacker timing
            seed_str = f"lottery_cycle:{cfg.lottery_seed if cfg.lottery_seed is not None else '0'}:{c}"
            cycle_prng = random.Random(seed_str)
            members = list(sccs[c])
            cycle_prng.shuffle(members)
        else:
            members = sorted(sccs[c], key=tx_key)
        for tx_id in members:
            if tx_id in tx_map:
                ordered_txs.append(tx_map[tx_id])
                batch_ids.append(curr_batch)
        curr_batch += 1

        for neighbor in cond_adj[c]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(heap, (component_key(neighbor), neighbor))

    return ordered_txs, batch_ids


def verify_themis_order(
    block_txs: List[Transaction],
    proposals: List[ThemisProposal],
    cfg: ThemisConfig,
) -> bool:
    """Verify that block transactions strictly adhere to Themis fair ordering rules."""
    tx_map = {tx.tx_id: tx for tx in block_txs}
    expected_txs, _ = themis_order_transactions(tx_map, proposals, cfg)
    expected_ids = [tx.tx_id for tx in expected_txs]
    actual_ids = [tx.tx_id for tx in block_txs]
    return expected_ids == actual_ids
