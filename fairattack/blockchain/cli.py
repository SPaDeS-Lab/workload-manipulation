"""CLI runner and demonstration for HotStuff BFT Blockchain with Themis Order-Fairness.

Usage:
  python -m fairattack.blockchain.cli --nodes 4 --rounds 3 --txs 6
"""
from __future__ import annotations
import argparse, json, sys, time
from typing import Dict, List
from .crypto import KeyPair
from .models import Transaction, Block
from .themis import ThemisConfig
from .consensus import HotStuffConfig
from .node import HotStuffNode
from .network import AsyncNetwork


def create_cluster(n: int = 4, f: int = 1, gamma: float = 0.75) -> Tuple[Dict[str, HotStuffNode], AsyncNetwork, Dict[str, KeyPair]]:
    """Instantiate an n-node HotStuff validator cluster with Themis fair ordering."""
    validator_keys = {f"node_{i}": KeyPair.generate(f"node_{i}", seed=f"val_{i}") for i in range(n)}
    hs_cfg = HotStuffConfig(n=n, f=f)
    themis_cfg = ThemisConfig(n=n, f=f, gamma=gamma)

    initial_balances = {
        "alice": 10_000,
        "bob": 5_000,
        "charlie": 2_500,
        "attacker": 1_000,
    }

    nodes = {}
    for nid, kp in validator_keys.items():
        node = HotStuffNode(
            node_id=nid,
            key_pair=kp,
            validator_keys=validator_keys,
            cfg=hs_cfg,
            themis_cfg=themis_cfg,
            initial_balances=initial_balances,
        )
        nodes[nid] = node

    network = AsyncNetwork(nodes)
    return nodes, network, validator_keys


def run_demo(nodes_count: int = 4, rounds_count: int = 3, txs_per_round: int = 5, gamma: float = 0.75):
    print("=" * 70)
    print("  HotStuff BFT Blockchain with Themis Order-Fairness")
    print(f"  Configuration: {nodes_count} Replicas (f={(nodes_count - 1)//3}), gamma={gamma}")
    print("=" * 70)

    f_faults = max(0, (nodes_count - 1) // 3)
    nodes, network, val_keys = create_cluster(n=nodes_count, f=f_faults, gamma=gamma)

    client_alice = KeyPair.generate("client_alice", seed="alice_secret")
    client_bob = KeyPair.generate("client_bob", seed="bob_secret")
    client_attacker = KeyPair.generate("client_attacker", seed="attacker_secret")

    for r in range(1, rounds_count + 1):
        print(f"\n--- Round {r} ---")
        # Submit transactions with varied arrival patterns
        tx_list = []
        for i in range(txs_per_round):
            if i % 3 == 0:
                tx = Transaction(
                    sender="alice",
                    receiver="bob",
                    amount=50 + i * 10,
                    payload={"op": "set", "key": f"tx_{r}_{i}", "val": f"transfer_{i}"},
                    nonce=r * 100 + i,
                )
                tx.sign(client_alice)
            elif i % 3 == 1:
                tx = Transaction(
                    sender="bob",
                    receiver="charlie",
                    amount=20 + i * 5,
                    payload={"op": "set", "key": f"tx_{r}_{i}", "val": f"payment_{i}"},
                    nonce=r * 100 + i,
                )
                tx.sign(client_bob)
            else:
                tx = Transaction(
                    sender="attacker",
                    receiver="charlie",
                    amount=10,
                    payload={"op": "set", "key": f"tx_{r}_{i}", "val": "crafted_tx"},
                    nonce=r * 100 + i,
                )
                tx.sign(client_attacker)
            tx_list.append(tx)

        # Broadcast transactions to validator nodes
        for tx in tx_list:
            network.broadcast_transaction(tx)

        print(f"  Broadcast {len(tx_list)} client transactions to validator mempools.")

        # Run 4-stage HotStuff consensus with Themis fair ordering
        committed_block = network.run_consensus_round()
        if committed_block:
            print(f"  [SUCCESS] Block #{committed_block.height} committed via HotStuff consensus!")
            print(f"    View: {committed_block.view} | Proposer: {committed_block.proposer_id}")
            print(f"    Block Hash: {committed_block.block_hash[:16]}... | Parent Hash: {committed_block.parent_hash[:16]}...")
            print(f"    Themis Fair-Ordered Transactions ({len(committed_block.transactions)} txs):")
            for idx, tx in enumerate(committed_block.transactions):
                print(f"      [{idx + 1}] {tx.sender} -> {tx.receiver} : {tx.amount} tokens (nonce {tx.nonce}, hash {tx.tx_id[:10]}...)")
            print(f"    State Root: {committed_block.state_root[:16]}...")
        else:
            print("  [ERROR] Consensus failed to commit a block.")

    # Final ledger state
    sample_node = next(iter(nodes.values()))
    print("\n" + "=" * 70)
    print("  Final Blockchain State:")
    print(f"    Total Blocks Committed: {len(sample_node.blockchain.blocks) - 1}")
    print("    Account Balances:")
    for acc in ("alice", "bob", "charlie", "attacker"):
        print(f"      {acc}: {sample_node.blockchain.get_balance(acc)} tokens")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="HotStuff BFT Blockchain with Themis Order-Fairness CLI")
    parser.add_argument("--nodes", type=int, default=4, help="Number of validator nodes (default: 4)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of consensus rounds to run (default: 3)")
    parser.add_argument("--txs", type=int, default=5, help="Transactions to submit per round (default: 5)")
    parser.add_argument("--gamma", type=float, default=0.75, help="Themis fairness parameter gamma (default: 0.75)")
    args = parser.parse_args()

    run_demo(nodes_count=args.nodes, rounds_count=args.rounds, txs_per_round=args.txs, gamma=args.gamma)


if __name__ == "__main__":
    main()
