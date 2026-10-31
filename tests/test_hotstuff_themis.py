"""Unit and integration tests for HotStuff BFT Blockchain with Themis Order-Fairness."""
from __future__ import annotations
from fairattack.blockchain.crypto import KeyPair, sha256_hex
from fairattack.blockchain.models import Transaction, Vote, QuorumCertificate, Block, Stage
from fairattack.blockchain.themis import ThemisConfig, ThemisProposal, themis_order_transactions, verify_themis_order
from fairattack.blockchain.consensus import HotStuffConfig, HotStuffConsensus
from fairattack.blockchain.ledger import Blockchain, LedgerState
from fairattack.blockchain.node import HotStuffNode
from fairattack.blockchain.network import AsyncNetwork
from fairattack.blockchain.cli import create_cluster


def test_crypto_signing_and_verification():
    kp = KeyPair.generate("node_0", seed="seed_test")
    msg = "hotstuff_themis_consensus_message"
    sig = kp.sign(msg)
    assert kp.verify(msg, sig) is True
    # Tampered message
    assert kp.verify("tampered_msg", sig) is False
    # Tampered signature
    assert kp.verify(msg, "0" * 64) is False


def test_transaction_creation_and_signing():
    client_kp = KeyPair.generate("alice", seed="alice_key")
    tx = Transaction(sender="alice", receiver="bob", amount=100, nonce=1)
    tx.sign(client_kp)
    assert tx.verify(client_kp) is True
    assert len(tx.tx_id) == 64

    # Tampered amount
    tx.amount = 200
    assert tx.verify(client_kp) is False


def test_themis_condorcet_cycle_forms_single_batch():
    """Verify that a 3-way Condorcet cycle (A->B->C->A) is resolved as a single SCC batch."""
    cfg = ThemisConfig(n=9, f=0, gamma=0.6)  # theta = ceil(5.4) = 6
    tx_a = Transaction(sender="a", receiver="x", nonce=10, tx_id="tx_a")
    tx_b = Transaction(sender="b", receiver="x", nonce=2, tx_id="tx_b")
    tx_c = Transaction(sender="c", receiver="x", nonce=20, tx_id="tx_c")
    tx_map = {"tx_a": tx_a, "tx_b": tx_b, "tx_c": tx_c}

    # 3 rotations with 3 replicas each (support 6/9 for each adjacent pair)
    proposals = []
    rots = [["tx_a", "tx_b", "tx_c"], ["tx_b", "tx_c", "tx_a"], ["tx_c", "tx_a", "tx_b"]]
    for i in range(9):
        proposals.append(ThemisProposal(replica_id=f"r_{i}", view=1, tx_order=rots[i % 3]))

    ordered, batches = themis_order_transactions(tx_map, proposals, cfg)
    # Must all belong to the same batch (Condorcet cycle)
    assert len(set(batches)) == 1
    # Intra-batch tie-break by smallest nonce: tx_b (nonce 2), tx_a (nonce 10), tx_c (nonce 20)
    assert [tx.tx_id for tx in ordered] == ["tx_b", "tx_a", "tx_c"]


def test_themis_attribute_blind_lottery_ordering():
    """Verify Lemma 2 (Lottery ties are unsteerable): P[t_a precedes t_v] = 1/2 under attribute-blind lottery."""
    # Cycle among victim tx_v (nonce 10) and attacker tx_a (crafted small nonce -5)
    tx_v = Transaction(sender="victim", receiver="x", nonce=10, tx_id="tx_v")
    tx_a = Transaction(sender="attacker", receiver="x", nonce=-5, tx_id="tx_a")
    tx_h = Transaction(sender="helper", receiver="x", nonce=5, tx_id="tx_h")
    tx_map = {"tx_v": tx_v, "tx_a": tx_a, "tx_h": tx_h}

    rots = [["tx_v", "tx_a", "tx_h"], ["tx_a", "tx_h", "tx_v"], ["tx_h", "tx_v", "tx_a"]]
    proposals = [ThemisProposal(replica_id=f"r_{i}", view=1, tx_order=rots[i % 3]) for i in range(9)]

    # 1. Without mitigation (nonce tiebreak): Attacker with nonce -5 always wins (100%)
    cfg_fixed = ThemisConfig(n=9, f=0, gamma=0.6, tiebreak="nonce")
    ord_fixed, _ = themis_order_transactions(tx_map, proposals, cfg_fixed)
    assert ord_fixed[0].tx_id == "tx_a"

    # 2. With Attribute-Blind Lottery Ordering (Section 6.2): uniform random permutation
    wins = 0
    trials = 100
    for s in range(trials):
        cfg_lottery = ThemisConfig(n=9, f=0, gamma=0.6, tiebreak="lottery", lottery_seed=s)
        ord_lottery, _ = themis_order_transactions(tx_map, proposals, cfg_lottery)
        pos = {tx.tx_id: idx for idx, tx in enumerate(ord_lottery)}
        if pos["tx_a"] < pos["tx_v"]:
            wins += 1

    # Frequency of attacker ordered before victim is approximately 1/2 (unsteerable)
    rate = wins / trials
    assert 0.35 <= rate <= 0.65



def test_themis_clear_order_respected():
    """Verify that when all replicas observe order A before B before C, order is strictly preserved."""
    cfg = ThemisConfig(n=4, f=1, gamma=0.75)  # theta = ceil(3 - 2) = 1
    tx_a = Transaction(sender="a", receiver="x", nonce=10, tx_id="tx_a")
    tx_b = Transaction(sender="b", receiver="x", nonce=5, tx_id="tx_b")
    tx_c = Transaction(sender="c", receiver="x", nonce=1, tx_id="tx_c")
    tx_map = {"tx_a": tx_a, "tx_b": tx_b, "tx_c": tx_c}

    proposals = [ThemisProposal(replica_id=f"r_{i}", view=1, tx_order=["tx_a", "tx_b", "tx_c"]) for i in range(3)]
    ordered, batches = themis_order_transactions(tx_map, proposals, cfg)
    assert [tx.tx_id for tx in ordered] == ["tx_a", "tx_b", "tx_c"]
    assert verify_themis_order(ordered, proposals, cfg) is True


def test_hotstuff_qc_validation():
    val_keys = {f"node_{i}": KeyPair.generate(f"node_{i}", seed=f"v_{i}") for i in range(4)}
    block_hash = "a" * 64
    votes = {}
    for i in range(3):  # 2f + 1 = 3
        v = Vote(stage=Stage.PREPARE, view=1, block_hash=block_hash, voter_id=f"node_{i}")
        v.sign(val_keys[f"node_{i}"])
        votes[f"node_{i}"] = v.signature

    qc = QuorumCertificate(stage=Stage.PREPARE, view=1, block_hash=block_hash, votes=votes)
    assert qc.is_valid(quorum_threshold=3, key_pairs=val_keys) is True
    # Insufficient votes
    del votes["node_0"]
    assert qc.is_valid(quorum_threshold=3, key_pairs=val_keys) is False


def test_hotstuff_multi_node_consensus_and_ledger():
    """Test full 4-node cluster committing a block with state transitions."""
    nodes, network, val_keys = create_cluster(n=4, f=1, gamma=0.75)
    alice_kp = KeyPair.generate("alice", seed="alice_key")

    tx1 = Transaction(sender="alice", receiver="bob", amount=150, nonce=1)
    tx1.sign(alice_kp)
    tx2 = Transaction(sender="bob", receiver="charlie", amount=50, nonce=1)
    tx2.sign(KeyPair.generate("bob", seed="bob_key"))

    network.broadcast_transaction(tx1)
    network.broadcast_transaction(tx2)

    committed = network.run_consensus_round()
    assert committed is not None
    assert committed.height == 1
    assert len(committed.transactions) == 2

    # Check all nodes have committed the block and updated ledger balances
    for node in nodes.values():
        assert len(node.blockchain.blocks) == 2  # genesis + block 1
        assert node.blockchain.get_balance("alice") == 10_000 - 150
        assert node.blockchain.get_balance("bob") == 5_000 + 150 - 50
        assert node.blockchain.get_balance("charlie") == 2_500 + 50
        assert node.blockchain.get_head().block_hash == committed.block_hash


def test_hotstuff_fault_tolerance_with_offline_node():
    """Verify consensus succeeds even with f=1 node offline in an n=4 cluster."""
    nodes, network, val_keys = create_cluster(n=4, f=1, gamma=0.75)
    # Node 1 is leader for view 1. Let node 3 be offline (follower).
    network.set_offline("node_3", offline=True)

    alice_kp = KeyPair.generate("alice", seed="alice_key")
    tx = Transaction(sender="alice", receiver="bob", amount=75, nonce=1)
    tx.sign(alice_kp)
    network.broadcast_transaction(tx)

    committed = network.run_consensus_round()
    assert committed is not None
    assert committed.height == 1

    # Remaining 3 nodes reached consensus
    for nid in ("node_0", "node_1", "node_2"):
        assert len(nodes[nid].blockchain.blocks) == 2
