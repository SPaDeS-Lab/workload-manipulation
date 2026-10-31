"""Data models for HotStuff BFT consensus and Themis order-fair blockchain.

Defines:
  * Transaction: client state-transition proposals with digital signatures.
  * Vote: consensus votes across the 3 HotStuff phases (Prepare, Pre-Commit, Commit).
  * QuorumCertificate (QC): aggregated vote threshold proofs from 2f+1 replicas.
  * Block: ledger unit carrying Themis fair-ordered transactions and justifying QCs.
"""
from __future__ import annotations
import enum, time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from .crypto import sha256_hex, canonical_json, KeyPair


class Stage(str, enum.Enum):
    PREPARE = "PREPARE"
    PRECOMMIT = "PRECOMMIT"
    COMMIT = "COMMIT"
    DECIDE = "DECIDE"


@dataclass
class Transaction:
    """Client transaction submitted to the blockchain."""
    sender: str
    receiver: str
    amount: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    nonce: int = 0
    timestamp: float = field(default_factory=time.time)
    signature: str = ""
    tx_id: str = ""

    def __post_init__(self):
        if not self.tx_id:
            self.tx_id = self.compute_hash()

    def digest(self) -> str:
        """Data payload to sign."""
        data = {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "payload": self.payload,
            "nonce": self.nonce,
            "timestamp": round(self.timestamp, 4),
        }
        return canonical_json(data)

    def compute_hash(self) -> str:
        return sha256_hex(self.digest())

    def sign(self, key_pair: KeyPair):
        """Sign this transaction with sender's key pair."""
        self.signature = key_pair.sign(self.digest())

    def verify(self, key_pair: Optional[KeyPair] = None) -> bool:
        """Verify transaction signature."""
        if not self.signature:
            return False
        if key_pair:
            return key_pair.verify(self.digest(), self.signature)
        return len(self.signature) == 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Transaction":
        return Transaction(**d)


@dataclass
class Vote:
    """Validator vote on a block for a given consensus stage and view."""
    stage: Stage
    view: int
    block_hash: str
    voter_id: str
    signature: str = ""

    def digest(self) -> str:
        return f"{self.stage.value}:{self.view}:{self.block_hash}:{self.voter_id}"

    def sign(self, key_pair: KeyPair):
        self.signature = key_pair.sign(self.digest())

    def verify(self, key_pair: KeyPair) -> bool:
        return key_pair.verify(self.digest(), self.signature)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "view": self.view,
            "block_hash": self.block_hash,
            "voter_id": self.voter_id,
            "signature": self.signature,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Vote":
        return Vote(
            stage=Stage(d["stage"]),
            view=d["view"],
            block_hash=d["block_hash"],
            voter_id=d["voter_id"],
            signature=d.get("signature", ""),
        )


@dataclass
class QuorumCertificate:
    """Quorum Certificate (QC) proving 2f+1 replicas agreed on a stage/view/block."""
    stage: Stage
    view: int
    block_hash: str
    votes: Dict[str, str] = field(default_factory=dict)  # voter_id -> signature

    def is_valid(self, quorum_threshold: int, key_pairs: Optional[Dict[str, KeyPair]] = None) -> bool:
        """Check if QC contains enough valid signatures from distinct validators."""
        if len(self.votes) < quorum_threshold:
            return False
        if key_pairs:
            for voter_id, sig in self.votes.items():
                if voter_id not in key_pairs:
                    return False
                v = Vote(stage=self.stage, view=self.view, block_hash=self.block_hash, voter_id=voter_id, signature=sig)
                if not v.verify(key_pairs[voter_id]):
                    return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "view": self.view,
            "block_hash": self.block_hash,
            "votes": dict(self.votes),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QuorumCertificate":
        return QuorumCertificate(
            stage=Stage(d["stage"]),
            view=d["view"],
            block_hash=d["block_hash"],
            votes=dict(d.get("votes", {})),
        )


@dataclass
class Block:
    """A proposed or committed block in the HotStuff blockchain."""
    height: int
    view: int
    proposer_id: str
    parent_hash: str
    justify_qc: Optional[QuorumCertificate]
    transactions: List[Transaction] = field(default_factory=list)
    themis_proposals: Dict[str, List[str]] = field(default_factory=dict)      # replica_id -> [tx_id, ...]
    themis_signatures: Dict[str, str] = field(default_factory=dict)           # replica_id -> signature
    state_root: str = ""
    timestamp: float = field(default_factory=time.time)
    block_hash: str = ""

    def __post_init__(self):
        if not self.block_hash:
            self.block_hash = self.compute_hash()

    def header_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "view": self.view,
            "proposer_id": self.proposer_id,
            "parent_hash": self.parent_hash,
            "justify_qc": self.justify_qc.to_dict() if self.justify_qc else None,
            "tx_ids": [tx.tx_id for tx in self.transactions],
            "state_root": self.state_root,
            "timestamp": round(self.timestamp, 4),
        }

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json(self.header_dict()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "view": self.view,
            "proposer_id": self.proposer_id,
            "parent_hash": self.parent_hash,
            "justify_qc": self.justify_qc.to_dict() if self.justify_qc else None,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "themis_proposals": {k: list(v) for k, v in self.themis_proposals.items()},
            "themis_signatures": dict(self.themis_signatures),
            "state_root": self.state_root,
            "timestamp": self.timestamp,
            "block_hash": self.block_hash,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Block":
        jqc = QuorumCertificate.from_dict(d["justify_qc"]) if d.get("justify_qc") else None
        txs = [Transaction.from_dict(tx) for tx in d.get("transactions", [])]
        return Block(
            height=d["height"],
            view=d["view"],
            proposer_id=d["proposer_id"],
            parent_hash=d["parent_hash"],
            justify_qc=jqc,
            transactions=txs,
            themis_proposals=d.get("themis_proposals", {}),
            themis_signatures=d.get("themis_signatures", {}),
            state_root=d.get("state_root", ""),
            timestamp=d.get("timestamp", 0.0),
            block_hash=d.get("block_hash", ""),
        )
