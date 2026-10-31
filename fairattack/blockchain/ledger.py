"""Ledger state machine and append-only blockchain storage.

Provides:
  * State execution: account balances, sequence nonces, and key-value state storage.
  * Deterministic state root calculation via SHA-256 canonical hashing.
  * Blockchain ledger holding committed blocks and supporting query interfaces.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .models import Block, Transaction, QuorumCertificate, Stage
from .crypto import sha256_hex, canonical_json


@dataclass
class LedgerState:
    """State machine maintaining account balances, nonces, and storage."""
    balances: Dict[str, int] = field(default_factory=dict)
    nonces: Dict[str, int] = field(default_factory=dict)
    kv_store: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "LedgerState":
        return LedgerState(
            balances=dict(self.balances),
            nonces=dict(self.nonces),
            kv_store=copy.deepcopy(self.kv_store),
        )

    def state_root(self) -> str:
        data = {
            "balances": self.balances,
            "nonces": self.nonces,
            "kv_store": self.kv_store,
        }
        return sha256_hex(canonical_json(data))

    def apply_transaction(self, tx: Transaction) -> bool:
        """Apply a transaction to the state. Returns True if successfully applied."""
        # Check sender balance if amount > 0
        if tx.amount > 0:
            sender_bal = self.balances.get(tx.sender, 0)
            if sender_bal < tx.amount:
                return False
            self.balances[tx.sender] = sender_bal - tx.amount
            self.balances[tx.receiver] = self.balances.get(tx.receiver, 0) + tx.amount

        # Update nonce
        self.nonces[tx.sender] = max(self.nonces.get(tx.sender, 0), tx.nonce)

        # Apply payload state updates if any
        if tx.payload:
            op = tx.payload.get("op")
            if op == "set":
                k = tx.payload.get("key")
                v = tx.payload.get("val")
                if k is not None:
                    self.kv_store[str(k)] = v
            elif op == "delete":
                k = tx.payload.get("key")
                if k in self.kv_store:
                    del self.kv_store[k]

        return True


class Blockchain:
    """Append-only blockchain ledger storing committed blocks."""

    def __init__(self, initial_balances: Optional[Dict[str, int]] = None):
        self.blocks: List[Block] = []
        self.blocks_by_hash: Dict[str, Block] = {}
        self.state = LedgerState(balances=dict(initial_balances or {}))
        self._init_genesis()

    def _init_genesis(self):
        genesis_qc = QuorumCertificate(stage=Stage.DECIDE, view=0, block_hash="0" * 64, votes={})
        genesis_block = Block(
            height=0,
            view=0,
            proposer_id="genesis",
            parent_hash="0" * 64,
            justify_qc=genesis_qc,
            transactions=[],
            state_root=self.state.state_root(),
            timestamp=0.0,
            block_hash="0" * 64,
        )
        self.blocks.append(genesis_block)
        self.blocks_by_hash[genesis_block.block_hash] = genesis_block

    def get_head(self) -> Block:
        return self.blocks[-1]

    def get_block_by_height(self, height: int) -> Optional[Block]:
        if 0 <= height < len(self.blocks):
            return self.blocks[height]
        return None

    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        return self.blocks_by_hash.get(block_hash)

    def commit_block(self, block: Block) -> bool:
        """Execute and commit a finalized block to the blockchain ledger."""
        head = self.get_head()
        if block.parent_hash != head.block_hash:
            return False

        # Apply transactions in order
        new_state = self.state.copy()
        for tx in block.transactions:
            new_state.apply_transaction(tx)

        # Update block state_root and append
        block.state_root = new_state.state_root()
        self.state = new_state
        self.blocks.append(block)
        self.blocks_by_hash[block.block_hash] = block
        return True

    def get_balance(self, address: str) -> int:
        return self.state.balances.get(address, 0)
