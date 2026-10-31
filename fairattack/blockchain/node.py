"""Blockchain replica node combining Mempool, Themis fair ordering, HotStuff BFT, and Ledger.

Defines:
  * HotStuffNode: A full validator node capable of processing transactions,
    forming Themis order proposals, driving consensus, and committing blocks.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Set, Tuple
from .crypto import KeyPair
from .models import Block, Transaction, Vote, QuorumCertificate, Stage
from .themis import ThemisConfig, ThemisProposal, themis_order_transactions, verify_themis_order
from .consensus import HotStuffConfig, HotStuffConsensus
from .ledger import Blockchain


class HotStuffNode:
    """Full HotStuff + Themis validator node."""

    def __init__(
        self,
        node_id: str,
        key_pair: KeyPair,
        validator_keys: Dict[str, KeyPair],
        cfg: Optional[HotStuffConfig] = None,
        themis_cfg: Optional[ThemisConfig] = None,
        initial_balances: Optional[Dict[str, int]] = None,
    ):
        self.node_id = node_id
        self.key_pair = key_pair
        self.validator_keys = validator_keys

        n = len(validator_keys)
        f = (n - 1) // 3
        self.cfg = cfg or HotStuffConfig(n=n, f=max(0, f))
        self.themis_cfg = themis_cfg or ThemisConfig(n=n, f=max(0, f))

        self.consensus = HotStuffConsensus(
            node_id=node_id,
            key_pair=key_pair,
            validator_keys=validator_keys,
            cfg=self.cfg,
            themis_cfg=self.themis_cfg,
        )
        self.blockchain = Blockchain(initial_balances=initial_balances)

        # Mempool: tx_id -> (Transaction, arrival_time)
        self.mempool: Dict[str, Tuple[Transaction, float]] = {}

        # Themis proposals collected for the current view: replica_id -> ThemisProposal
        self.collected_themis_proposals: Dict[str, ThemisProposal] = {}

    # -------------------------------------------------------------------------
    # Mempool & Transaction Ingestion
    # -------------------------------------------------------------------------
    def submit_transaction(self, tx: Transaction) -> bool:
        """Receive transaction from client, verify signature, and record local arrival."""
        if not tx.verify():
            return False
        if tx.tx_id not in self.mempool:
            self.mempool[tx.tx_id] = (tx, time.time())
        return True

    def get_local_order(self) -> List[str]:
        """Local arrival order of pending mempool transactions."""
        sorted_items = sorted(self.mempool.items(), key=lambda item: item[1][1])
        return [tx_id for tx_id, _ in sorted_items]

    def create_themis_proposal(self) -> ThemisProposal:
        """Generate and sign local order proposal for the current view."""
        order = self.get_local_order()
        prop = ThemisProposal(
            replica_id=self.node_id,
            view=self.consensus.view,
            tx_order=order,
        )
        prop.sign(self.key_pair)
        return prop

    def receive_themis_proposal(self, proposal: ThemisProposal) -> bool:
        """Leader collects Themis proposal from a replica."""
        sender_kp = self.validator_keys.get(proposal.replica_id)
        if not sender_kp or not proposal.verify(sender_kp):
            return False
        self.collected_themis_proposals[proposal.replica_id] = proposal
        return True

    # -------------------------------------------------------------------------
    # Leader Proposal Creation
    # -------------------------------------------------------------------------
    def propose_block(self) -> Optional[Tuple[Block, List[ThemisProposal]]]:
        """Leader compiles Themis fair order and generates proposal block."""
        if not self.consensus.is_leader():
            return None

        # Automatically include our own proposal
        if self.node_id not in self.collected_themis_proposals:
            self.collected_themis_proposals[self.node_id] = self.create_themis_proposal()

        # Need at least n - f proposals
        required_proposals = self.cfg.n - self.cfg.f
        if len(self.collected_themis_proposals) < required_proposals:
            return None

        # Build map of all pending transactions
        tx_map = {tx_id: tx for tx_id, (tx, _) in self.mempool.items()}
        proposals_list = list(self.collected_themis_proposals.values())

        # Execute Themis fair ordering algorithm
        ordered_txs, _ = themis_order_transactions(tx_map, proposals_list, self.themis_cfg)

        parent = self.blockchain.get_head()
        themis_proposals_dict = {p.replica_id: p.tx_order for p in proposals_list}
        themis_signatures_dict = {p.replica_id: p.signature for p in proposals_list}

        block = self.consensus.create_proposal_block(
            height=parent.height + 1,
            parent=parent,
            ordered_txs=ordered_txs,
            themis_proposals=themis_proposals_dict,
            themis_signatures=themis_signatures_dict,
            state_root=self.blockchain.state.state_root(),
        )
        return block, proposals_list

    # -------------------------------------------------------------------------
    # Consensus Round Step Handlers
    # -------------------------------------------------------------------------
    def handle_proposal(
        self,
        block: Block,
        proposals_list: Optional[List[ThemisProposal]] = None,
    ) -> Optional[Vote]:
        return self.consensus.on_receive_proposal(block, proposals_list)

    def handle_prepare_vote(self, vote: Vote) -> Optional[QuorumCertificate]:
        return self.consensus.on_receive_prepare_vote(vote)

    def handle_prepare_qc(self, qc: QuorumCertificate) -> Optional[Vote]:
        return self.consensus.on_receive_prepare_qc(qc)

    def handle_precommit_vote(self, vote: Vote) -> Optional[QuorumCertificate]:
        return self.consensus.on_receive_precommit_vote(vote)

    def handle_precommit_qc(self, qc: QuorumCertificate) -> Optional[Vote]:
        return self.consensus.on_receive_precommit_qc(qc)

    def handle_commit_vote(self, vote: Vote) -> Optional[QuorumCertificate]:
        return self.consensus.on_receive_commit_vote(vote)

    def handle_commit_qc(self, qc: QuorumCertificate) -> Optional[Block]:
        committed_block = self.consensus.on_receive_commit_qc(qc)
        if committed_block:
            # Commit to ledger
            self.blockchain.commit_block(committed_block)
            # Remove committed transactions from mempool
            for tx in committed_block.transactions:
                self.mempool.pop(tx.tx_id, None)
            # Clear collected proposals for previous view
            self.collected_themis_proposals.clear()
        return committed_block
