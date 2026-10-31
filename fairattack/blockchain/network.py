"""Asynchronous multi-node network transport and consensus coordinator.

Enables:
  * Running concurrent HotStuff validator clusters.
  * Broadcasting client transactions to validator mempools.
  * Executing complete multi-phase HotStuff consensus rounds with Themis fair ordering.
  * Injecting network delays and testing Byzantine fault tolerance (e.g. f offline nodes).
"""
from __future__ import annotations
import asyncio, random, time
from typing import Dict, List, Optional, Set, Tuple
from .models import Block, Transaction, Vote, QuorumCertificate
from .node import HotStuffNode
from .themis import ThemisProposal


class AsyncNetwork:
    """Network coordinator driving consensus across a cluster of HotStuff nodes."""

    def __init__(self, nodes: Dict[str, HotStuffNode]):
        self.nodes = nodes
        self.offline_nodes: Set[str] = set()

    def set_offline(self, node_id: str, offline: bool = True):
        """Simulate an offline or Byzantine crashing node."""
        if offline:
            self.offline_nodes.add(node_id)
        else:
            self.offline_nodes.discard(node_id)

    def broadcast_transaction(self, tx: Transaction):
        """Deliver a client transaction to all online nodes."""
        for nid, node in self.nodes.items():
            if nid not in self.offline_nodes:
                node.submit_transaction(tx)

    def run_consensus_round(self) -> Optional[Block]:
        """Execute one complete 4-stage HotStuff consensus round with Themis fair ordering.

        Steps:
          0. Exchange Themis order proposals among replicas.
          1. Leader proposes block with Themis-ordered txs -> replicas vote PREPARE.
          2. Leader aggregates Prepare QC -> broadcasts to replicas -> replicas vote PRECOMMIT.
          3. Leader aggregates Pre-Commit QC -> broadcasts to replicas -> replicas vote COMMIT.
          4. Leader aggregates Commit QC -> broadcasts to replicas -> replicas decide & commit to ledger.
        """
        # Determine leader for current view
        # Assume all online nodes agree on view; find leader
        sample_node = next(iter(self.nodes.values()))
        curr_view = sample_node.consensus.view
        leader_id = sample_node.cfg.leader_id(curr_view)
        leader = self.nodes.get(leader_id)

        if not leader or leader_id in self.offline_nodes:
            # Leader is offline, view change would be needed in live pacemaker
            return None

        # ---------------------------------------------------------------------
        # Step 0: Themis Proposals Exchange
        # ---------------------------------------------------------------------
        for nid, node in self.nodes.items():
            if nid not in self.offline_nodes:
                proposal = node.create_themis_proposal()
                leader.receive_themis_proposal(proposal)

        # ---------------------------------------------------------------------
        # Step 1: Proposal & Prepare Phase
        # ---------------------------------------------------------------------
        prop_res = leader.propose_block()
        if not prop_res:
            return None
        proposed_block, proposals_list = prop_res

        prepare_qc: Optional[QuorumCertificate] = None
        for nid, node in self.nodes.items():
            if nid in self.offline_nodes:
                continue
            vote = node.handle_proposal(proposed_block, proposals_list)
            if vote:
                qc = leader.handle_prepare_vote(vote)
                if qc:
                    prepare_qc = qc

        if not prepare_qc:
            return None

        # ---------------------------------------------------------------------
        # Step 2: Pre-Commit Phase
        # ---------------------------------------------------------------------
        precommit_qc: Optional[QuorumCertificate] = None
        for nid, node in self.nodes.items():
            if nid in self.offline_nodes:
                continue
            vote = node.handle_prepare_qc(prepare_qc)
            if vote:
                qc = leader.handle_precommit_vote(vote)
                if qc:
                    precommit_qc = qc

        if not precommit_qc:
            return None

        # ---------------------------------------------------------------------
        # Step 3: Commit Phase
        # ---------------------------------------------------------------------
        commit_qc: Optional[QuorumCertificate] = None
        for nid, node in self.nodes.items():
            if nid in self.offline_nodes:
                continue
            vote = node.handle_precommit_qc(precommit_qc)
            if vote:
                qc = leader.handle_commit_vote(vote)
                if qc:
                    commit_qc = qc

        if not commit_qc:
            return None

        # ---------------------------------------------------------------------
        # Step 4: Decide Phase (Commit to Ledger)
        # ---------------------------------------------------------------------
        committed_block: Optional[Block] = None
        for nid, node in self.nodes.items():
            if nid in self.offline_nodes:
                continue
            blk = node.handle_commit_qc(commit_qc)
            if blk:
                committed_block = blk

        return committed_block
