"""HotStuff BFT Consensus State Machine (Yin et al., PODC 2019).

Implements the 3-phase HotStuff BFT consensus protocol:
  1. Prepare Phase: Leader proposes block with justify QC; replicas verify safety & Themis order, vote PREPARE.
  2. Pre-Commit Phase: Leader broadcasts prepare QC; replicas lock on QC and vote PRECOMMIT.
  3. Commit Phase: Leader broadcasts precommit QC; replicas vote COMMIT.
  4. Decide Phase: Leader broadcasts commit QC; replicas commit block to blockchain ledger.
"""
from __future__ import annotations
import copy, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from .models import Block, Transaction, Vote, QuorumCertificate, Stage
from .crypto import KeyPair, sha256_hex
from .themis import ThemisConfig, ThemisProposal, verify_themis_order


@dataclass
class HotStuffConfig:
    """Consensus parameters for HotStuff BFT."""
    n: int = 4
    f: int = 1
    view_timeout_sec: float = 2.0

    @property
    def quorum_size(self) -> int:
        """2f + 1 quorum requirement."""
        return 2 * self.f + 1

    def leader_id(self, view: int) -> str:
        """Round-robin leader election for view."""
        return f"node_{view % self.n}"


class HotStuffConsensus:
    """Per-validator HotStuff consensus state machine."""

    def __init__(
        self,
        node_id: str,
        key_pair: KeyPair,
        validator_keys: Dict[str, KeyPair],
        cfg: HotStuffConfig,
        themis_cfg: ThemisConfig,
    ):
        self.node_id = node_id
        self.key_pair = key_pair
        self.validator_keys = validator_keys
        self.cfg = cfg
        self.themis_cfg = themis_cfg

        self.view: int = 1
        self.locked_qc: Optional[QuorumCertificate] = None
        self.high_qc: Optional[QuorumCertificate] = None

        # Vote aggregation buffers (for leader)
        self.prepare_votes: Dict[str, Vote] = {}
        self.precommit_votes: Dict[str, Vote] = {}
        self.commit_votes: Dict[str, Vote] = {}

        # Proposed blocks cache: block_hash -> Block
        self.blocks: Dict[str, Block] = {}

    def is_leader(self, view: Optional[int] = None) -> bool:
        v = self.view if view is None else view
        return self.cfg.leader_id(v) == self.node_id

    # -------------------------------------------------------------------------
    # Leader Proposal Construction
    # -------------------------------------------------------------------------
    def create_proposal_block(
        self,
        height: int,
        parent: Block,
        ordered_txs: List[Transaction],
        themis_proposals: Dict[str, List[str]],
        themis_signatures: Dict[str, str],
        state_root: str,
    ) -> Block:
        """Leader builds a proposed block extending the parent with Themis-ordered txs."""
        justify = self.high_qc or parent.justify_qc
        block = Block(
            height=height,
            view=self.view,
            proposer_id=self.node_id,
            parent_hash=parent.block_hash,
            justify_qc=justify,
            transactions=ordered_txs,
            themis_proposals=themis_proposals,
            themis_signatures=themis_signatures,
            state_root=state_root,
            timestamp=time.time(),
        )
        self.blocks[block.block_hash] = block
        return block

    # -------------------------------------------------------------------------
    # Safety Check (Voting Rule)
    # -------------------------------------------------------------------------
    def check_safety(self, block: Block) -> bool:
        """HotStuff safety predicate: extends locked_qc or view > locked_qc.view."""
        if self.locked_qc is None:
            return True
        # Extends locked QC's block
        if block.parent_hash == self.locked_qc.block_hash:
            return True
        # Liveness rule: view is strictly higher than locked QC
        if block.view > self.locked_qc.view:
            return True
        return False

    # -------------------------------------------------------------------------
    # Phase 1: Prepare
    # -------------------------------------------------------------------------
    def on_receive_proposal(
        self,
        block: Block,
        proposals_list: Optional[List[ThemisProposal]] = None,
    ) -> Optional[Vote]:
        """Replica verifies proposed block and Themis fair order proof, then votes PREPARE."""
        # 1. Basic checks
        if block.view < self.view:
            return None
        self.blocks[block.block_hash] = block

        # 2. Safety check against locked_qc
        if not self.check_safety(block):
            return None

        # 3. Themis Order Verification (if proposals provided)
        if proposals_list is not None and block.transactions:
            if len(proposals_list) < (self.cfg.n - self.cfg.f):
                return None
            if not verify_themis_order(block.transactions, proposals_list, self.themis_cfg):
                return None

        # 4. Cast signed PREPARE vote
        vote = Vote(
            stage=Stage.PREPARE,
            view=block.view,
            block_hash=block.block_hash,
            voter_id=self.node_id,
        )
        vote.sign(self.key_pair)
        return vote

    def on_receive_prepare_vote(self, vote: Vote) -> Optional[QuorumCertificate]:
        """Leader aggregates PREPARE votes into prepare_qc once 2f+1 threshold is reached."""
        if not self.is_leader(vote.view):
            return None
        if vote.stage != Stage.PREPARE:
            return None
        voter_kp = self.validator_keys.get(vote.voter_id)
        if not voter_kp or not vote.verify(voter_kp):
            return None

        self.prepare_votes[vote.voter_id] = vote
        if len(self.prepare_votes) >= self.cfg.quorum_size:
            qc = QuorumCertificate(
                stage=Stage.PREPARE,
                view=vote.view,
                block_hash=vote.block_hash,
                votes={vid: v.signature for vid, v in self.prepare_votes.items()},
            )
            self.high_qc = qc
            return qc
        return None

    # -------------------------------------------------------------------------
    # Phase 2: Pre-Commit
    # -------------------------------------------------------------------------
    def on_receive_prepare_qc(self, qc: QuorumCertificate) -> Optional[Vote]:
        """Replica verifies prepare_qc, updates locked_qc, and votes PRECOMMIT."""
        if qc.stage != Stage.PREPARE:
            return None
        if not qc.is_valid(self.cfg.quorum_size, self.validator_keys):
            return None

        self.locked_qc = qc
        self.high_qc = qc

        vote = Vote(
            stage=Stage.PRECOMMIT,
            view=qc.view,
            block_hash=qc.block_hash,
            voter_id=self.node_id,
        )
        vote.sign(self.key_pair)
        return vote

    def on_receive_precommit_vote(self, vote: Vote) -> Optional[QuorumCertificate]:
        """Leader aggregates PRECOMMIT votes into precommit_qc once 2f+1 threshold is reached."""
        if not self.is_leader(vote.view):
            return None
        if vote.stage != Stage.PRECOMMIT:
            return None
        voter_kp = self.validator_keys.get(vote.voter_id)
        if not voter_kp or not vote.verify(voter_kp):
            return None

        self.precommit_votes[vote.voter_id] = vote
        if len(self.precommit_votes) >= self.cfg.quorum_size:
            qc = QuorumCertificate(
                stage=Stage.PRECOMMIT,
                view=vote.view,
                block_hash=vote.block_hash,
                votes={vid: v.signature for vid, v in self.precommit_votes.items()},
            )
            self.high_qc = qc
            return qc
        return None

    # -------------------------------------------------------------------------
    # Phase 3: Commit
    # -------------------------------------------------------------------------
    def on_receive_precommit_qc(self, qc: QuorumCertificate) -> Optional[Vote]:
        """Replica verifies precommit_qc and votes COMMIT."""
        if qc.stage != Stage.PRECOMMIT:
            return None
        if not qc.is_valid(self.cfg.quorum_size, self.validator_keys):
            return None

        self.high_qc = qc

        vote = Vote(
            stage=Stage.COMMIT,
            view=qc.view,
            block_hash=qc.block_hash,
            voter_id=self.node_id,
        )
        vote.sign(self.key_pair)
        return vote

    def on_receive_commit_vote(self, vote: Vote) -> Optional[QuorumCertificate]:
        """Leader aggregates COMMIT votes into commit_qc once 2f+1 threshold is reached."""
        if not self.is_leader(vote.view):
            return None
        if vote.stage != Stage.COMMIT:
            return None
        voter_kp = self.validator_keys.get(vote.voter_id)
        if not voter_kp or not vote.verify(voter_kp):
            return None

        self.commit_votes[vote.voter_id] = vote
        if len(self.commit_votes) >= self.cfg.quorum_size:
            qc = QuorumCertificate(
                stage=Stage.COMMIT,
                view=vote.view,
                block_hash=vote.block_hash,
                votes={vid: v.signature for vid, v in self.commit_votes.items()},
            )
            self.high_qc = qc
            return qc
        return None

    # -------------------------------------------------------------------------
    # Phase 4: Decide & View Advance
    # -------------------------------------------------------------------------
    def on_receive_commit_qc(self, qc: QuorumCertificate) -> Optional[Block]:
        """Replica verifies commit_qc, commits the block, and advances view."""
        if qc.stage != Stage.COMMIT:
            return None
        if not qc.is_valid(self.cfg.quorum_size, self.validator_keys):
            return None

        block = self.blocks.get(qc.block_hash)
        if not block:
            return None

        # Reset vote buffers and advance view
        self.prepare_votes.clear()
        self.precommit_votes.clear()
        self.commit_votes.clear()
        self.view = max(self.view + 1, qc.view + 1)
        return block
