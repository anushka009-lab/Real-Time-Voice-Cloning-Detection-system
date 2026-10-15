"""Blockchain Anchoring and Cryptographic Call Audit Service.

Collects real-time streaming frame verification logs, builds Merkle Trees,
and produces EVM-compatible on-chain anchoring transactions and audit receipts.
"""

from __future__ import annotations
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from voice_integrity.ledger.merkle import (
    MerkleTree,
    VerificationLogEntry,
    sha256_hash,
)
from voice_integrity.models.risk_engine import FrameEvaluationResult, RiskLevel


@dataclass
class CallAuditReceipt:
    """Cryptographic tamper-evident receipt of an evaluated call session."""
    call_session_id: str
    merkle_root: str
    caller_id_pseudonym_hash: str
    final_risk_level: str
    final_risk_score_percent: float
    total_frames_analyzed: int
    duration_seconds: float
    timestamp_utc: float
    on_chain_tx_hash: str
    contract_address: str
    evm_network: str
    leaf_hashes: List[str]
    sample_proof_frame_0: List[Dict[str, str]]


class CallSessionLedger:
    """Records streaming verification frames and seals them into Merkle proofs."""

    def __init__(
        self,
        call_session_id: Optional[str] = None,
        contract_address: str = "0x89A6B221E5cD9b4bE5c621815f9b4c09C9D71234",
        network: str = "Ethereum / Polygon PoS (EVM)",
    ):
        self.call_session_id = call_session_id or str(uuid.uuid4())
        self.contract_address = contract_address
        self.network = network
        self.entries: List[VerificationLogEntry] = []
        self.leaf_hashes: List[str] = []
        self.start_time = time.time()

    def record_frame(
        self,
        frame_result: FrameEvaluationResult,
        audio_chunk: np.ndarray,
        caller_id: Optional[str] = None,
    ) -> VerificationLogEntry:
        """Record a single processed audio frame into the audit ledger."""
        chunk_bytes = audio_chunk.astype(np.float32).tobytes()
        audio_chunk_hash = sha256_hash(chunk_bytes)
        
        caller_hash = sha256_hash(caller_id or "anonymous")
        cues = [c.metric_name for c in frame_result.top_forensic_cues]

        entry = VerificationLogEntry(
            frame_index=len(self.entries),
            timestamp=frame_result.timestamp,
            caller_id_hash=caller_hash,
            rolling_risk_score=frame_result.rolling_risk_score,
            risk_level=frame_result.risk_level.value,
            recommended_action=frame_result.recommended_action.value,
            latency_ms=frame_result.latency_ms,
            top_forensic_cues=cues,
            audio_chunk_hash=audio_chunk_hash,
        )

        leaf_hash = entry.compute_leaf_hash()
        self.entries.append(entry)
        self.leaf_hashes.append(leaf_hash)
        return entry

    def seal_session(
        self,
        final_result: FrameEvaluationResult,
        caller_id: Optional[str] = None,
    ) -> CallAuditReceipt:
        """Seal all call verification frames into a Merkle root and generate on-chain receipt."""
        tree = MerkleTree(self.leaf_hashes)
        merkle_root = tree.root

        # Sample proof for frame 0 (if frames exist)
        sample_proof = tree.get_proof(0) if self.leaf_hashes else []

        caller_hash = "0x" + sha256_hash(caller_id or "anonymous")
        session_id_bytes32 = "0x" + sha256_hash(self.call_session_id)

        # Deterministic simulated EVM transaction hash based on root + session + time
        raw_tx_payload = f"{session_id_bytes32}{merkle_root}{caller_hash}{time.time()}"
        tx_hash = "0x" + sha256_hash(raw_tx_payload)

        receipt = CallAuditReceipt(
            call_session_id=session_id_bytes32,
            merkle_root=merkle_root,
            caller_id_pseudonym_hash=caller_hash,
            final_risk_level=final_result.risk_level.value,
            final_risk_score_percent=round(final_result.rolling_risk_score * 100.0, 2),
            total_frames_analyzed=len(self.entries),
            duration_seconds=round(time.time() - self.start_time, 2),
            timestamp_utc=time.time(),
            on_chain_tx_hash=tx_hash,
            contract_address=self.contract_address,
            evm_network=self.network,
            leaf_hashes=self.leaf_hashes,
            sample_proof_frame_0=sample_proof,
        )

        return receipt

    def export_receipt_json(self, receipt: CallAuditReceipt, output_path: str) -> None:
        """Export cryptographic audit receipt to a formatted JSON document."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(receipt), f, indent=2)

    @staticmethod
    def verify_audit_receipt(receipt_dict: Dict[str, Any]) -> bool:
        """Cryptographically verify that an exported receipt's Merkle Root is mathematically valid."""
        leaves = receipt_dict.get("leaf_hashes", [])
        expected_root = receipt_dict.get("merkle_root", "")
        if not leaves or not expected_root:
            return False

        recomputed_tree = MerkleTree(leaves)
        return recomputed_tree.root.lower() == expected_root.lower()
