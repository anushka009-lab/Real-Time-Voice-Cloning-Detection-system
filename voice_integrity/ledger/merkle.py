"""Cryptographic Merkle Tree and Tamper-Evident Verification Module.

Computes Merkle Trees over real-time call verification frame logs to produce
cryptographic proofs (Merkle Roots) for anchoring into EVM-compatible blockchains.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional


def sha256_hash(data: bytes | str) -> str:
    """Compute SHA-256 hexadecimal hash."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_pair(left_hex: str, right_hex: str) -> str:
    """Concatenate two hex hashes in sorted canonical order and hash them."""
    # Lexicographically sort pairs to prevent Merkle tree ordering malleability
    if left_hex <= right_hex:
        combined = bytes.fromhex(left_hex) + bytes.fromhex(right_hex)
    else:
        combined = bytes.fromhex(right_hex) + bytes.fromhex(left_hex)
    return hashlib.sha256(combined).hexdigest()


@dataclass
class VerificationLogEntry:
    """Single audio chunk verification log record for cryptographic sealing."""
    frame_index: int
    timestamp: float
    caller_id_hash: str
    rolling_risk_score: float
    risk_level: str
    recommended_action: str
    latency_ms: float
    top_forensic_cues: List[str]
    audio_chunk_hash: str

    def to_canonical_json(self) -> str:
        """Deterministic canonical JSON serialization for consistent hashing."""
        data = asdict(self)
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def compute_leaf_hash(self) -> str:
        """Compute SHA-256 leaf hash for the log entry."""
        return sha256_hash(self.to_canonical_json())


class MerkleTree:
    """Tamper-evident Merkle Tree for voice integrity verification logs."""

    def __init__(self, leaves: Optional[List[str]] = None):
        self.leaves: List[str] = leaves or []
        self.levels: List[List[str]] = []
        if self.leaves:
            self._build_tree()

    def add_leaf(self, leaf_hash: str) -> None:
        """Add a new leaf hash and rebuild tree."""
        self.leaves.append(leaf_hash)
        self._build_tree()

    def _build_tree(self) -> None:
        """Construct the Merkle tree levels up to the root."""
        if not self.leaves:
            self.levels = [["0" * 64]]
            return

        current_level = list(self.leaves)
        self.levels = [current_level]

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Duplicate last leaf if odd count (standard Bitcoin/Ethereum convention)
                    right = left
                next_level.append(hash_pair(left, right))
            current_level = next_level
            self.levels.append(current_level)

    @property
    def root(self) -> str:
        """Return 32-byte Merkle root in hex format (prefixed with 0x)."""
        if not self.levels or not self.levels[-1]:
            return "0x" + "0" * 64
        return "0x" + self.levels[-1][0]

    def get_proof(self, leaf_index: int) -> List[Dict[str, str]]:
        """Generate Merkle audit proof (path of sibling hashes) for a given leaf.
        
        Returns:
            list of dicts: [{"position": "left" | "right", "hash": "0x..."}]
        """
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise IndexError("Leaf index out of bounds")

        proof: List[Dict[str, str]] = []
        idx = leaf_index

        for level in self.levels[:-1]:
            is_right_child = (idx % 2 == 1)
            sibling_idx = idx - 1 if is_right_child else idx + 1
            if sibling_idx < len(level):
                sibling_hash = level[sibling_idx]
            else:
                sibling_hash = level[idx]  # Duplicated odd node

            proof.append({
                "position": "left" if is_right_child else "right",
                "hash": "0x" + sibling_hash,
            })
            idx //= 2

        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Dict[str, str]], expected_root: str) -> bool:
        """Verify whether a leaf hash belongs to the given Merkle Root using the proof."""
        current = leaf_hash.replace("0x", "")
        expected = expected_root.replace("0x", "")

        for item in proof:
            sibling = item["hash"].replace("0x", "")
            current = hash_pair(current, sibling)

        return current.lower() == expected.lower()
