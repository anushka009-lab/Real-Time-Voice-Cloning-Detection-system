"""Ledger and Blockchain Layer for Voice Integrity Verification.

Provides tamper-evident Merkle Trees, cryptographic audit logs, and EVM smart contract
anchoring for voice authentication proofs.
"""

from voice_integrity.ledger.merkle import (
    MerkleTree,
    VerificationLogEntry,
    sha256_hash,
    hash_pair,
)
from voice_integrity.ledger.anchor import (
    CallSessionLedger,
    CallAuditReceipt,
)

__all__ = [
    "MerkleTree",
    "VerificationLogEntry",
    "CallSessionLedger",
    "CallAuditReceipt",
    "sha256_hash",
    "hash_pair",
]
