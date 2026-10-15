"""VoiceIntegrity: AI-Powered Real-Time Voice Integrity & Anti-Cloning Verification Framework.

A high-performance acoustic, spectral, prosodic, and neural vocoder verification pipeline
for detecting synthetic speech and voice cloning in live telephony and VoIP streams.
"""

from voice_integrity.audio_io import (
    load_wav,
    save_wav,
    resample_audio,
    stream_chunker,
    simulate_voip_channel,
    simulate_mu_law,
    apply_telephony_bandpass,
    add_gaussian_noise,
)
from voice_integrity.dsp.spectral import SpectralAnalyzer
from voice_integrity.dsp.prosody import ProsodyAnalyzer
from voice_integrity.models.vocoder_detector import VocoderFootprintDetector
from voice_integrity.models.biometric_verifier import SpeakerBiometricVerifier
from voice_integrity.models.risk_engine import (
    VoiceIntegrityRiskEngine,
    RiskLevel,
    RecommendedAction,
    FrameEvaluationResult,
    ForensicEvidence,
)
from voice_integrity.ledger import (
    MerkleTree,
    CallSessionLedger,
    CallAuditReceipt,
)

__version__ = "1.1.0"

__all__ = [
    "load_wav",
    "save_wav",
    "resample_audio",
    "stream_chunker",
    "simulate_voip_channel",
    "simulate_mu_law",
    "apply_telephony_bandpass",
    "add_gaussian_noise",
    "SpectralAnalyzer",
    "ProsodyAnalyzer",
    "VocoderFootprintDetector",
    "SpeakerBiometricVerifier",
    "VoiceIntegrityRiskEngine",
    "RiskLevel",
    "RecommendedAction",
    "FrameEvaluationResult",
    "ForensicEvidence",
    "MerkleTree",
    "CallSessionLedger",
    "CallAuditReceipt",
]
