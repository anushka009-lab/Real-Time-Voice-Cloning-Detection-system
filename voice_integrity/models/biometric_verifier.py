"""Speaker Voiceprint Biometric & Cross-Session Consistency Module.

Extracts discriminative speaker identity embeddings and performs cross-session
consistency verification against enrolled genuine reference samples.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from voice_integrity.dsp.spectral import SpectralAnalyzer
from voice_integrity.dsp.prosody import ProsodyAnalyzer


class SpeakerBiometricVerifier:
    """Acoustic identity embedding extractor and cross-session biometric verifier."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.spectral_analyzer = SpectralAnalyzer(sample_rate=sample_rate)
        self.prosody_analyzer = ProsodyAnalyzer(sample_rate=sample_rate)
        
        # In-memory enrolled voiceprint vault: {speaker_id: embedding_vector}
        self.enrolled_vault: Dict[str, np.ndarray] = {}

    def extract_voiceprint(self, audio: np.ndarray) -> np.ndarray:
        """Extract a fixed-length acoustic speaker identity embedding vector (56 dimensions).
        
        Combines:
        - 20 LFCC means
        - 20 LFCC standard deviations
        - 4 Spectral moments (centroid, spread, rolloff, flux)
        - 6 Prosodic statistics (mean F0, std F0, CV F0, Jitter local, Shimmer local, HNR)
        - 6 High-frequency subband energy distributions
        """
        mag, phase, complex_spec = self.spectral_analyzer.compute_stft(audio)
        lfcc = self.spectral_analyzer.compute_lfcc(mag)
        
        lfcc_mean = np.mean(lfcc, axis=1)  # 20 dims
        lfcc_std = np.std(lfcc, axis=1)    # 20 dims
        
        spec_desc = self.spectral_analyzer.compute_spectral_descriptors(mag)
        desc_vec = np.array([
            spec_desc["spectral_centroid"] / 4000.0,
            spec_desc["spectral_spread"] / 2000.0,
            spec_desc["spectral_rolloff"] / 8000.0,
            spec_desc["spectral_flux"] / 100.0,
        ], dtype=np.float32)  # 4 dims
        
        prosody_feats = self.prosody_analyzer.analyze_audio_segment(audio)
        prosody_vec = np.array([
            prosody_feats["mean_f0"] / 300.0,
            prosody_feats["std_f0"] / 100.0,
            prosody_feats["cv_f0"],
            prosody_feats["jitter_local_percent"] / 5.0,
            prosody_feats["shimmer_local_percent"] / 10.0,
            prosody_feats["hnr_db"] / 30.0,
        ], dtype=np.float32)  # 6 dims
        
        # 6 subband relative energy splits
        n_freqs = mag.shape[0]
        splits = np.array_split(np.arange(n_freqs), 6)
        subband_energies = []
        total_e = np.sum(mag ** 2) + 1e-12
        for s in splits:
            sb_e = np.sum(mag[s, :] ** 2) / total_e
            subband_energies.append(sb_e)
        subband_vec = np.array(subband_energies, dtype=np.float32)  # 6 dims
        
        # Concatenate into 56-dim feature vector
        raw_vector = np.concatenate([lfcc_mean, lfcc_std, desc_vec, prosody_vec, subband_vec])
        
        # L2 Unit-Sphere Normalization
        norm = np.linalg.norm(raw_vector)
        if norm > 1e-8:
            embedding = raw_vector / norm
        else:
            embedding = raw_vector
            
        return embedding.astype(np.float32)

    def enroll_speaker(self, speaker_id: str, audio: np.ndarray) -> np.ndarray:
        """Enroll or update a genuine speaker reference voiceprint."""
        embedding = self.extract_voiceprint(audio)
        self.enrolled_vault[speaker_id] = embedding
        return embedding

    def verify_speaker(
        self,
        speaker_id: str,
        test_audio: np.ndarray,
        match_threshold: float = 0.78
    ) -> Dict[str, any]:
        """Compare incoming audio against enrolled genuine speaker reference.
        
        Returns:
            similarity: Cosine similarity (-1.0 to 1.0)
            is_match: bool
            status: 'VERIFIED', 'BIOMETRIC_MISMATCH', 'NOT_ENROLLED'
        """
        if speaker_id not in self.enrolled_vault:
            return {
                "speaker_id": speaker_id,
                "similarity": 0.0,
                "is_match": None,
                "status": "NOT_ENROLLED",
                "message": f"Speaker '{speaker_id}' has no prior enrolled baseline voiceprint.",
            }

        ref_embedding = self.enrolled_vault[speaker_id]
        test_embedding = self.extract_voiceprint(test_audio)

        # Cosine similarity between L2-normalized vectors = dot product
        similarity = float(np.dot(ref_embedding, test_embedding))
        similarity = max(min(similarity, 1.0), -1.0)

        is_match = similarity >= match_threshold
        status = "VERIFIED" if is_match else "BIOMETRIC_MISMATCH"

        return {
            "speaker_id": speaker_id,
            "similarity": similarity,
            "threshold": match_threshold,
            "is_match": is_match,
            "status": status,
            "confidence_percent": round(max(0.0, similarity) * 100.0, 1),
        }
