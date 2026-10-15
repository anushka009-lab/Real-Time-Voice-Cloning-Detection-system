"""Unit and Integration Tests for Voice Integrity & Anti-Cloning Engine.
"""

import unittest
import numpy as np
from voice_integrity.audio_io import (
    load_wav,
    save_wav,
    simulate_voip_channel,
    simulate_mu_law,
    apply_telephony_bandpass,
)
from voice_integrity.dsp.spectral import SpectralAnalyzer
from voice_integrity.dsp.prosody import ProsodyAnalyzer
from voice_integrity.models.vocoder_detector import VocoderFootprintDetector
from voice_integrity.models.biometric_verifier import SpeakerBiometricVerifier
from voice_integrity.models.risk_engine import (
    VoiceIntegrityRiskEngine,
    RiskLevel,
    RecommendedAction,
)
from voice_integrity.benchmark.evaluator import generate_speech_sample


class TestVoiceIntegrity(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.spectral_analyzer = SpectralAnalyzer(sample_rate=self.sample_rate)
        self.prosody_analyzer = ProsodyAnalyzer(sample_rate=self.sample_rate)
        self.vocoder_detector = VocoderFootprintDetector(sample_rate=self.sample_rate)
        self.risk_engine = VoiceIntegrityRiskEngine(sample_rate=self.sample_rate)

    def test_audio_degradation_pipeline(self):
        """Verify telephony channel simulation preserves audio dimensions."""
        audio = np.sin(2.0 * np.pi * 440.0 * np.arange(16000) / 16000).astype(np.float32)
        degraded = simulate_voip_channel(audio, sample_rate=16000)
        self.assertEqual(len(degraded), len(audio))
        self.assertTrue(np.all(np.isfinite(degraded)))

    def test_spectral_analyzer_dimensions(self):
        """Verify STFT, LFCC, and bispectrum computation on test signal."""
        audio = np.random.normal(0, 0.1, size=16000).astype(np.float32)
        mag, phase, complex_spec = self.spectral_analyzer.compute_stft(audio)
        self.assertEqual(mag.shape, phase.shape)
        
        lfcc = self.spectral_analyzer.compute_lfcc(mag)
        self.assertEqual(lfcc.shape[0], self.spectral_analyzer.n_lfcc)
        
        bicoherence = self.spectral_analyzer.compute_bispectrum_coupling(complex_spec)
        self.assertGreaterEqual(bicoherence, 0.0)
        self.assertLessEqual(bicoherence, 1.0)

    def test_prosody_analyzer_jitter_shimmer(self):
        """Verify pitch, jitter, and shimmer extraction."""
        # 1-second 150Hz sine tone
        t = np.arange(16000) / 16000.0
        audio = (0.5 * np.sin(2.0 * np.pi * 150.0 * t)).astype(np.float32)
        
        feats = self.prosody_analyzer.analyze_audio_segment(audio)
        self.assertIn("jitter_local_percent", feats)
        self.assertIn("shimmer_local_percent", feats)
        self.assertIn("mean_f0", feats)
        self.assertAlmostEqual(feats["mean_f0"], 150.0, delta=10.0)

    def test_vocoder_detector_lpc(self):
        """Verify LPC residual analysis and high-frequency periodicity."""
        audio = generate_speech_sample(is_synthetic=False, duration_sec=1.0)
        feats = self.vocoder_detector.analyze_audio_segment(audio)
        self.assertIn("residual_kurtosis", feats)
        self.assertIn("high_freq_harmonicity", feats)
        self.assertGreater(feats["residual_kurtosis"], 0.0)

    def test_speaker_biometrics(self):
        """Verify speaker enrollment and self-similarity consistency."""
        verifier = SpeakerBiometricVerifier(sample_rate=self.sample_rate)
        audio_ref = generate_speech_sample(is_synthetic=False, duration_sec=2.0, base_f0=130.0)
        audio_same = generate_speech_sample(is_synthetic=False, duration_sec=2.0, base_f0=132.0)
        audio_diff = generate_speech_sample(is_synthetic=False, duration_sec=2.0, base_f0=240.0)

        verifier.enroll_speaker("USER_ALICE", audio_ref)
        
        # Self-similarity should be high
        res_same = verifier.verify_speaker("USER_ALICE", audio_same)
        self.assertGreater(res_same["similarity"], 0.70)
        
        # Different speaker register should have lower similarity
        res_diff = verifier.verify_speaker("USER_ALICE", audio_diff)
        self.assertLess(res_diff["similarity"], res_same["similarity"])

    def test_risk_engine_discrimination(self):
        """Verify engine accurately assigns lower risk to genuine voice and higher risk to clone."""
        genuine_audio = generate_speech_sample(is_synthetic=False, duration_sec=2.5, base_f0=140.0)
        cloned_audio = generate_speech_sample(is_synthetic=True, duration_sec=2.5, base_f0=140.0)

        # Feed genuine audio
        self.risk_engine.reset_state()
        chunk_size = 4000  # 250ms at 16kHz
        for i in range(0, len(genuine_audio) - chunk_size, chunk_size):
            res_genuine = self.risk_engine.process_chunk(genuine_audio[i : i + chunk_size])

        # Feed cloned audio
        self.risk_engine.reset_state()
        for i in range(0, len(cloned_audio) - chunk_size, chunk_size):
            res_clone = self.risk_engine.process_chunk(cloned_audio[i : i + chunk_size])

        # Risk score for clone must exceed genuine audio
        self.assertGreater(res_clone.rolling_risk_score, res_genuine.rolling_risk_score)
        self.assertLess(res_genuine.rolling_risk_score, 0.40)
        self.assertGreater(res_clone.rolling_risk_score, 0.55)

    def test_real_time_chunk_latency(self):
        """Verify chunk latency is well under real-time budget (e.g. < 40ms for 250ms audio chunk)."""
        chunk = np.random.normal(0, 0.1, size=4000).astype(np.float32)
        res = self.risk_engine.process_chunk(chunk)
        self.assertLess(res.latency_ms, 50.0)

    def test_blockchain_merkle_tree_integrity(self):
        """Verify Merkle tree construction, proof generation, and tamper detection."""
        from voice_integrity.ledger import MerkleTree, CallSessionLedger, sha256_hash
        
        leaves = [sha256_hash(f"frame_log_{i}") for i in range(8)]
        tree = MerkleTree(leaves)
        self.assertTrue(tree.root.startswith("0x"))
        self.assertEqual(len(tree.root), 66)  # '0x' + 64 hex chars
        
        # Test valid proof for leaf 0 and leaf 5
        for idx in [0, 5]:
            proof = tree.get_proof(idx)
            is_valid = MerkleTree.verify_proof(leaves[idx], proof, tree.root)
            self.assertTrue(is_valid)

        # Test tamper detection: modified leaf MUST fail verification
        tampered_leaf = sha256_hash("tampered_modified_frame_log")
        tamper_check = MerkleTree.verify_proof(tampered_leaf, tree.get_proof(0), tree.root)
        self.assertFalse(tamper_check)

    def test_call_session_ledger_sealing(self):
        """Verify call session ledger seals frames into on-chain receipt."""
        from voice_integrity.ledger import CallSessionLedger
        ledger = CallSessionLedger()
        
        chunk = np.random.normal(0, 0.1, size=4000).astype(np.float32)
        res = self.risk_engine.process_chunk(chunk)
        
        # Record 4 frames
        for _ in range(4):
            ledger.record_frame(res, chunk, caller_id="+91-98765-43210")
            
        receipt = ledger.seal_session(res, caller_id="+91-98765-43210")
        self.assertEqual(receipt.total_frames_analyzed, 4)
        self.assertTrue(receipt.merkle_root.startswith("0x"))
        self.assertTrue(receipt.on_chain_tx_hash.startswith("0x"))


if __name__ == "__main__":
    unittest.main()
