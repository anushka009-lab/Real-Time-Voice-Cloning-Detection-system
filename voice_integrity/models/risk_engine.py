"""Real-Time Impersonation Risk Scoring Engine & Banking Guardrail Module.

Fuses acoustic, spectral, prosodic, and vocoder features into a calibrated
real-time Impersonation Risk Score (0.0 to 1.0 / 0% to 100%).
Provides explainable forensic cues and configurable banking/enterprise guardrails.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from voice_integrity.dsp.spectral import SpectralAnalyzer
from voice_integrity.dsp.prosody import ProsodyAnalyzer
from voice_integrity.models.vocoder_detector import VocoderFootprintDetector
from voice_integrity.models.biometric_verifier import SpeakerBiometricVerifier


class RiskLevel(str, Enum):
    AUTHENTIC_HUMAN = "AUTHENTIC_HUMAN"      # Score < 0.35
    ELEVATED_SUSPICIOUS = "ELEVATED_SUSPICIOUS"  # 0.35 <= Score < 0.70
    CRITICAL_CLONE = "CRITICAL_CLONE"        # Score >= 0.70


class RecommendedAction(str, Enum):
    PROCEED_NORMAL = "PROCEED_NORMAL"
    STEP_UP_VERIFICATION = "STEP_UP_VERIFICATION"      # Require In-App Push / SMS OTP
    MANDATORY_CALL_BACK = "MANDATORY_CALL_BACK"        # Disconnect & call registered phone
    FREEZE_TRANSACTION = "FREEZE_TRANSACTION"          # Lock fund transfer & alert fraud desk


@dataclass
class ForensicEvidence:
    metric_name: str
    observed_value: float
    typical_human_range: str
    contribution_weight: float
    anomaly_description: str


@dataclass
class FrameEvaluationResult:
    timestamp: float
    frame_risk_score: float
    rolling_risk_score: float
    risk_level: RiskLevel
    recommended_action: RecommendedAction
    layer_scores: Dict[str, float]
    top_forensic_cues: List[ForensicEvidence]
    is_speech_active: bool
    latency_ms: float
    context_metadata: Dict[str, Any] = field(default_factory=dict)


class VoiceIntegrityRiskEngine:
    """Real-time streaming multi-layer voice authenticity and risk scoring engine."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_duration_sec: float = 1.0,
        smoothing_alpha: float = 0.35,  # Exponential moving average weight
        suspicious_threshold: float = 0.38,
        critical_threshold: float = 0.68,
        enable_biometrics: bool = True,
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_duration_sec * sample_rate))
        self.smoothing_alpha = smoothing_alpha
        self.suspicious_threshold = suspicious_threshold
        self.critical_threshold = critical_threshold
        self.enable_biometrics = enable_biometrics

        # Sub-analyzers
        self.spectral_analyzer = SpectralAnalyzer(sample_rate=sample_rate)
        self.prosody_analyzer = ProsodyAnalyzer(sample_rate=sample_rate)
        self.vocoder_detector = VocoderFootprintDetector(sample_rate=sample_rate)
        self.biometric_verifier = SpeakerBiometricVerifier(sample_rate=sample_rate) if enable_biometrics else None

        # Streaming buffer state
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.rolling_score = 0.10  # Initial neutral human baseline
        self.total_frames_analyzed = 0
        self.active_speech_frames = 0

    def reset_state(self) -> None:
        """Reset internal streaming buffer and rolling score for a new call session."""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.rolling_score = 0.10
        self.total_frames_analyzed = 0
        self.active_speech_frames = 0

    def _score_prosodic_layer(self, prosody: Dict[str, float]) -> Tuple[float, List[ForensicEvidence]]:
        """Score prosodic & vocal perturbation layer (0.0 to 1.0).
        
        Neural TTS models produce over-smoothed pitch with abnormally low jitter/shimmer,
        or irregular pitch discontinuities.
        """
        score = 0.0
        evidence: List[ForensicEvidence] = []
        
        if prosody.get("has_sufficient_voiced", 0.0) < 0.5:
            # Unvoiced or silence segment
            return 0.15, evidence

        jitter = prosody.get("jitter_local_percent", 1.5)
        shimmer = prosody.get("shimmer_local_percent", 3.5)
        cv_f0 = prosody.get("cv_f0", 0.15)
        hnr = prosody.get("hnr_db", 18.0)

        # 1. Jitter Anomaly: Abnormally smooth (< 0.65%) or chaotic (> 7.0%)
        if jitter < 0.65:
            anomaly_sev = min(1.0, (0.65 - jitter) / 0.50)
            score += 0.50 * anomaly_sev
            evidence.append(ForensicEvidence(
                metric_name="jitter_local_percent",
                observed_value=round(jitter, 3),
                typical_human_range="1.0% - 3.0%",
                contribution_weight=0.50 * anomaly_sev,
                anomaly_description="Unnatural robotic pitch smoothing (synthetic vocoder signature)."
            ))
        elif jitter > 7.0:
            anomaly_sev = min(1.0, (jitter - 7.0) / 6.0)
            score += 0.30 * anomaly_sev
            evidence.append(ForensicEvidence(
                metric_name="jitter_local_percent",
                observed_value=round(jitter, 3),
                typical_human_range="1.0% - 3.0%",
                contribution_weight=0.30 * anomaly_sev,
                anomaly_description="Excessive cycle-to-cycle frequency jitter (vocoder glitching / frame concatenation)."
            ))

        # 2. Shimmer Anomaly: Unnaturally uniform amplitude (< 2.0%)
        if shimmer < 2.0:
            anomaly_sev = min(1.0, (2.0 - shimmer) / 1.5)
            score += 0.40 * anomaly_sev
            evidence.append(ForensicEvidence(
                metric_name="shimmer_local_percent",
                observed_value=round(shimmer, 3),
                typical_human_range="2.5% - 6.0%",
                contribution_weight=0.40 * anomaly_sev,
                anomaly_description="Artificially constant period amplitude (lack of physiological vocal fold tremor)."
            ))

        # 3. Monotonic Pitch Inflection (low coefficient of variation in pitch)
        if cv_f0 < 0.05 and prosody.get("voiced_fraction", 0.0) > 0.4:
            anomaly_sev = min(1.0, (0.05 - cv_f0) / 0.04)
            score += 0.25 * anomaly_sev
            evidence.append(ForensicEvidence(
                metric_name="pitch_coefficient_variation",
                observed_value=round(cv_f0, 4),
                typical_human_range="0.10 - 0.35",
                contribution_weight=0.25 * anomaly_sev,
                anomaly_description="Monotonic pitch contour (flat synthesized prosody)."
            ))

        # 4. HNR Anomaly: Synthetic sterility (> 27 dB)
        if hnr > 27.0:
            score += 0.20
            evidence.append(ForensicEvidence(
                metric_name="harmonics_to_noise_ratio_db",
                observed_value=round(hnr, 1),
                typical_human_range="10 dB - 24 dB",
                contribution_weight=0.20,
                anomaly_description="Hyper-harmonic sterile signal (absence of glottal aerodynamic aspiration noise)."
            ))

        return min(1.0, score), evidence

    def _score_spectral_layer(self, spectral: Dict[str, Any]) -> Tuple[float, List[ForensicEvidence]]:
        """Score spectral & acoustic phase layer (0.0 to 1.0).
        
        Neural vocoders disrupt quadratic phase coupling (Bicoherence)
        and introduce checkerboard artifacts or unnatural high-frequency cutoffs.
        """
        score = 0.0
        evidence: List[ForensicEvidence] = []

        bicoherence = spectral.get("bicoherence_index", 0.5)
        checkerboard = spectral.get("checkerboard_index", 0.0)
        phase_irreg = spectral.get("phase_irregularity", 1.5)
        flatness = spectral.get("spectral_flatness", 0.02)

        # 1. Bicoherence (Phase Coupling): Low bicoherence indicates synthetic phase decoupling
        if bicoherence < 0.22:
            anomaly_sev = min(1.0, (0.22 - bicoherence) / 0.16)
            score += 0.55 * anomaly_sev
            evidence.append(ForensicEvidence(
                metric_name="bicoherence_phase_coupling",
                observed_value=round(bicoherence, 3),
                typical_human_range="0.25 - 0.85",
                contribution_weight=0.55 * anomaly_sev,
                anomaly_description="Acoustic phase decoupling: absence of human vocal tract non-linear quadratic coupling."
            ))

        # 2. Checkerboard Artifact: Transposed convolution ripple
        if checkerboard > 0.30:
            anomaly_sev = min(1.0, (checkerboard - 0.30) / 0.40)
            score += 0.45 * anomaly_sev
            evidence.append(ForensicEvidence(
                metric_name="vocoder_checkerboard_index",
                observed_value=round(checkerboard, 3),
                typical_human_range="0.0 - 0.25",
                contribution_weight=0.45 * anomaly_sev,
                anomaly_description="Periodic spectral ripple from neural vocoder transposed convolutions (HiFi-GAN/MelGAN footprint)."
            ))

        # 3. High-Frequency Phase Incoherence
        if phase_irreg > 2.0:
            score += 0.25
            evidence.append(ForensicEvidence(
                metric_name="phase_irregularity",
                observed_value=round(phase_irreg, 3),
                typical_human_range="0.5 - 1.8",
                contribution_weight=0.25,
                anomaly_description="High-frequency phase discontinuity and dispersion."
            ))

        # 4. Spectral Flatness Divergence (synthetic floor < 0.001)
        if flatness < 0.002:
            score += 0.30
            evidence.append(ForensicEvidence(
                metric_name="spectral_flatness",
                observed_value=round(flatness, 6),
                typical_human_range="0.01 - 0.08",
                contribution_weight=0.30,
                anomaly_description="Unnaturally sterile noise floor (synthetic audio devoid of acoustic ambient room floor)."
            ))

        return min(1.0, score), evidence

    def _score_vocoder_layer(self, vocoder: Dict[str, float]) -> Tuple[float, List[ForensicEvidence]]:
        """Score LPC residual and neural vocoder excitation layer (0.0 to 1.0)."""
        score = 0.0
        evidence: List[ForensicEvidence] = []

        kurtosis = vocoder.get("residual_kurtosis", 6.0)
        hf_harmonicity = vocoder.get("high_freq_harmonicity", 0.1)

        # Natural speech glottal pulses give moderate-high kurtosis (typically 4.0 - 15.0).
        # TTS models with Gaussian latent codes or over-diffused residuals deviate significantly.
        if kurtosis < 3.2:
            score += 0.35
            evidence.append(ForensicEvidence(
                metric_name="lpc_residual_kurtosis",
                observed_value=round(kurtosis, 2),
                typical_human_range="4.5 - 15.0",
                contribution_weight=0.35,
                anomaly_description="Suppressed glottal impulse spikes in LPC residual (synthetic diffuse excitation)."
            ))
        elif kurtosis > 25.0:
            score += 0.25
            evidence.append(ForensicEvidence(
                metric_name="lpc_residual_kurtosis",
                observed_value=round(kurtosis, 2),
                typical_human_range="4.5 - 15.0",
                contribution_weight=0.25,
                anomaly_description="Extreme impulsive residual spikes (synthesized audio concatenation glitch)."
            ))

        # Transposed convolution sub-band periodicity
        if hf_harmonicity > 0.40:
            score += 0.30
            evidence.append(ForensicEvidence(
                metric_name="hf_subband_harmonicity",
                observed_value=round(hf_harmonicity, 3),
                typical_human_range="0.0 - 0.25",
                contribution_weight=0.30,
                anomaly_description="High-frequency periodicity leaking from neural upsampler layers."
            ))

        return min(1.0, score), evidence

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        caller_id: Optional[str] = None,
        transaction_amount_inr: float = 0.0,
        enrolled_speaker_id: Optional[str] = None,
    ) -> FrameEvaluationResult:
        """Ingest a streaming audio chunk (e.g. 200ms - 500ms) and evaluate impersonation risk.
        
        Args:
            audio_chunk: 1D float32 audio chunk
            caller_id: Optional phone number / SIP address
            transaction_amount_inr: Amount of pending banking transaction (if applicable)
            enrolled_speaker_id: If provided, runs cross-session biometric consistency check
        """
        start_time = time.perf_counter()
        
        # Append to sliding audio buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        if len(self.audio_buffer) > self.window_samples:
            self.audio_buffer = self.audio_buffer[-self.window_samples :]

        # Check energy / active speech in buffer
        rms = float(np.sqrt(np.mean(self.audio_buffer ** 2)))
        is_speech = rms >= 0.01

        self.total_frames_analyzed += 1

        if not is_speech:
            # Silence / low background: decay risk score smoothly toward baseline
            self.rolling_score = 0.90 * self.rolling_score + 0.10 * 0.10
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return FrameEvaluationResult(
                timestamp=time.time(),
                frame_risk_score=round(self.rolling_score, 3),
                rolling_risk_score=round(self.rolling_score, 3),
                risk_level=RiskLevel.AUTHENTIC_HUMAN,
                recommended_action=RecommendedAction.PROCEED_NORMAL,
                layer_scores={"prosodic": 0.0, "spectral": 0.0, "vocoder": 0.0, "biometric": 0.0},
                top_forensic_cues=[],
                is_speech_active=False,
                latency_ms=round(elapsed_ms, 2),
                context_metadata={"rms_energy": round(rms, 5), "silence": True}
            )

        self.active_speech_frames += 1

        # Extract features across current analysis window
        window_audio = self.audio_buffer
        prosody_feats = self.prosody_analyzer.analyze_audio_segment(window_audio)
        spectral_feats = self.spectral_analyzer.analyze_audio_segment(window_audio)
        vocoder_feats = self.vocoder_detector.analyze_audio_segment(window_audio)

        # Compute Layer Scores & Forensic Evidence
        prosody_score, prosody_evidence = self._score_prosodic_layer(prosody_feats)
        spectral_score, spectral_evidence = self._score_spectral_layer(spectral_feats)
        vocoder_score, vocoder_evidence = self._score_vocoder_layer(vocoder_feats)

        # Cross-session Biometric Verification Layer (if speaker is enrolled)
        biometric_score = 0.0
        biometric_evidence: List[ForensicEvidence] = []
        bio_info: Dict[str, Any] = {}
        if self.enable_biometrics and self.biometric_verifier and enrolled_speaker_id:
            bio_res = self.biometric_verifier.verify_speaker(enrolled_speaker_id, window_audio)
            bio_info = bio_res
            if bio_res["status"] == "BIOMETRIC_MISMATCH":
                sim = bio_res["similarity"]
                biometric_score = min(1.0, max(0.0, (0.75 - sim) / 0.5))
                biometric_evidence.append(ForensicEvidence(
                    metric_name="biometric_cosine_similarity",
                    observed_value=round(sim, 3),
                    typical_human_range=">= 0.78",
                    contribution_weight=0.45,
                    anomaly_description=f"Speaker identity mismatch against enrolled baseline for '{enrolled_speaker_id}'."
                ))

        # Multi-Layer Weighted Ensemble Fusion
        if enrolled_speaker_id and self.biometric_verifier and bio_info.get("status") != "NOT_ENROLLED":
            # 4-Layer Ensemble
            base_score = (
                0.35 * prosody_score +
                0.30 * spectral_score +
                0.20 * vocoder_score +
                0.15 * biometric_score
            )
        else:
            # 3-Layer Acoustic-DSP Ensemble
            base_score = (
                0.40 * prosody_score +
                0.35 * spectral_score +
                0.25 * vocoder_score
            )

        # Multi-cue corroboration bonus:
        # When 2 or more independent layers detect anomalies (> 0.35), compound risk
        active_layers = [s for s in [prosody_score, spectral_score, vocoder_score] if s >= 0.35]
        if len(active_layers) >= 2:
            raw_frame_score = min(1.0, base_score * 1.30 + 0.08 * len(active_layers))
        else:
            raw_frame_score = base_score

        # Exponential Moving Average (EMA) Temporal Smoothing
        self.rolling_score = (
            self.smoothing_alpha * raw_frame_score +
            (1.0 - self.smoothing_alpha) * self.rolling_score
        )

        # Contextual Risk Escalation (e.g. High-value transaction > 50,000 INR)
        context_adjusted_score = self.rolling_score
        if transaction_amount_inr > 50000:
            # Increase sensitivity threshold for high-value financial transactions
            context_adjusted_score = min(1.0, context_adjusted_score * 1.12)

        # Determine Risk Level
        if context_adjusted_score >= self.critical_threshold:
            risk_level = RiskLevel.CRITICAL_CLONE
            if transaction_amount_inr > 10000:
                recommended_action = RecommendedAction.FREEZE_TRANSACTION
            else:
                recommended_action = RecommendedAction.MANDATORY_CALL_BACK
        elif context_adjusted_score >= self.suspicious_threshold:
            risk_level = RiskLevel.ELEVATED_SUSPICIOUS
            recommended_action = RecommendedAction.STEP_UP_VERIFICATION
        else:
            risk_level = RiskLevel.AUTHENTIC_HUMAN
            recommended_action = RecommendedAction.PROCEED_NORMAL

        # Combine and sort all forensic evidence by contribution weight
        all_evidence = prosody_evidence + spectral_evidence + vocoder_evidence + biometric_evidence
        all_evidence.sort(key=lambda x: x.contribution_weight, reverse=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FrameEvaluationResult(
            timestamp=time.time(),
            frame_risk_score=round(raw_frame_score, 4),
            rolling_risk_score=round(context_adjusted_score, 4),
            risk_level=risk_level,
            recommended_action=recommended_action,
            layer_scores={
                "prosodic": round(prosody_score, 3),
                "spectral": round(spectral_score, 3),
                "vocoder": round(vocoder_score, 3),
                "biometric": round(biometric_score, 3),
            },
            top_forensic_cues=all_evidence[:4],
            is_speech_active=True,
            latency_ms=round(elapsed_ms, 2),
            context_metadata={
                "caller_id": caller_id,
                "transaction_amount_inr": transaction_amount_inr,
                "enrolled_speaker_id": enrolled_speaker_id,
                "biometric_info": bio_info,
            }
        )
