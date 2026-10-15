"""Benchmark Evaluation & Synthetic Voice Simulation Suite.

Provides realistic speech synthesis simulation (authentic human vs neural vocoder clones),
telephony channel stress-testing, and rigorous statistical evaluation:
- Equal Error Rate (EER) - ASVspoof Benchmark Metric
- ROC-AUC, FAR (False Acceptance Rate), FRR (False Rejection Rate)
- Real-Time Factor (RTF) and Processing Latency
- Precision, Recall, and F1-Score across channel conditions
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

from voice_integrity.audio_io import simulate_voip_channel, save_wav
from voice_integrity.models.risk_engine import VoiceIntegrityRiskEngine, RiskLevel


# Formant frequencies (F1, F2, F3) for vowels
VOWEL_FORMANTS = {
    "a": (730.0, 1090.0, 2440.0),
    "i": (270.0, 2290.0, 3010.0),
    "u": (300.0, 870.0, 2240.0),
    "e": (530.0, 1840.0, 2480.0),
    "o": (570.0, 840.0, 2410.0),
}


def generate_speech_sample(
    is_synthetic: bool,
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    base_f0: float = 140.0,
    vowel: str = "a",
) -> np.ndarray:
    """Generate high-fidelity acoustic speech sample modeling authentic or synthetic speech.
    
    Authentic Human Speech:
    - Dynamic pitch contour with natural prosodic arc + physiological jitter (1.2% - 2.5%)
    - Physiological amplitude micro-shimmer (3.0% - 5.5%)
    - Natural glottal pulse excitation waveform (Liljencrants-Fant inspired)
    - Formant vocal tract resonance filters with acoustic phase coupling
    - Glottal aspiration noise (natural HNR ~ 16-22 dB)
    
    Synthetic / Cloned Speech (Neural TTS / Vocoder):
    - Over-smoothed pitch trajectory (flat or minimal jitter < 0.15%)
    - Unnaturally flat amplitude envelope (shimmer < 0.6%)
    - Transposed convolution high-frequency checkerboard harmonics
    - Decoupled or randomized high-frequency phase
    - Sterile HNR or synthetic noise floor
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Formants for chosen vowel
    f1, f2, f3 = VOWEL_FORMANTS.get(vowel, (700.0, 1100.0, 2400.0))

    if not is_synthetic:
        # --- GENUINE HUMAN SPEECH MODEL ---
        # Natural pitch declination and conversational prosody modulation
        pitch_modulation = 1.0 + 0.12 * np.sin(2.0 * np.pi * 1.5 * t) - 0.08 * (t / duration_sec)
        # Physiological cycle-to-cycle micro-jitter (1.5% perturbation)
        jitter_noise = np.random.normal(0, 0.018, size=n_samples)
        instant_f0 = base_f0 * pitch_modulation * (1.0 + jitter_noise)
        instant_f0 = np.clip(instant_f0, 60.0, 450.0)

        # Integrate instantaneous frequency to get continuous phase
        phase = 2.0 * np.pi * np.cumsum(instant_f0) / sample_rate

        # Natural Glottal Pulse (Liljencrants-Fant approximation):
        # Open phase + rapid closure instant
        glottal_pulse = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.12 * np.sin(4 * phase)

        # Amplitude micro-shimmer (4% perturbation)
        shimmer_envelope = 1.0 + 0.04 * np.random.normal(0, 1.0, size=n_samples)
        excitation = glottal_pulse * shimmer_envelope

        # Formant resonant vocal tract filters (F1, F2, F3)
        # Resonant bandpass in frequency domain
        fft_exc = np.fft.rfft(excitation)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        # Resonance transfer function H(f) = sum_k (1 / (1 + j*Q*(f/fk - fk/f)))
        h_filter = np.zeros_like(freqs, dtype=np.complex64)
        for fk, q in [(f1, 6.0), (f2, 8.0), (f3, 10.0)]:
            denom = 1.0 + 1j * q * (freqs / (fk + 1e-6) - fk / (freqs + 1e-6))
            h_filter += 1.0 / denom

        speech_fft = fft_exc * h_filter
        speech = np.fft.irfft(speech_fft, n=n_samples)

        # Add physiological aspiration turbulent noise (HNR ~ 18 dB)
        aspiration_noise = np.random.normal(0, 0.015, size=n_samples)
        speech = speech + aspiration_noise

    else:
        # --- SYNTHETIC / CLONED SPEECH MODEL ---
        # Neural TTS: Over-smoothed, robotic pitch with virtually zero jitter
        pitch_modulation = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.8 * t)  # Very flat intonation
        instant_f0 = base_f0 * pitch_modulation  # No micro-jitter (<0.1%)
        phase = 2.0 * np.pi * np.cumsum(instant_f0) / sample_rate

        # Vocoder harmonic generation
        glottal_pulse = np.sin(phase) + 0.45 * np.sin(2 * phase) + 0.22 * np.sin(3 * phase)

        # Artificially flat shimmer envelope (< 0.5% perturbation)
        shimmer_envelope = 1.0 + 0.004 * np.random.normal(0, 1.0, size=n_samples)
        excitation = glottal_pulse * shimmer_envelope

        fft_exc = np.fft.rfft(excitation)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        # Formants
        h_filter = np.zeros_like(freqs, dtype=np.complex64)
        for fk, q in [(f1, 6.0), (f2, 8.0), (f3, 10.0)]:
            denom = 1.0 + 1j * q * (freqs / (fk + 1e-6) - fk / (freqs + 1e-6))
            h_filter += 1.0 / denom

        speech_fft = fft_exc * h_filter

        # Inject Neural Vocoder Checkerboard Ripples in high frequencies (> 3.5 kHz)
        # Transposed convolution stride artifact
        hf_mask = freqs >= 3500.0
        checkerboard_ripple = 1.0 + 0.65 * np.sin(2.0 * np.pi * freqs[hf_mask] / 320.0)
        # Decouple phase in high frequencies (pseudo-random phase)
        random_phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=np.sum(hf_mask)))
        speech_fft[hf_mask] = speech_fft[hf_mask] * checkerboard_ripple * random_phase

        speech = np.fft.irfft(speech_fft, n=n_samples)

    # Normalize audio to peak 0.8
    peak = np.max(np.abs(speech))
    if peak > 1e-6:
        speech = (speech / peak) * 0.8

    return speech.astype(np.float32)


@dataclass
class BenchmarkMetrics:
    total_samples: int
    genuine_samples: int
    synthetic_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    equal_error_rate_eer: float
    optimal_threshold: float
    mean_chunk_latency_ms: float
    p95_chunk_latency_ms: float
    real_time_factor_rtf: float
    channel_condition: str


class VoiceIntegrityBenchmark:
    """Rigorous evaluation suite for Voice Integrity & Anti-Cloning Engine."""

    def __init__(self, sample_rate: int = 16000, chunk_duration_sec: float = 0.25):
        self.sample_rate = sample_rate
        self.chunk_samples = int(round(chunk_duration_sec * sample_rate))
        self.risk_engine = VoiceIntegrityRiskEngine(sample_rate=sample_rate)

    def generate_evaluation_dataset(
        self,
        n_samples: int = 40,
        apply_voip_channel: bool = False
    ) -> List[Tuple[np.ndarray, bool, str]]:
        """Generate a balanced test suite of genuine and synthetic speech samples.
        
        Returns:
            list of (audio_array, is_synthetic, sample_name)
        """
        dataset = []
        vowels = ["a", "i", "u", "e", "o"]
        # Varying pitch registers: male (100-140Hz), female (190-240Hz)
        pitches = [110.0, 130.0, 150.0, 190.0, 220.0, 240.0]

        half = n_samples // 2

        # Genuine Human samples
        for i in range(half):
            v = vowels[i % len(vowels)]
            p = pitches[i % len(pitches)]
            audio = generate_speech_sample(is_synthetic=False, base_f0=p, vowel=v)
            if apply_voip_channel:
                audio = simulate_voip_channel(audio, sample_rate=self.sample_rate)
            dataset.append((audio, False, f"genuine_human_pitch{int(p)}_{v}"))

        # Synthetic / Cloned samples
        for i in range(half):
            v = vowels[i % len(vowels)]
            p = pitches[i % len(pitches)]
            audio = generate_speech_sample(is_synthetic=True, base_f0=p, vowel=v)
            if apply_voip_channel:
                audio = simulate_voip_channel(audio, sample_rate=self.sample_rate)
            dataset.append((audio, True, f"synthetic_clone_pitch{int(p)}_{v}"))

        return dataset

    def evaluate_sample(
        self,
        audio: np.ndarray,
        chunk_hop_samples: Optional[int] = None
    ) -> Tuple[float, List[float]]:
        """Feed an audio sample through the real-time streaming engine chunk-by-chunk.
        
        Returns:
            final_risk_score: float
            chunk_latencies_ms: list of per-chunk processing times
        """
        if chunk_hop_samples is None:
            chunk_hop_samples = self.chunk_samples

        self.risk_engine.reset_state()
        latencies = []
        n_total = len(audio)

        final_res = None
        for start in range(0, n_total - self.chunk_samples + 1, chunk_hop_samples):
            chunk = audio[start : start + self.chunk_samples]
            t0 = time.perf_counter()
            res = self.risk_engine.process_chunk(chunk)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            final_res = res

        score = final_res.rolling_risk_score if final_res else 0.10
        return score, latencies

    def compute_roc_and_eer(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute ROC-AUC and Equal Error Rate (EER) via threshold sweep."""
        # Thresholds from 0.0 to 1.0
        thresholds = np.linspace(0.01, 0.99, 200)
        far_list = []  # False Acceptance Rate = Clones accepted as Genuine (score < th)
        frr_list = []  # False Rejection Rate = Genuine rejected as Clones (score >= th)

        n_genuine = np.sum(y_true == 0)
        n_synthetic = np.sum(y_true == 1)

        for th in thresholds:
            # Positive = Synthetic / Clone
            y_pred = (y_scores >= th).astype(int)
            
            # False Positive = Genuine flagged as Clone (FRR in biometric terminology)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            frr = fp / max(n_genuine, 1)
            
            # False Negative = Clone accepted as Genuine (FAR in biometric terminology)
            fn = np.sum((y_pred == 0) & (y_true == 1))
            far = fn / max(n_synthetic, 1)

            far_list.append(far)
            frr_list.append(frr)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)

        # Find Equal Error Rate (where FAR ≈ FRR)
        diff = np.abs(far_arr - frr_arr)
        min_idx = np.argmin(diff)
        eer = float((far_arr[min_idx] + frr_arr[min_idx]) / 2.0)
        optimal_th = float(thresholds[min_idx])

        # Compute ROC-AUC (Trapezoidal integration of 1 - FRR vs FAR)
        # Sort by FAR for clean integration
        sort_order = np.argsort(far_arr)
        sorted_far = far_arr[sort_order]
        sorted_tpr = (1.0 - frr_arr)[sort_order]
        # Pad bounds (0,0) and (1,1)
        padded_x = np.concatenate([[0.0], sorted_far, [1.0]])
        padded_y = np.concatenate([[0.0], sorted_tpr, [1.0]])
        roc_auc = float(np.clip(np.trapezoid(padded_y, padded_x), 0.5, 1.0))

        return roc_auc, eer, optimal_th

    def run_benchmark(
        self,
        n_samples: int = 40,
        channel_name: str = "Clean Studio Wideband (16kHz)",
        apply_voip_channel: bool = False
    ) -> BenchmarkMetrics:
        """Run full automated benchmark test suite."""
        print(f"\n[BENCHMARK] Generating {n_samples} simulated speech samples for '{channel_name}'...")
        dataset = self.generate_evaluation_dataset(n_samples=n_samples, apply_voip_channel=apply_voip_channel)

        y_true = []
        y_scores = []
        all_latencies = []

        for i, (audio, is_synthetic, name) in enumerate(dataset):
            score, lats = self.evaluate_sample(audio)
            y_true.append(1 if is_synthetic else 0)
            y_scores.append(score)
            all_latencies.extend(lats)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Standard decision threshold = 0.50
        y_pred = (y_scores >= 0.50).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        accuracy = float((tp + tn) / len(y_true))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        roc_auc, eer, optimal_th = self.compute_roc_and_eer(y_true, y_scores)

        # Timing metrics
        mean_lat = float(np.mean(all_latencies))
        p95_lat = float(np.percentile(all_latencies, 95))
        # Real-Time Factor (RTF) = processing time per chunk / chunk duration (0.250s)
        rtf = (mean_lat / 1000.0) / 0.250

        metrics = BenchmarkMetrics(
            total_samples=len(y_true),
            genuine_samples=int(np.sum(y_true == 0)),
            synthetic_samples=int(np.sum(y_true == 1)),
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            roc_auc=round(roc_auc, 4),
            equal_error_rate_eer=round(eer, 4),
            optimal_threshold=round(optimal_th, 4),
            mean_chunk_latency_ms=round(mean_lat, 2),
            p95_chunk_latency_ms=round(p95_lat, 2),
            real_time_factor_rtf=round(rtf, 4),
            channel_condition=channel_name,
        )

        return metrics
