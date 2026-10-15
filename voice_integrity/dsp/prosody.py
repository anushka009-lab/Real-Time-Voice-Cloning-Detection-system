"""Prosody and Behavioral Analysis Module.

Extracts discriminative biomechanical and prosodic indicators:
- Fundamental Frequency (F0) tracking with sub-sample peak refinement
- Cycle-to-cycle Jitter metrics (Local, RAP, PPQ5)
- Cycle-to-cycle Shimmer metrics (Local, dB, APQ3, APQ5)
- Harmonics-to-Noise Ratio (HNR)
- Pitch dynamics, inflection variance, and pause micro-distribution
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


class ProsodyAnalyzer:
    """Biomechanical vocal fold perturbation and prosodic dynamics engine."""

    def __init__(
        self,
        sample_rate: int = 16000,
        f0_min: float = 65.0,    # Deep male voice lower bound
        f0_max: float = 500.0,   # High female / child voice upper bound
        frame_length_ms: float = 35.0,  # 35ms frame to encompass at least 2 pitch periods at 65Hz
        hop_length_ms: float = 10.0,    # 10ms frame hop
    ):
        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.frame_len = int(round(frame_length_ms * sample_rate / 1000.0))
        self.hop_len = int(round(hop_length_ms * sample_rate / 1000.0))
        
        # Lag bounds corresponding to pitch search range
        self.min_lag = int(round(sample_rate / f0_max))
        self.max_lag = int(round(sample_rate / f0_min))

    def extract_f0_contour(
        self,
        audio: np.ndarray,
        voiced_threshold: float = 0.35
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract continuous F0 pitch contour, voiced flags, and frame amplitudes.
        
        Uses Normalized Cross-Correlation Function (NCCF) with parabolic peak interpolation
        for sub-sample period accuracy.
        
        Returns:
            f0_contour: (N,) F0 in Hz (0.0 for unvoiced)
            voiced_mask: (N,) boolean mask
            amplitudes: (N,) RMS frame amplitude
        """
        n_samples = len(audio)
        if n_samples < self.frame_len:
            return np.array([]), np.array([], dtype=bool), np.array([])
            
        n_frames = 1 + (n_samples - self.frame_len) // self.hop_len
        f0_contour = np.zeros(n_frames, dtype=np.float32)
        voiced_mask = np.zeros(n_frames, dtype=bool)
        amplitudes = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_len
            frame = audio[start : start + self.frame_len]
            
            # Frame energy
            rms = float(np.sqrt(np.mean(frame ** 2)))
            amplitudes[i] = rms
            
            # Skip near-silent frames
            if rms < 0.005:
                continue

            # Remove DC bias
            frame_centered = frame - np.mean(frame)
            norm_frame = np.sum(frame_centered ** 2)
            if norm_frame < 1e-8:
                continue

            # Autocorrelation via FFT
            n_fft = 2 ** int(np.ceil(np.log2(2 * self.frame_len)))
            fft_frame = np.fft.rfft(frame_centered, n=n_fft)
            corr = np.fft.irfft(fft_frame * np.conj(fft_frame), n=n_fft)[: self.frame_len]

            # Normalize correlation
            r0 = corr[0] if corr[0] > 1e-12 else 1.0
            nccf = corr / r0
            
            # Find peak within [min_lag, max_lag]
            search_region = nccf[self.min_lag : min(self.max_lag, len(nccf))]
            if len(search_region) == 0:
                continue

            rel_peak = np.argmax(search_region)
            peak_val = search_region[rel_peak]
            lag = self.min_lag + rel_peak

            if peak_val >= voiced_threshold:
                # Sub-sample refinement via parabolic interpolation
                if 0 < lag < len(nccf) - 1:
                    alpha = nccf[lag - 1]
                    beta = nccf[lag]
                    gamma = nccf[lag + 1]
                    denom = 2 * (2 * beta - alpha - gamma)
                    if abs(denom) > 1e-8:
                        delta = (alpha - gamma) / denom
                        refined_lag = lag + delta
                    else:
                        refined_lag = float(lag)
                else:
                    refined_lag = float(lag)

                f0 = self.sample_rate / max(refined_lag, 1.0)
                if self.f0_min <= f0 <= self.f0_max:
                    f0_contour[i] = float(f0)
                    voiced_mask[i] = True

        return f0_contour, voiced_mask, amplitudes

    def compute_jitter_metrics(self, f0_contour: np.ndarray, voiced_mask: np.ndarray) -> Dict[str, float]:
        """Compute cycle-to-cycle frequency perturbations (Jitter).
        
        Natural human speech has inherent neuromuscular micro-tremors (Jitter 0.5% - 3.0%).
        Neural speech synthesis models (VITS, HiFi-GAN, FastSpeech) produce over-smoothed
        pitch contours with abnormally low jitter (< 0.25%) or abrupt unnatural jumps.
        """
        voiced_f0 = f0_contour[voiced_mask]
        if len(voiced_f0) < 5:
            return {
                "jitter_local_percent": 0.0,
                "jitter_rap_percent": 0.0,
                "jitter_ppq5_percent": 0.0,
                "has_sufficient_voiced": 0.0,
            }

        # Pitch periods in seconds T_i = 1 / F0_i
        periods = 1.0 / voiced_f0
        n_periods = len(periods)
        mean_period = np.mean(periods)

        if mean_period < 1e-6:
            return {
                "jitter_local_percent": 0.0,
                "jitter_rap_percent": 0.0,
                "jitter_ppq5_percent": 0.0,
                "has_sufficient_voiced": 0.0,
            }

        # 1. Jitter (local) = sum(|T_i - T_{i+1}|) / ((N - 1) * mean_T) * 100%
        abs_diffs = np.abs(np.diff(periods))
        jitter_local = float((np.mean(abs_diffs) / mean_period) * 100.0)

        # 2. Jitter (RAP - Relative Average Perturbation) = 3-point moving average
        rap_diffs = []
        for i in range(1, n_periods - 1):
            local_avg = (periods[i - 1] + periods[i] + periods[i + 1]) / 3.0
            rap_diffs.append(abs(periods[i] - local_avg))
        jitter_rap = float((np.mean(rap_diffs) / mean_period) * 100.0) if rap_diffs else 0.0

        # 3. Jitter (PPQ5 - Five-point Period Perturbation Quotient)
        ppq5_diffs = []
        for i in range(2, n_periods - 2):
            local_avg5 = np.mean(periods[i - 2 : i + 3])
            ppq5_diffs.append(abs(periods[i] - local_avg5))
        jitter_ppq5 = float((np.mean(ppq5_diffs) / mean_period) * 100.0) if ppq5_diffs else 0.0

        return {
            "jitter_local_percent": jitter_local,
            "jitter_rap_percent": jitter_rap,
            "jitter_ppq5_percent": jitter_ppq5,
            "has_sufficient_voiced": 1.0,
        }

    def compute_shimmer_metrics(
        self,
        amplitudes: np.ndarray,
        voiced_mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute cycle-to-cycle amplitude perturbations (Shimmer).
        
        Genuine human voice contains physiological amplitude micro-variation (Shimmer 2.0% - 6.0%).
        Neural TTS often exhibits unnaturally consistent or synthetic flat amplitude envelope.
        """
        voiced_amps = amplitudes[voiced_mask]
        if len(voiced_amps) < 5:
            return {
                "shimmer_local_percent": 0.0,
                "shimmer_db": 0.0,
                "shimmer_apq3_percent": 0.0,
                "shimmer_apq5_percent": 0.0,
            }

        mean_amp = np.mean(voiced_amps)
        n_amps = len(voiced_amps)
        if mean_amp < 1e-6:
            return {
                "shimmer_local_percent": 0.0,
                "shimmer_db": 0.0,
                "shimmer_apq3_percent": 0.0,
                "shimmer_apq5_percent": 0.0,
            }

        # 1. Shimmer (local) = sum(|A_i - A_{i+1}|) / ((N-1) * mean_A) * 100%
        abs_diffs = np.abs(np.diff(voiced_amps))
        shimmer_local = float((np.mean(abs_diffs) / mean_amp) * 100.0)

        # 2. Shimmer (dB) = mean(|20 * log10(A_{i+1} / A_i)|)
        # Avoid divide-by-zero
        safe_ratios = np.maximum(voiced_amps[1:] / np.maximum(voiced_amps[:-1], 1e-8), 1e-5)
        shimmer_db = float(np.mean(np.abs(20.0 * np.log10(safe_ratios))))

        # 3. Shimmer (APQ3 - 3-point Amplitude Perturbation Quotient)
        apq3_diffs = []
        for i in range(1, n_amps - 1):
            local_avg3 = (voiced_amps[i - 1] + voiced_amps[i] + voiced_amps[i + 1]) / 3.0
            apq3_diffs.append(abs(voiced_amps[i] - local_avg3))
        shimmer_apq3 = float((np.mean(apq3_diffs) / mean_amp) * 100.0) if apq3_diffs else 0.0

        # 4. Shimmer (APQ5 - 5-point Amplitude Perturbation Quotient)
        apq5_diffs = []
        for i in range(2, n_amps - 2):
            local_avg5 = np.mean(voiced_amps[i - 2 : i + 3])
            apq5_diffs.append(abs(voiced_amps[i] - local_avg5))
        shimmer_apq5 = float((np.mean(apq5_diffs) / mean_amp) * 100.0) if apq5_diffs else 0.0

        return {
            "shimmer_local_percent": shimmer_local,
            "shimmer_db": shimmer_db,
            "shimmer_apq3_percent": shimmer_apq3,
            "shimmer_apq5_percent": shimmer_apq5,
        }

    def compute_hnr(self, audio: np.ndarray, f0_contour: np.ndarray, voiced_mask: np.ndarray) -> float:
        """Compute average Harmonics-to-Noise Ratio (HNR in dB) for voiced segments.
        
        Uses autocorrelation peak-to-noise ratio formulation.
        Human speech typically achieves 12 to 24 dB.
        Over-smooth TTS speech often exceeds 28 dB or has flat artificial noise.
        """
        voiced_indices = np.where(voiced_mask)[0]
        if len(voiced_indices) < 2:
            return 15.0  # Default neutral HNR

        hnr_values = []
        for idx in voiced_indices:
            start = idx * self.hop_len
            frame = audio[start : start + self.frame_len]
            if len(frame) < self.frame_len:
                continue
            frame = frame - np.mean(frame)
            total_energy = np.sum(frame ** 2)
            if total_energy < 1e-8:
                continue

            # Normalized Autocorrelation
            n_fft = 2 ** int(np.ceil(np.log2(2 * self.frame_len)))
            fft_frame = np.fft.rfft(frame, n=n_fft)
            corr = np.fft.irfft(fft_frame * np.conj(fft_frame), n=n_fft)[: self.frame_len]
            r0 = corr[0]
            if r0 < 1e-8:
                continue

            f0 = f0_contour[idx]
            if f0 <= 0:
                continue
            period_lag = int(round(self.sample_rate / f0))
            if 0 < period_lag < len(corr):
                r_max = corr[period_lag]
                # Normalized correlation rx = r_max / r0
                rx = float(r_max / r0)
                if 0.01 < rx < 0.999:
                    # HNR = 10 * log10(rx / (1 - rx))
                    hnr_db = 10.0 * np.log10(rx / (1.0 - rx))
                    hnr_values.append(hnr_db)

        if not hnr_values:
            return 15.0
        return float(np.median(hnr_values))

    def compute_prosodic_dynamics(
        self,
        f0_contour: np.ndarray,
        voiced_mask: np.ndarray,
        amplitudes: np.ndarray
    ) -> Dict[str, float]:
        """Analyze higher-level speech cadence: pitch variation, voiced fraction, rhythm."""
        n_total = len(f0_contour)
        if n_total == 0:
            return {
                "mean_f0": 0.0,
                "std_f0": 0.0,
                "cv_f0": 0.0,
                "voiced_fraction": 0.0,
                "pause_ratio": 0.0,
            }

        voiced_f0 = f0_contour[voiced_mask]
        voiced_count = len(voiced_f0)
        voiced_fraction = voiced_count / n_total

        if voiced_count > 0:
            mean_f0 = float(np.mean(voiced_f0))
            std_f0 = float(np.std(voiced_f0))
            cv_f0 = float(std_f0 / (mean_f0 + 1e-8))  # Coefficient of variation
        else:
            mean_f0, std_f0, cv_f0 = 0.0, 0.0, 0.0

        # Silence / pause frames (amplitude below noise floor threshold)
        pause_frames = np.sum(amplitudes < 0.01)
        pause_ratio = float(pause_frames / n_total)

        return {
            "mean_f0": mean_f0,
            "std_f0": std_f0,
            "cv_f0": cv_f0,
            "voiced_fraction": voiced_fraction,
            "pause_ratio": pause_ratio,
        }

    def analyze_audio_segment(self, audio: np.ndarray) -> Dict[str, float]:
        """Comprehensive prosody analysis pipeline."""
        f0_contour, voiced_mask, amplitudes = self.extract_f0_contour(audio)
        jitter = self.compute_jitter_metrics(f0_contour, voiced_mask)
        shimmer = self.compute_shimmer_metrics(amplitudes, voiced_mask)
        hnr = self.compute_hnr(audio, f0_contour, voiced_mask)
        dynamics = self.compute_prosodic_dynamics(f0_contour, voiced_mask, amplitudes)

        return {
            **jitter,
            **shimmer,
            "hnr_db": hnr,
            **dynamics,
        }
