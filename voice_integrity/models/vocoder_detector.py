"""Neural Vocoder and Synthetic Speech Footprint Detector.

Detects artifacts inherent to neural audio synthesis:
- Transposed convolution checkerboard ripples (HiFi-GAN, BigVGAN, MelGAN)
- High-frequency phase incoherence and phase dispersion
- Linear Prediction Coding (LPC) residual excitation anomalies
- Synthetic noise-floor flatness and bandwidth cutoff markers
"""

from __future__ import annotations
import math
from typing import Dict, Tuple
import numpy as np


class VocoderFootprintDetector:
    """Specialized detector for neural vocoder and synthetic speech artifacts."""

    def __init__(self, sample_rate: int = 16000, lpc_order: int = 14):
        self.sample_rate = sample_rate
        self.lpc_order = lpc_order

    def _levinson_durbin(self, r: np.ndarray, order: int) -> np.ndarray:
        """Pure NumPy Levinson-Durbin recursion for Linear Predictive Coding (LPC)."""
        a = np.zeros(order + 1, dtype=np.float32)
        e = r[0]
        a[0] = 1.0

        for i in range(1, order + 1):
            if abs(e) < 1e-12:
                break
            # Reflection coefficient
            lambda_val = -np.sum(a[:i] * r[i:0:-1]) / e
            # Update prediction polynomial
            a[1 : i + 1] = a[1 : i + 1] + lambda_val * a[i - 1 :: -1]
            e = e * (1.0 - lambda_val ** 2)

        return a

    def compute_lpc_residual_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract glottal excitation residual characteristics via LPC inverse filtering.
        
        Human glottal excitation exhibits characteristic impulsive spikes during Glottal
        Closure Instants (GCI) separated by turbulent flow, leading to high kurtosis and
        skewed residual statistics. Neural vocoders produce smoother or synthetic pseudo-random residuals.
        """
        if len(audio) < self.lpc_order * 4:
            return {
                "residual_kurtosis": 3.0,
                "residual_energy_ratio": 0.5,
                "residual_entropy": 1.0,
            }

        # Autocorrelation of audio frame
        n_fft = 2 ** int(np.ceil(np.log2(2 * len(audio))))
        fft_audio = np.fft.rfft(audio, n=n_fft)
        autocorr = np.fft.irfft(fft_audio * np.conj(fft_audio), n=n_fft)[: self.lpc_order + 1]
        
        if autocorr[0] < 1e-9:
            return {
                "residual_kurtosis": 3.0,
                "residual_energy_ratio": 0.5,
                "residual_entropy": 1.0,
            }

        # Calculate LPC coefficients
        a = self._levinson_durbin(autocorr, self.lpc_order)
        
        # Filter audio with LPC polynomial to obtain residual e[n] = x[n] + sum(a_k * x[n-k])
        residual = np.convolve(audio, a, mode="valid")
        if len(residual) < 10:
            return {
                "residual_kurtosis": 3.0,
                "residual_energy_ratio": 0.5,
                "residual_entropy": 1.0,
            }

        # Residual variance and kurtosis
        res_mean = np.mean(residual)
        res_var = np.var(residual) + 1e-12
        res_std = np.sqrt(res_var)
        
        # Kurtosis = E[(x - mu)^4] / sigma^4
        kurtosis = float(np.mean(((residual - res_mean) / res_std) ** 4))
        
        # Ratio of residual energy to original signal energy
        orig_energy = np.mean(audio ** 2) + 1e-12
        energy_ratio = float(np.mean(residual ** 2) / orig_energy)

        # Spectral entropy of residual (uniform noise has entropy close to 1.0)
        res_fft = np.abs(np.fft.rfft(residual)) ** 2
        res_fft_norm = res_fft / (np.sum(res_fft) + 1e-12)
        safe_prob = np.maximum(res_fft_norm, 1e-12)
        entropy = float(-np.sum(safe_prob * np.log2(safe_prob)) / np.log2(len(safe_prob)))

        return {
            "residual_kurtosis": kurtosis,
            "residual_energy_ratio": energy_ratio,
            "residual_entropy": entropy,
        }

    def compute_vocoder_periodicity_signature(self, audio: np.ndarray) -> Dict[str, float]:
        """Detect periodicity artifacts in high-frequency spectral sub-bands.
        
        In neural vocoders with upsampling stages (e.g. strides [8, 8, 2, 2]),
        sub-band correlation patterns frequently leak into the 4000Hz - 8000Hz region.
        """
        n_fft = 512
        hop = 160
        n_frames = 1 + (len(audio) - n_fft) // hop
        if n_frames < 3:
            return {"high_freq_harmonicity": 0.0, "subband_divergence": 0.0}

        frames = np.zeros((n_frames, n_fft), dtype=np.float32)
        window = np.hanning(n_fft)
        for i in range(n_frames):
            frames[i] = audio[i * hop : i * hop + n_fft] * window

        mag = np.abs(np.fft.rfft(frames, axis=1)).T  # (n_freqs, n_frames)
        freqs = np.linspace(0, self.sample_rate / 2.0, mag.shape[0])

        # Sub-band 1: Voice formants (300 - 3400 Hz)
        # Sub-band 2: High frequency (3400 - 8000 Hz)
        band1 = (freqs >= 300) & (freqs < 3400)
        band2 = freqs >= 3400

        energy1 = np.mean(mag[band1, :] ** 2) + 1e-12
        energy2 = np.mean(mag[band2, :] ** 2) + 1e-12
        
        # Subband divergence ratio
        divergence = float(np.abs(np.log10(energy2 / energy1)))

        # Autocorrelation of high-frequency sub-band along the frequency axis
        # (Transposed convolutions create periodic peaks along the frequency spectrum)
        hf_spectrum = np.mean(mag[band2, :], axis=1)
        if len(hf_spectrum) > 8:
            norm_hf = hf_spectrum - np.mean(hf_spectrum)
            corr_hf = np.correlate(norm_hf, norm_hf, mode="full")
            corr_hf = corr_hf[len(corr_hf) // 2 :]
            # Exclude lag 0
            if corr_hf[0] > 1e-8:
                norm_corr = corr_hf / corr_hf[0]
                # Maximum peak at non-zero lag
                secondary_peak = float(np.max(norm_corr[2:])) if len(norm_corr) > 2 else 0.0
            else:
                secondary_peak = 0.0
        else:
            secondary_peak = 0.0

        return {
            "high_freq_harmonicity": secondary_peak,
            "subband_divergence": divergence,
        }

    def analyze_audio_segment(self, audio: np.ndarray) -> Dict[str, float]:
        """Run full neural vocoder footprint extraction."""
        lpc_feats = self.compute_lpc_residual_features(audio)
        periodicity = self.compute_vocoder_periodicity_signature(audio)
        return {**lpc_feats, **periodicity}
