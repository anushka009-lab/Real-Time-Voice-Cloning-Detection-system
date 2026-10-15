"""Acoustic and Spectral DSP Feature Extraction Module.

Extracts discriminative features for detecting AI-generated or cloned voices:
- Short-Time Fourier Transform (STFT) & Phase Spectrograms
- Linear Frequency Cepstral Coefficients (LFCC) - ASVspoof Benchmark Standard
- Higher-Order Spectral Analysis (Bispectrum & Phase Coupling Index)
- Neural Vocoder Artifact Metric (High-Frequency Checkerboard / Cutoff Analysis)
- Classical Spectral Descriptors (Flux, Roll-off, Centroid, Flatness)
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional
import numpy as np


class SpectralAnalyzer:
    """Multi-layer spectral and acoustic analysis engine."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,  # 10ms hop at 16kHz
        win_length: int = 400,  # 25ms window at 16kHz
        n_lfcc: int = 20,
        n_filters: int = 32,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_lfcc = n_lfcc
        self.n_filters = n_filters
        
        # Precompute window
        self.window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_length) / (win_length - 1)))
        
        # Precompute Linear Filterbank for LFCC
        self.linear_filterbank = self._build_linear_filterbank()
        
        # Precompute DCT-II matrix for LFCC
        self.dct_matrix = self._build_dct_matrix()

    def _build_linear_filterbank(self) -> np.ndarray:
        """Construct triangular filterbank with linearly spaced centers across [0, Nyquist].
        
        Unlike Mel filterbanks (which compress high frequencies for perception),
        linear filterbanks retain fine resolution in higher harmonics where neural vocoder
        (HiFi-GAN, WaveGlow, Diffusion) synthesis artifacts predominantly reside.
        """
        n_freqs = self.n_fft // 2 + 1
        freqs = np.linspace(0, self.sample_rate / 2.0, n_freqs)
        
        # Linear filter center frequencies
        f_min = 0.0
        f_max = self.sample_rate / 2.0
        center_freqs = np.linspace(f_min, f_max, self.n_filters + 2)
        
        bank = np.zeros((self.n_filters, n_freqs), dtype=np.float32)
        for i in range(self.n_filters):
            f_left = center_freqs[i]
            f_center = center_freqs[i + 1]
            f_right = center_freqs[i + 2]
            
            # Rising slope
            rising = (freqs >= f_left) & (freqs <= f_center)
            if f_center > f_left:
                bank[i, rising] = (freqs[rising] - f_left) / (f_center - f_left)
                
            # Falling slope
            falling = (freqs > f_center) & (freqs <= f_right)
            if f_right > f_center:
                bank[i, falling] = (f_right - freqs[falling]) / (f_right - f_center)
                
        return bank

    def _build_dct_matrix(self) -> np.ndarray:
        """Type-II Discrete Cosine Transform basis matrix."""
        dct = np.zeros((self.n_lfcc, self.n_filters), dtype=np.float32)
        for k in range(self.n_lfcc):
            for n in range(self.n_filters):
                dct[k, n] = np.cos(np.pi * k * (n + 0.5) / self.n_filters)
        dct[0, :] *= 1.0 / np.sqrt(self.n_filters)
        dct[1:, :] *= np.sqrt(2.0 / self.n_filters)
        return dct

    def compute_stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute complex STFT, magnitude, and phase matrices.
        
        Returns:
            mag: (n_freqs, n_frames) magnitude spectrum
            phase: (n_freqs, n_frames) unwrapped phase spectrum
            complex_spec: (n_freqs, n_frames) complex STFT
        """
        if len(audio) < self.win_length:
            audio = np.pad(audio, (0, self.win_length - len(audio)))
            
        n_frames = 1 + (len(audio) - self.win_length) // self.hop_length
        if n_frames <= 0:
            n_frames = 1
            
        frames = np.zeros((n_frames, self.n_fft), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            segment = audio[start : start + self.win_length]
            if len(segment) < self.win_length:
                segment = np.pad(segment, (0, self.win_length - len(segment)))
            frames[i, :self.win_length] = segment * self.window

        complex_spec = np.fft.rfft(frames, n=self.n_fft, axis=1).T  # (n_freqs, n_frames)
        mag = np.abs(complex_spec)
        phase = np.angle(complex_spec)
        return mag, phase, complex_spec

    def compute_lfcc(self, mag_spec: np.ndarray) -> np.ndarray:
        """Compute Linear Frequency Cepstral Coefficients (LFCC) from magnitude spectrum.
        
        Returns:
            lfcc: (n_lfcc, n_frames)
        """
        power_spec = (mag_spec ** 2) / self.n_fft
        filter_energies = np.dot(self.linear_filterbank, power_spec)  # (n_filters, n_frames)
        # Numerical stability floor
        log_energies = np.log(np.maximum(filter_energies, 1e-12))
        lfcc = np.dot(self.dct_matrix, log_energies)  # (n_lfcc, n_frames)
        return lfcc

    def compute_bispectrum_coupling(self, complex_spec: np.ndarray, f0_hint: Optional[float] = None) -> float:
        """Calculate Quadratic Phase Coupling (Harmonic Bicoherence) Index.
        
        Biological vocal tracts produce nonlinear quadratic phase coupling between
        fundamental harmonics f1, f2, and f1+f2.
        Neural vocoders synthesize audio with decoupled or pseudo-random high harmonic phases.
        
        Returns:
            bicoherence_metric: float (0.0 to 1.0)
            Higher value -> authentic human nonlinear vocal tract coupling.
            Lower value -> synthetic speech phase decoupling.
        """
        n_freqs, n_frames = complex_spec.shape
        if n_frames < 4 or n_freqs < 16:
            return 0.5  # Neutral default for short segments

        # Estimate fundamental bin if not given
        if f0_hint and f0_hint > 60.0:
            bin_f0 = int(round(f0_hint * self.n_fft / self.sample_rate))
        else:
            # Locate first spectral peak in 75-350 Hz range
            min_bin = int(round(75.0 * self.n_fft / self.sample_rate))
            max_bin = int(round(350.0 * self.n_fft / self.sample_rate))
            avg_mag = np.mean(np.abs(complex_spec[min_bin:max_bin, :]), axis=1)
            bin_f0 = min_bin + int(np.argmax(avg_mag)) if len(avg_mag) > 0 else 4

        bin_f0 = max(2, min(bin_f0, n_freqs // 6))

        # Evaluate harmonic triplets: k1 * f0 + k2 * f0 = (k1 + k2) * f0
        bicoherence_sum = 0.0
        count = 0
        for k1 in range(1, 5):
            for k2 in range(1, 5):
                k3 = k1 + k2
                b1 = k1 * bin_f0
                b2 = k2 * bin_f0
                b3 = k3 * bin_f0
                if b3 >= n_freqs:
                    continue

                x1 = complex_spec[b1, :]
                x2 = complex_spec[b2, :]
                x3 = complex_spec[b3, :]

                b123 = x1 * x2 * np.conj(x3)
                mean_b = np.abs(np.mean(b123)) ** 2
                norm_denom = np.mean(np.abs(x1 * x2) ** 2) * np.mean(np.abs(x3) ** 2)

                if norm_denom > 1e-12:
                    bic = mean_b / norm_denom
                    bicoherence_sum += float(np.clip(bic, 0.0, 1.0))
                    count += 1

        if count == 0:
            return 0.5
        avg_bic = bicoherence_sum / count
        return float(np.clip(avg_bic, 0.0, 1.0))

    def compute_vocoder_artifacts(
        self,
        mag_spec: np.ndarray,
        phase_spec: np.ndarray
    ) -> Dict[str, float]:
        """Detect neural vocoder spectral artifacts:
        1. Checkerboard / periodic ripple artifact in high frequencies (transposed convolutions).
        2. High-frequency phase divergence & incoherence.
        3. Spectral cutoff anomaly (sharp drop-off at model cutoff e.g. 11kHz / 7.5kHz).
        4. Spectral Flatness in high-frequency bands.
        """
        n_freqs, n_frames = mag_spec.shape
        freqs = np.linspace(0, self.sample_rate / 2.0, n_freqs)
        
        # 1. High-frequency phase incoherence
        d2_phase = np.diff(phase_spec, n=2, axis=0)
        wrapped_d2 = np.remainder(d2_phase + np.pi, 2 * np.pi) - np.pi
        phase_irregularity = float(np.mean(np.abs(wrapped_d2)))
        
        # 2. High-frequency spectral energy ratio (> 4 kHz vs total)
        hf_mask = freqs >= 4000.0
        total_energy = np.sum(mag_spec ** 2) + 1e-10
        hf_energy = np.sum(mag_spec[hf_mask, :] ** 2)
        hf_ratio = float(hf_energy / total_energy)
        
        # 3. Transposed Convolution Checkerboard / Periodic Ripple Index:
        # Neural upsamplers (e.g. stride 2/4/8) create periodic ripple in high-frequency subband.
        hf_mag = mag_spec[hf_mask, :]
        if hf_mag.shape[0] > 16:
            hf_profile = np.mean(hf_mag, axis=1)
            norm_hf = hf_profile - np.mean(hf_profile)
            corr = np.correlate(norm_hf, norm_hf, mode="full")
            corr = corr[len(corr) // 2 :]
            if corr[0] > 1e-8:
                norm_corr = corr / corr[0]
                # Look for secondary periodic peak between lag 2 and 14 bins
                secondary_peak = float(np.max(norm_corr[2:14])) if len(norm_corr) > 14 else 0.0
                checkerboard_index = max(0.0, secondary_peak)
            else:
                checkerboard_index = 0.0
        else:
            checkerboard_index = 0.0
            
        # 4. Spectral Flatness (Wiener entropy): Geometric Mean / Arithmetic Mean
        mean_power = np.mean(mag_spec ** 2, axis=0) + 1e-12
        geom_mean_power = np.exp(np.mean(np.log(np.maximum(mag_spec ** 2, 1e-12)), axis=0))
        flatness = float(np.mean(geom_mean_power / mean_power))
        
        return {
            "phase_irregularity": phase_irregularity,
            "hf_energy_ratio": hf_ratio,
            "checkerboard_index": checkerboard_index,
            "spectral_flatness": flatness,
        }

    def compute_spectral_descriptors(self, mag_spec: np.ndarray) -> Dict[str, float]:
        """Compute standard spectral features: centroid, roll-off, flux, and spread."""
        n_freqs, n_frames = mag_spec.shape
        freqs = np.linspace(0, self.sample_rate / 2.0, n_freqs)[:, np.newaxis]
        
        # Spectral Centroid
        col_sums = np.sum(mag_spec, axis=0) + 1e-12
        centroid = np.sum(freqs * mag_spec, axis=0) / col_sums
        mean_centroid = float(np.mean(centroid))
        
        # Spectral Spread (variance around centroid)
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag_spec, axis=0) / col_sums)
        mean_spread = float(np.mean(spread))
        
        # Spectral Roll-off (85% cumulative energy)
        cum_energy = np.cumsum(mag_spec, axis=0)
        total_e = cum_energy[-1, :]
        rolloff_idx = np.zeros(n_frames, dtype=int)
        for t in range(n_frames):
            threshold = 0.85 * total_e[t]
            idx = np.where(cum_energy[:, t] >= threshold)[0]
            rolloff_idx[t] = idx[0] if len(idx) > 0 else n_freqs - 1
        mean_rolloff = float(np.mean(freqs[rolloff_idx, 0]))
        
        # Spectral Flux (Euclidean distance between successive frames)
        if n_frames > 1:
            diff = np.diff(mag_spec, axis=1)
            flux = np.sqrt(np.sum(diff ** 2, axis=0))
            mean_flux = float(np.mean(flux))
        else:
            mean_flux = 0.0
            
        return {
            "spectral_centroid": mean_centroid,
            "spectral_spread": mean_spread,
            "spectral_rolloff": mean_rolloff,
            "spectral_flux": mean_flux,
        }

    def analyze_audio_segment(self, audio: np.ndarray) -> Dict[str, any]:
        """Full spectral & acoustic feature extraction on an audio segment."""
        mag, phase, complex_spec = self.compute_stft(audio)
        lfcc = self.compute_lfcc(mag)
        
        bicoherence = self.compute_bispectrum_coupling(complex_spec)
        vocoder_metrics = self.compute_vocoder_artifacts(mag, phase)
        descriptors = self.compute_spectral_descriptors(mag)
        
        # LFCC statistics (mean and std across time frames)
        lfcc_mean = np.mean(lfcc, axis=1)
        lfcc_std = np.std(lfcc, axis=1)
        
        return {
            "lfcc_mean": lfcc_mean.tolist(),
            "lfcc_std": lfcc_std.tolist(),
            "bicoherence_index": bicoherence,
            **vocoder_metrics,
            **descriptors,
        }
