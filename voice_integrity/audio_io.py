"""Audio I/O and Channel Simulation Module.

Provides robust, pure-NumPy audio loading, saving, streaming chunkers,
and telecommunication channel simulation (G.711 companding, bandpass filtering,
packet jitter, noise injection).
"""

from __future__ import annotations
import math
import struct
import wave
from typing import Generator, Tuple, Optional
import numpy as np


def load_wav(path: str, target_sr: Optional[int] = 16000) -> Tuple[np.ndarray, int]:
    """Load a WAV file into a normalized float32 numpy array (-1.0 to 1.0).
    
    Supports 8-bit, 16-bit, 24-bit, and 32-bit PCM audio.
    Automatically downmixes multichannel audio to mono.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 1:
        # 8-bit unsigned PCM
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        # 16-bit signed PCM
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        data = data / 32768.0
    elif sampwidth == 3:
        # 24-bit signed PCM (packed 3 bytes)
        raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        n_samples = len(raw_arr) // 3
        # Reshape to (N, 3), pad with sign byte to 4 bytes for int32
        reshaped = raw_arr.reshape((n_samples, 3))
        padded = np.zeros((n_samples, 4), dtype=np.uint8)
        padded[:, 1:] = reshaped
        # Handle sign extension
        sign_bit = (reshaped[:, 2] & 0x80) != 0
        padded[:, 0] = np.where(sign_bit, 0xFF, 0x00)
        data = padded.view("<i4").flatten().astype(np.float32)
        data = data / (2**23)
    elif sampwidth == 4:
        # 32-bit signed PCM
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        data = data / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    # Downmix stereo/multichannel to mono
    if n_channels > 1:
        data = data.reshape((-1, n_channels)).mean(axis=1)

    # Resample if target sample rate differs and is requested
    if target_sr is not None and target_sr != framerate:
        data = resample_audio(data, framerate, target_sr)
        framerate = target_sr

    return data.astype(np.float32), framerate


def save_wav(path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Save a float32 numpy array as a 16-bit PCM WAV file."""
    clipped = np.clip(audio, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear / sinc interpolation resampling in pure NumPy."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    num_target_samples = int(round(duration * target_sr))
    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
    resampled = np.interp(target_indices, orig_indices, audio)
    return resampled.astype(np.float32)


def stream_chunker(
    audio: np.ndarray,
    chunk_size: int,
    hop_size: int
) -> Generator[np.ndarray, None, None]:
    """Slide through audio buffer producing overlapping or consecutive chunks."""
    num_samples = len(audio)
    for start in range(0, num_samples - chunk_size + 1, hop_size):
        yield audio[start : start + chunk_size]


# ---------------------------------------------------------------------------
# Telephony Channel & Codec Degradation Simulation
# ---------------------------------------------------------------------------

def simulate_mu_law(audio: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Simulate G.711 mu-law companding quantization used in PSTN / VoIP telephony."""
    clipped = np.clip(audio, -1.0, 1.0)
    # Compression: y = sgn(x) * ln(1 + mu*|x|) / ln(1 + mu)
    sign = np.sign(clipped)
    compressed = sign * np.log1p(mu * np.abs(clipped)) / np.log1p(mu)
    # 8-bit quantization (-128 to 127)
    quantized = np.round(compressed * 127.0) / 127.0
    # Expansion: x = sgn(y) * ((1 + mu)^|y| - 1) / mu
    sign_q = np.sign(quantized)
    expanded = sign_q * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu
    return expanded.astype(np.float32)


def apply_telephony_bandpass(
    audio: np.ndarray,
    sample_rate: int = 16000,
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0
) -> np.ndarray:
    """Simulate PSTN / narrowband telephone frequency response (300Hz - 3400Hz).
    Implemented via FFT frequency-domain brickwall + cosine roll-off window.
    """
    n = len(audio)
    if n == 0:
        return audio
    fft_vals = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Soft transition band for realistic anti-aliasing filter
    mask = np.zeros_like(freqs, dtype=np.float32)
    # Bandpass region
    in_band = (freqs >= low_cutoff) & (freqs <= high_cutoff)
    mask[in_band] = 1.0

    # Low transition
    low_trans = (freqs >= low_cutoff * 0.7) & (freqs < low_cutoff)
    mask[low_trans] = 0.5 * (1.0 - np.cos(np.pi * (freqs[low_trans] - low_cutoff * 0.7) / (low_cutoff * 0.3)))

    # High transition
    high_trans = (freqs > high_cutoff) & (freqs <= high_cutoff * 1.15)
    mask[high_trans] = 0.5 * (1.0 + np.cos(np.pi * (freqs[high_trans] - high_cutoff) / (high_cutoff * 0.15)))

    filtered_fft = fft_vals * mask
    filtered_audio = np.fft.irfft(filtered_fft, n=n)
    return filtered_audio.astype(np.float32)


def add_gaussian_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add realistic background acoustic / line noise at specified Signal-to-Noise Ratio."""
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        signal_power = 1e-10
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio))
    return (audio + noise).astype(np.float32)


def simulate_voip_channel(
    audio: np.ndarray,
    sample_rate: int = 16000,
    snr_db: float = 25.0,
    packet_loss_rate: float = 0.02
) -> np.ndarray:
    """Comprehensive VoIP/telephony channel pipeline:
    G.711 mu-law quantization -> PSTN Bandpass (300Hz-3.4kHz) -> Packet loss jitter -> Line Noise.
    """
    out = simulate_mu_law(audio)
    out = apply_telephony_bandpass(out, sample_rate=sample_rate)
    
    # Simulate packet loss dropouts (20ms packets)
    packet_size = int(0.020 * sample_rate)
    num_packets = len(out) // packet_size
    for p in range(num_packets):
        if np.random.rand() < packet_loss_rate:
            start = p * packet_size
            out[start : start + packet_size] *= 0.05  # Packet concealment attenuation
            
    out = add_gaussian_noise(out, snr_db=snr_db)
    return out
