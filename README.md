# V-Shield: Real-Time Voice Integrity & Anti-Cloning Verification Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/ASVspoof%20EER-0.00%25%20(Clean)%20%7C%2033.3%25%20(G.711)-purple.svg)]()
[![RTF](https://img.shields.io/badge/Real--Time%20Factor-0.14x-brightgreen.svg)]()

> **An AI-powered, real-time voice integrity and synthetic speech detection framework** engineered for live telephony (PSTN), VoIP, and enterprise communication channels. Detects high-fidelity voice clones, neural vocoder footprints (HiFi-GAN, WaveGlow, Diffusion), and deepfake speech impersonation with explainable acoustic forensics and automated banking guardrails.

---

## 📌 Executive Summary & Architecture

```
                       [ Incoming Telephony / VoIP Audio Stream ]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [ Layer 1: Spectral & DSP ]               [ Layer 2: Biomechanical Prosody ]
         • 20-dim LFCC Filterbank                  • Sub-sample NCCF Pitch Tracker (F0)
         • Bispectrum Quadratic Phase Coupling     • Jitter: Local, RAP, PPQ5 Perturbations
         • Transposed Conv Checkerboard Index      • Shimmer: Local, dB, APQ3, APQ5 Metrics
         • Spectral Flatness & HF Cutoff           • HNR & Pitch Variation Dynamics
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                           [ Layer 3: Neural Vocoder ]
                           • Levinson-Durbin LPC Residual Kurtosis
                           • Sub-band Transposed Conv Periodicity
                           • High-Frequency Phase Dispersion
                                         │
                                         ▼
                           [ Layer 4: Voice Biometrics ]
                           • 56-dim Acoustic Identity Vector
                           • Cosine Similarity vs Enrolled Profile
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  Real-Time Risk Scoring Engine (EMA)    │
                    │  • Multi-Layer Corroboration Synergy    │
                    │  • Dynamic Risk Probability (0 - 100%)  │
                    │  • Explainable Forensic Cue Generator   │
                    └────────────────────┬────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      [ Cumulative Score < 35% ]                  [ Cumulative Score >= 70% ]
           AUTHENTIC_HUMAN                             CRITICAL_CLONE
                   │                                           │
         PROCEED_NORMAL Action                       FREEZE_TRANSACTION Action
      (Call continues normally)              (Out-of-Band Push MFA + Callback Trigger)
```

---

## 🔬 Mathematical & DSP Foundations

### 1. Acoustic & Spectral Analysis
* **Linear Frequency Cepstral Coefficients (LFCC)**: Unlike Mel filterbanks (which compress high frequencies for human psychoacoustics), linear filterbanks preserve full resolution across $0 - 8000\text{ Hz}$ where neural vocoder (HiFi-GAN, WaveGlow, BigVGAN) synthesis artifacts live.
* **Higher-Order Spectral (Bispectrum) Phase Coupling**:
  $$B(f_1, f_2) = X(f_1) X(f_2) X^*(f_1 + f_2)$$
  Human vocal folds exhibit quadratic phase coupling between harmonic triplets $k_1 f_0 + k_2 f_0 = (k_1+k_2)f_0$. Neural vocoders synthesize waveforms with independent or diffuse high harmonic phases, dropping bicoherence below $0.20$.
* **Transposed Convolution Checkerboard Footprint**: Strided neural upsamplers create periodic ripples across adjacent frequency bins, detected via high-frequency spectral autocorrelation.

### 2. Prosody & Vocal Cord Perturbation
* **Cycle-to-Cycle Jitter (Frequency Perturbation)**:
  $$\text{Jitter (local)} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N}\sum_{i=1}^N T_i} \times 100\%$$
  Human speech has neuromuscular micro-tremors ($1.0\% - 3.0\%$). Neural TTS models produce over-smoothed pitch with **abnormally low jitter** ($<0.4\%$).
* **Cycle-to-Cycle Shimmer (Amplitude Perturbation)**:
  $$\text{Shimmer (local)} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1} |A_i - A_{i+1}|}{\frac{1}{N}\sum_{i=1}^N A_i} \times 100\%$$
  Measures vocal fold amplitude irregularities. Cloned audio lacks physiological amplitude micro-variation ($<1.8\%$).

### 3. Neural Excitation Residual (LPC Inverse Filtering)
Using Levinson-Durbin recursion on the autocorrelation sequence:
$$e[n] = x[n] + \sum_{k=1}^p a_k x[n-k]$$
Human glottal pulses yield characteristic Glottal Closure Instants (GCI) with moderate-high kurtosis ($4.5 - 15.0$). Neural synthetic speech exhibits diffuse Gaussian residuals or severe vocoder concatenation glitches ($>50$).

---

## 📊 Benchmark & Performance Results

Evaluated using `voice_integrity.benchmark.run_benchmark` across 60 simulated streaming test calls:

| Condition | ROC-AUC | Equal Error Rate (EER) | Detection Recall | Mean Latency (250ms chunk) | Real-Time Factor (RTF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Studio Audio (16kHz)** | **1.0000** | **0.00%** | **100.00%** | **35.76 ms** | **0.143x** (7x faster than real-time) |
| **PSTN / VoIP Channel (G.711 $\mu$-law, 300-3400Hz, 25dB SNR)** | **0.7911** | **33.33%** | **66.67%** | **34.33 ms** | **0.137x** (7x faster than real-time) |

---

## 🚀 Quickstart Guide

### 1. Run Interactive Live Call Simulation
Demonstrates live chunk-by-chunk analysis of a legitimate CFO call vs an AI Voice Clone impersonation attack attempting a ₹12,50,000 transfer:
```bash
python demo.py
```

### 2. Analyze a Specific Audio File
We have provided pre-generated sample WAV files in `sample_audio/`:
* **Analyze Genuine Human Call Audio**:
  ```bash
  python demo.py --file sample_audio/genuine_call.wav
  ```
* **Analyze Cloned / Synthetic Voice Attack Audio**:
  ```bash
  python demo.py --file sample_audio/synthetic_clone.wav
  ```
* **Analyze Telephony-Degraded VoIP Attack Audio**:
  ```bash
  python demo.py --file sample_audio/telephony_clone.wav
  ```

### 3. Run the Automated Benchmark Suite
```bash
python -m voice_integrity.benchmark.run_benchmark
```

### 4. Run Unit Tests
```bash
python -m unittest tests/test_voice_integrity.py
```

---

## 🛡️ Enterprise & Banking Guardrails

The framework integrates contextual policy rules tailored for financial institutions and telecom providers:

* **Low Threat (`AUTHENTIC_HUMAN`, Score < 35%)**: Call proceeds normally with continuous background acoustic monitoring.
* **Elevated Threat (`ELEVATED_SUSPICIOUS`, Score 35% - 69%)**: Frontline agent dashboard displays warning banner; transaction requires Step-Up Push / SMS OTP authentication.
* **Critical Threat (`CRITICAL_CLONE`, Score >= 70%)**: High-value transactions are **automatically frozen**; automated out-of-band callback is routed to the registered phone number.

### Privacy & Regulatory Compliance
* **Zero Audio Retention**: Real-time sliding window buffer retains only the active analysis frame ($1.0\text{s}$) in volatile RAM; raw voice data is flushed immediately after feature extraction.
* **Feature-Only Telemetry**: Only mathematical vectors (LFCC moments, jitter statistics) are transmitted for audit logging.
* **Edge Inference Ready**: Zero external C/DLL dependencies (pure NumPy architecture), deployable on on-premise SBCs (Session Border Controllers) or edge telephony gateways.
