"""Benchmark Runner CLI.

Executes comprehensive evaluation of the Voice Integrity & Anti-Cloning Engine:
- Clean 16kHz Audio Channel vs Degraded G.711 VoIP Telephony Channel
- Computes ASVspoof-standard Equal Error Rate (EER) and ROC-AUC
- Evaluates real-time processing latency and Real-Time Factor (RTF)
- Generates JSON summary reports and visual console analytics.
"""

from __future__ import annotations
import json
import os
import sys
from dataclasses import asdict
from voice_integrity.benchmark.evaluator import VoiceIntegrityBenchmark, BenchmarkMetrics


def print_header(title: str) -> None:
    width = 75
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)


def print_metrics_table(m: BenchmarkMetrics) -> None:
    print(f"\nCondition: \033[1;36m{m.channel_condition}\033[0m")
    print("-" * 65)
    print(f"  Total Test Samples Evaluated:   {m.total_samples:4d} (Genuine: {m.genuine_samples}, Synthetic: {m.synthetic_samples})")
    print(f"  Classification Accuracy:        \033[1;32m{m.accuracy * 100:6.2f}%\033[0m")
    print(f"  Precision:                      {m.precision * 100:6.2f}%")
    print(f"  Recall (Detection Rate):        \033[1;32m{m.recall * 100:6.2f}%\033[0m")
    print(f"  F1-Score:                       {m.f1_score:6.4f}")
    print(f"  ROC-AUC:                        \033[1;33m{m.roc_auc:6.4f}\033[0m")
    print(f"  Equal Error Rate (EER):         \033[1;35m{m.equal_error_rate_eer * 100:6.2f}%\033[0m  (ASVspoof standard)")
    print(f"  Optimal Decision Threshold:     {m.optimal_threshold:6.4f}")
    print(f"  Mean Chunk Latency (250ms chunk): {m.mean_chunk_latency_ms:5.2f} ms")
    print(f"  P95 Chunk Latency:              {m.p95_chunk_latency_ms:5.2f} ms")
    print(f"  Real-Time Factor (RTF):         \033[1;32m{m.real_time_factor_rtf:6.4f}x\033[0m  (< 1.0 means faster than real-time)")
    print("-" * 65)


def main() -> None:
    print_header("VOICE INTEGRITY & ANTI-CLONING ENGINE: BENCHMARK SUITE")
    print("Initializing Multi-Layer DSP & Neural Vocoder Detection Benchmark...")

    benchmark = VoiceIntegrityBenchmark(sample_rate=16000, chunk_duration_sec=0.25)
    
    # 1. Benchmark on Clean Wideband Studio Audio
    clean_metrics = benchmark.run_benchmark(
        n_samples=30,
        channel_name="Clean Studio Wideband (16kHz)",
        apply_voip_channel=False
    )
    print_metrics_table(clean_metrics)

    # 2. Benchmark on Degraded Telephony / VoIP Channel (G.711 mu-law, Bandpass 300Hz-3.4kHz, Noise, Jitter)
    voip_metrics = benchmark.run_benchmark(
        n_samples=30,
        channel_name="PSTN / VoIP Channel (G.711 mu-law, 300-3400Hz Bandpass, 25dB SNR)",
        apply_voip_channel=True
    )
    print_metrics_table(voip_metrics)

    # Export results to JSON
    report = {
        "benchmark_summary": {
            "engine": "VoiceIntegrity DSP & Neural Vocoder Verification Framework",
            "version": "1.0.0",
            "sample_rate": 16000,
            "chunk_size_ms": 250,
        },
        "results": [asdict(clean_metrics), asdict(voip_metrics)],
    }

    out_path = "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n\033[1;32m[SUCCESS]\033[0m Benchmark report saved to: {os.path.abspath(out_path)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
