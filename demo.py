"""Interactive Demonstration Script for Voice Integrity & Anti-Cloning Engine.

Simulates real-time live call streaming, dynamic risk score evolution,
explainable acoustic forensics, and banking fraud guardrail triggers.

Usage:
    python demo.py                  # Runs interactive live streaming simulation
    python demo.py --file call.wav  # Analyzes a custom audio file
"""

from __future__ import annotations
import argparse
import os
import sys
import time

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from voice_integrity import (
    VoiceIntegrityRiskEngine,
    RiskLevel,
    RecommendedAction,
    load_wav,
    save_wav,
)
from voice_integrity.benchmark.evaluator import generate_speech_sample


def format_risk_bar(score: float, width: int = 20) -> str:
    """Render an ASCII dynamic progress bar colored by threat severity."""
    filled = int(round(score * width))
    bar = "=" * filled + "-" * (width - filled)
    
    # ANSI color codes
    if score >= 0.70:
        return f"\033[1;31m[{bar}] {score * 100:5.1f}%\033[0m"  # Red
    elif score >= 0.35:
        return f"\033[1;33m[{bar}] {score * 100:5.1f}%\033[0m"  # Yellow
    else:
        return f"\033[1;32m[{bar}] {score * 100:5.1f}%\033[0m"  # Green


def run_streaming_simulation(
    engine: VoiceIntegrityRiskEngine,
    audio: np.ndarray,
    scenario_title: str,
    caller_id: str,
    transaction_amount_inr: float = 0.0,
    enrolled_speaker_id: str = None,
    chunk_ms: int = 250,
) -> None:
    """Simulate live streaming telephony ingestion chunk-by-chunk."""
    print("\n" + "=" * 78)
    print(f" SCENARIO: {scenario_title.upper()}")
    print("=" * 78)
    print(f" Caller ID: {caller_id} | Channel: SIP VoIP (G.711) | Txn Value: Rs. {transaction_amount_inr:,.2f}")
    if enrolled_speaker_id:
        print(f" Cross-Session Biometric Verification: Active for '{enrolled_speaker_id}'")
    print("-" * 78)
    print(f"{'Time':>6} | {'Risk Gauge':^32} | {'Level':^15} | {'Latency':>8} | {'Top Anomaly Cue'}")
    print("-" * 78)

    engine.reset_state()
    sample_rate = engine.sample_rate
    chunk_samples = int(round(chunk_ms * sample_rate / 1000.0))
    n_total = len(audio)

    start_sim_time = 0.0
    final_res = None

    # Initialize Blockchain Ledger for this call session
    from voice_integrity.ledger import CallSessionLedger, MerkleTree
    session_ledger = CallSessionLedger()

    for start in range(0, n_total - chunk_samples + 1, chunk_samples):
        chunk = audio[start : start + chunk_samples]
        res = engine.process_chunk(
            audio_chunk=chunk,
            caller_id=caller_id,
            transaction_amount_inr=transaction_amount_inr,
            enrolled_speaker_id=enrolled_speaker_id,
        )
        final_res = res
        start_sim_time += chunk_ms / 1000.0

        # Record frame in Cryptographic Ledger
        session_ledger.record_frame(res, chunk, caller_id=caller_id)

        bar = format_risk_bar(res.rolling_risk_score, width=20)
        
        # Colorize Risk Level
        if res.risk_level == RiskLevel.CRITICAL_CLONE:
            level_str = "\033[1;31mCRITICAL_CLONE\033[0m"
        elif res.risk_level == RiskLevel.ELEVATED_SUSPICIOUS:
            level_str = "\033[1;33mSUSPICIOUS\033[0m"
        else:
            level_str = "\033[1;32mAUTHENTIC\033[0m"

        cue = res.top_forensic_cues[0].metric_name if res.top_forensic_cues else "None (Nominal)"
        print(f"{start_sim_time:5.2f}s | {bar} | {level_str:^24} | {res.latency_ms:6.2f}ms | {cue}")

    # Seal Call Session into Merkle Tree & Anchor to Blockchain Ledger
    receipt = session_ledger.seal_session(final_res, caller_id=caller_id)
    safe_name = "".join(c if c.isalnum() else "_" for c in scenario_title).lower().strip("_")
    receipt_filename = f"audit_receipt_{safe_name[:24]}.json"
    session_ledger.export_receipt_json(receipt, receipt_filename)

    # Summary of final decision
    print("-" * 78)
    print(f"\033[1mFINAL CALL EVALUATION:\033[0m")
    print(f"  • Cumulative Risk Score:    {final_res.rolling_risk_score * 100:.1f}%")
    print(f"  • Authenticity Status:      {final_res.risk_level.value}")
    print(f"  • Recommended Action:       \033[1;34m{final_res.recommended_action.value}\033[0m")
    
    if final_res.top_forensic_cues:
        print(f"  • Forensic Breakdown:")
        for c in final_res.top_forensic_cues:
            print(f"      - {c.metric_name}: observed={c.observed_value} (expected: {c.typical_human_range}) -> {c.anomaly_description}")

    print("\n\033[1;36m[LEDGER / BLOCKCHAIN ANCHORING LAYER]\033[0m")
    print(f"  • Call Session ID:          {receipt.call_session_id}")
    print(f"  • Total Frames Sealed:      {receipt.total_frames_analyzed} (250ms chunks)")
    print(f"  • Merkle Root Hash:         \033[1;32m{receipt.merkle_root}\033[0m")
    print(f"  • Smart Contract Address:   {receipt.contract_address}")
    print(f"  • EVM Simulated Tx Hash:    \033[1;33m{receipt.on_chain_tx_hash}\033[0m")

    # Verify Merkle proof for Frame 0
    if receipt.sample_proof_frame_0:
        f0_valid = MerkleTree.verify_proof(receipt.leaf_hashes[0], receipt.sample_proof_frame_0, receipt.merkle_root)
        print(f"  • Cryptographic Proof (Frame #0): \033[1;32m{'VERIFIED VALID' if f0_valid else 'FAILED'}\033[0m (Tamper-evident)")
    print(f"  • Audit Receipt Exported:   {receipt_filename}")
            
    if final_res.risk_level == RiskLevel.CRITICAL_CLONE:
        if final_res.recommended_action == RecommendedAction.FREEZE_TRANSACTION:
            print("\n\033[1;41;37m [BANKING GUARDRAIL TRIGGERED] TRANSACTION FROZEN (Rs. 12,50,000.00). \033[0m")
            print("\033[1;31m Automated supervisor callback and Out-of-Band Push MFA initiated to executive phone.\033[0m")
        else:
            print("\n\033[1;41;37m [CRITICAL ALERT] SYNTHETIC / CLONED VOICE DETECTED! \033[0m")
            print("\033[1;31m Voice cloning fingerprints confirmed. Mandatory out-of-band call-back required.\033[0m")
    elif final_res.risk_level == RiskLevel.ELEVATED_SUSPICIOUS:
        print("\n\033[1;43;30m [WARNING] Anomalous acoustic indicators detected. Step-up OTP authentication required. \033[0m")
    else:
        print("\n\033[1;42;30m [PASSED] Acoustic & Prosodic profile verified. Genuine human caller. \033[0m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Integrity Anti-Cloning Demonstration")
    parser.add_argument("--file", type=str, help="Path to WAV file to evaluate", default=None)
    args = parser.parse_args()

    engine = VoiceIntegrityRiskEngine(sample_rate=16000, window_duration_sec=1.2)

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        audio, sr = load_wav(args.file, target_sr=16000)
        run_streaming_simulation(
            engine,
            audio,
            scenario_title=f"Custom File Analysis ({os.path.basename(args.file)})",
            caller_id="+91-98765-43210",
            transaction_amount_inr=0.0
        )
        return

    # Default Interactive Scenario Demonstration
    print("\n" + "=" * 78)
    print(" V-SHIELD: REAL-TIME VOICE INTEGRITY & CLONE DETECTION FRAMEWORK".center(78))
    print("=" * 78)
    print(" Demonstrating real-time DSP, acoustic phase coupling, jitter/shimmer,")
    print(" neural vocoder footprint extraction, and automated banking guardrails.")

    # 1. Enroll Genuine Speaker Baseline (e.g. Enrolled Corporate CFO Voiceprint)
    print("\n[STEP 1] Enrolling Genuine Corporate CFO Voice Baseline into Biometric Vault...")
    genuine_ref = generate_speech_sample(is_synthetic=False, duration_sec=3.5, base_f0=135.0, vowel="a")
    engine.biometric_verifier.enroll_speaker("CFO_VIKRAM_SHARMA", genuine_ref)
    print("  -> Baseline enrolled: 56-dimensional acoustic & spectral identity vector stored.")

    # 2. Scenario 1: Legitimate Call from CFO Vikram Sharma requesting vendor payment
    genuine_call = generate_speech_sample(is_synthetic=False, duration_sec=3.5, base_f0=138.0, vowel="a")
    run_streaming_simulation(
        engine,
        genuine_call,
        scenario_title="Legitimate Inbound Call - Genuine Corporate CFO",
        caller_id="+91-98110-12345",
        transaction_amount_inr=250000.0,
        enrolled_speaker_id="CFO_VIKRAM_SHARMA",
    )

    # 3. Scenario 2: Deepfake AI Voice Clone Impersonation Attack
    # Attacker uses cloned voice of CFO Vikram Sharma to authorize emergency fund transfer of Rs 12,50,000
    print("\n" + "!" * 78)
    print(" SIMULATING REAL-TIME SOCIAL ENGINEERING / VOICE CLONE ATTACK...")
    print("!" * 78)
    cloned_call = generate_speech_sample(is_synthetic=True, duration_sec=3.5, base_f0=135.0, vowel="a")
    run_streaming_simulation(
        engine,
        cloned_call,
        scenario_title="AI Voice Clone / Deepfake Impersonation Attack (High-Value Transfer)",
        caller_id="+91-98110-12345",  # Caller ID Spoofed
        transaction_amount_inr=1250000.0,
        enrolled_speaker_id="CFO_VIKRAM_SHARMA",
    )


if __name__ == "__main__":
    main()
