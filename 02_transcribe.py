#!/usr/bin/env python3
"""
02_transcribe.py — Run 3 STT models on all audio files.

Models:
  1. Whisper Large V3 Turbo  (faster-whisper, 809M params)
  2. Qwen3-ASR-0.6B          (qwen-asr, 0.6B params)
  3. Qwen3-ASR-1.7B          (qwen-asr, 1.7B params)

Usage:
    python 02_transcribe.py

Reads from:  data/pairs.json, data/audio_manifest.json, data/audio_wav/
Writes to:   data/transcripts/<model_name>/, data/transcripts/final/
"""
import os
import sys
import json
import time
import re
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from config import (
    DATA_DIR, AUDIO_WAV_DIR, TRANSCRIPTS_DIR,
    STT_MODELS,
)

FINAL_DIR = os.path.join(TRANSCRIPTS_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)


# ════════════════════════════════════════════════
# STT Engine Wrappers
# ════════════════════════════════════════════════

def transcribe_faster_whisper(model_id, wav_files, audio_dir, output_dir):
    """Transcribe using faster-whisper (for Whisper models)."""
    from faster_whisper import WhisperModel

    os.makedirs(output_dir, exist_ok=True)

    print(f"   Loading faster-whisper model '{model_id}'...")
    load_start = time.time()
    model = WhisperModel(model_id, device="cuda", compute_type="float16")
    print(f"   Loaded in {time.time() - load_start:.1f}s")

    results = {}

    for wav_file in tqdm(wav_files, desc="   Transcribing", leave=False):
        wav_path = os.path.join(audio_dir, wav_file)
        if not os.path.exists(wav_path):
            print(f"\n   ⚠️  Not found: {wav_file}")
            continue

        start = time.time()
        try:
            segments, info = model.transcribe(
                wav_path,
                beam_size=5,
                language=None,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            segment_list = []
            text_parts = []
            for seg in segments:
                segment_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
                text_parts.append(seg.text.strip())

            full_text = " ".join(text_parts)
            elapsed = time.time() - start

            # Get audio duration from manifest
            audio_dur = _get_audio_duration(wav_file)
            rtf = elapsed / audio_dur if audio_dur > 0 else 0

            result = {
                "wav_file": wav_file,
                "full_text": full_text,
                "num_segments": len(segment_list),
                "segments": segment_list,
                "audio_duration_sec": round(audio_dur, 1),
                "transcription_time_sec": round(elapsed, 1),
                "real_time_factor": round(rtf, 4),
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
            }

        except Exception as e:
            elapsed = time.time() - start
            result = {
                "wav_file": wav_file,
                "full_text": "",
                "error": str(e),
                "transcription_time_sec": round(elapsed, 1),
            }
            print(f"\n   ❌ {wav_file}: {e}")

        results[wav_file] = result

        # Save individual transcript
        safe_name = Path(wav_file).stem
        with open(os.path.join(output_dir, f"{safe_name}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def transcribe_qwen_asr(model_id, wav_files, audio_dir, output_dir):
    """Transcribe using Qwen3-ASR (qwen-asr package)."""
    from qwen_asr import Qwen3ASRModel

    os.makedirs(output_dir, exist_ok=True)

    print(f"   Loading Qwen3-ASR model '{model_id}'...")
    load_start = time.time()
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=512,  # Higher for long consultations
    )
    print(f"   Loaded in {time.time() - load_start:.1f}s")

    results = {}

    for wav_file in tqdm(wav_files, desc="   Transcribing", leave=False):
        wav_path = os.path.join(audio_dir, wav_file)
        if not os.path.exists(wav_path):
            print(f"\n   ⚠️  Not found: {wav_file}")
            continue

        start = time.time()
        try:
            # Qwen3-ASR returns a list of result objects with .text and .language
            output = model.transcribe(
                audio=wav_path,
                language="English",
            )

            # Extract text from results
            if isinstance(output, list) and len(output) > 0:
                full_text = output[0].text if hasattr(output[0], 'text') else str(output[0])
                detected_lang = output[0].language if hasattr(output[0], 'language') else "en"
            elif hasattr(output, 'text'):
                full_text = output.text
                detected_lang = getattr(output, 'language', "en")
            else:
                full_text = str(output)
                detected_lang = "en"

            full_text = full_text.strip()
            elapsed = time.time() - start

            audio_dur = _get_audio_duration(wav_file)
            rtf = elapsed / audio_dur if audio_dur > 0 else 0

            result = {
                "wav_file": wav_file,
                "full_text": full_text,
                "audio_duration_sec": round(audio_dur, 1),
                "transcription_time_sec": round(elapsed, 1),
                "real_time_factor": round(rtf, 4),
                "language": str(detected_lang),
            }

        except Exception as e:
            elapsed = time.time() - start
            result = {
                "wav_file": wav_file,
                "full_text": "",
                "error": str(e),
                "transcription_time_sec": round(elapsed, 1),
            }
            print(f"\n   ❌ {wav_file}: {e}")

        results[wav_file] = result

        safe_name = Path(wav_file).stem
        with open(os.path.join(output_dir, f"{safe_name}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════

_audio_manifest = None

def _get_audio_duration(wav_file):
    """Get audio duration from manifest (cached)."""
    global _audio_manifest
    if _audio_manifest is None:
        manifest_path = os.path.join(DATA_DIR, "audio_manifest.json")
        with open(manifest_path) as f:
            _audio_manifest = {a["wav_file"]: a for a in json.load(f)}
    entry = _audio_manifest.get(wav_file, {})
    return entry.get("duration_seconds", 0)


def detect_form_type_from_transcript(text, max_words=200):
    """Detect form type from the beginning of a transcript."""
    words = text.split()[:max_words]
    start = " ".join(words).lower()

    if any(kw in start for kw in ["initial consultation", "initial form"]):
        return "initial"
    if any(kw in start for kw in ["follow-up", "follow up", "followup"]):
        return "follow_up"
    if "triage" in start:
        return "triage"
    return "unknown"


def detect_all_form_types(text):
    """Scan entire transcript for multiple form type headers."""
    text_lower = text.lower()
    patterns = {
        "initial": [r"initial\s+consultation\s+form", r"initial\s+form"],
        "follow_up": [r"follow[\s-]*up\s+consultation\s+form", r"follow[\s-]*up\s+form"],
        "triage": [r"triage\s+form"],
    }
    occurrences = []
    for form_type, pats in patterns.items():
        for pat in pats:
            for match in re.finditer(pat, text_lower):
                occurrences.append((match.start(), form_type))

    occurrences.sort(key=lambda x: x[0])
    found = []
    for _, ft in occurrences:
        if not found or found[-1] != ft:
            found.append(ft)
    return found


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("🎙️  Step 2: Speech-to-Text Transcription")
    print("═" * 60)

    # Load data
    with open(os.path.join(DATA_DIR, "pairs.json")) as f:
        pairs = json.load(f)
    with open(os.path.join(DATA_DIR, "audio_manifest.json")) as f:
        audio_manifest = json.load(f)

    # Unique WAV files to transcribe
    all_wav_files = sorted(set(a["wav_file"] for a in audio_manifest))
    matched_pairs = [p for p in pairs if p["status"] == "matched"]

    total_min = sum(a["duration_minutes"] for a in audio_manifest)
    print(f"\n   Audio files: {len(all_wav_files)}")
    print(f"   Matched pairs: {len(matched_pairs)}")
    print(f"   Total audio: {total_min:.1f} min")
    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Run each STT model ──
    all_results = {}  # {model_name: {wav_file: result}}

    for model_info in STT_MODELS:
        model_name = model_info["name"]
        engine = model_info["engine"]
        model_id = model_info["model_id"]

        model_output_dir = os.path.join(TRANSCRIPTS_DIR, model_name)

        print(f"\n{'─' * 60}")
        print(f"🔊 {model_name} ({model_info['params']})")
        print(f"   {model_info['description']}")
        print(f"{'─' * 60}")

        model_start = time.time()

        if engine == "faster-whisper":
            results = transcribe_faster_whisper(model_id, all_wav_files, AUDIO_WAV_DIR, model_output_dir)
        elif engine == "qwen-asr":
            results = transcribe_qwen_asr(model_id, all_wav_files, AUDIO_WAV_DIR, model_output_dir)
        else:
            print(f"   ❌ Unknown engine: {engine}")
            continue

        model_elapsed = time.time() - model_start
        all_results[model_name] = results

        # Stats
        successful = {k: v for k, v in results.items() if "error" not in v}
        total_audio = sum(v.get("audio_duration_sec", 0) for v in successful.values())
        total_proc = sum(v.get("transcription_time_sec", 0) for v in successful.values())
        total_words = sum(len(v.get("full_text", "").split()) for v in successful.values())
        avg_rtf = total_proc / total_audio if total_audio > 0 else 0

        print(f"\n   📊 {model_name} summary:")
        print(f"      Processed: {len(successful)}/{len(all_wav_files)} files")
        print(f"      Total audio: {total_audio/60:.1f} min")
        print(f"      Processing time: {total_proc/60:.1f} min (wall: {model_elapsed/60:.1f} min)")
        print(f"      Avg RTF: {avg_rtf:.4f} ({1/avg_rtf:.0f}× real-time)" if avg_rtf > 0 else "")
        print(f"      Total words: {total_words}")

    # ── Save benchmark summary ──
    benchmark = {}
    for model_name, results in all_results.items():
        successful = {k: v for k, v in results.items() if "error" not in v}
        total_audio = sum(v.get("audio_duration_sec", 0) for v in successful.values())
        total_proc = sum(v.get("transcription_time_sec", 0) for v in successful.values())
        benchmark[model_name] = {
            "files": len(successful),
            "total_audio_min": round(total_audio / 60, 1),
            "total_proc_min": round(total_proc / 60, 1),
            "avg_rtf": round(total_proc / total_audio, 4) if total_audio > 0 else 0,
            "total_words": sum(len(v.get("full_text", "").split()) for v in successful.values()),
        }

    bench_path = os.path.join(TRANSCRIPTS_DIR, "model_benchmark_stats.json")
    with open(bench_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    # ── Per-file comparison summary ──
    comparison = {}
    for wav_file in all_wav_files:
        comparison[wav_file] = {}
        for model_name in all_results:
            r = all_results[model_name].get(wav_file, {})
            comparison[wav_file][model_name] = {
                "word_count": len(r.get("full_text", "").split()),
                "transcription_time_sec": r.get("transcription_time_sec", 0),
                "real_time_factor": r.get("real_time_factor", 0),
                "has_error": "error" in r,
            }

    comp_path = os.path.join(TRANSCRIPTS_DIR, "transcription_summary.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Resolve unspecified form types ──
    print(f"\n{'─' * 60}")
    print("🔍 Resolving unspecified form types from transcripts...")

    # Use the model with most words (likely most accurate) for resolution
    best_model = max(benchmark, key=lambda m: benchmark[m].get("total_words", 0))
    print(f"   Using {best_model} transcripts for resolution")

    unresolved = [p for p in pairs if p["status"] == "needs_resolution"]
    resolutions = []

    for pair in unresolved:
        wav_file = pair["wav_file"]
        result = all_results.get(best_model, {}).get(wav_file, {})
        text = result.get("full_text", "")

        if text:
            detected = detect_form_type_from_transcript(text)
            print(f"   {pair['rec_id']} ({wav_file}): detected '{detected}'")
            if detected != "unknown":
                pair["form_type"] = detected
                pair["status"] = "resolved_from_transcript"
                resolutions.append({"rec_id": pair["rec_id"], "resolved_type": detected})

    # ── Detect multi-form recordings ──
    print(f"\n🔎 Checking for multi-form recordings...")
    multi_form = []
    for wav_file in all_wav_files:
        result = all_results.get(best_model, {}).get(wav_file, {})
        text = result.get("full_text", "")
        if text:
            types_found = detect_all_form_types(text)
            if len(types_found) > 1:
                multi_form.append({"wav_file": wav_file, "form_types": types_found})
                print(f"   📎 {wav_file}: contains {' → '.join(types_found)}")

    if not multi_form:
        print("   None found.")

    # Save resolutions
    res_path = os.path.join(DATA_DIR, "form_type_resolutions.json")
    with open(res_path, "w") as f:
        json.dump({"resolved": resolutions, "multi_form": multi_form}, f, indent=2)

    # Save updated pairs
    pairs_path = os.path.join(DATA_DIR, "pairs.json")
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)

    # ── Export final transcripts for Step 3 ──
    print(f"\n{'─' * 60}")
    print("📋 Exporting final transcripts...")

    processable = [p for p in pairs if p["status"] in ("matched", "resolved_from_transcript")]

    # Deduplicate by (rec_id, form_type)
    seen = set()
    unique_processable = []
    for p in processable:
        key = f"{p['rec_id']}_{p['form_type']}"
        if key not in seen:
            seen.add(key)
            unique_processable.append(p)

    manifest_entries = []

    for pair in unique_processable:
        rec_id = pair["rec_id"]
        form_type = pair["form_type"]
        wav_file = pair["wav_file"]

        # Collect transcripts from all models
        entry = {
            "rec_id": rec_id,
            "form_type": form_type,
            "wav_file": wav_file,
            "ground_truth_file": pair.get("ground_truth_file"),
            "transcripts": {},
        }

        for model_name in all_results:
            r = all_results[model_name].get(wav_file, {})
            entry["transcripts"][model_name] = {
                "text": r.get("full_text", ""),
                "word_count": len(r.get("full_text", "").split()),
                "time_sec": r.get("transcription_time_sec", 0),
                "rtf": r.get("real_time_factor", 0),
            }

        # Save
        filename = f"{rec_id}_{form_type}.json"
        filepath = os.path.join(FINAL_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        # Best model's word count for manifest
        best_wc = entry["transcripts"].get(best_model, {}).get("word_count", 0)
        manifest_entries.append({
            "rec_id": rec_id,
            "form_type": form_type,
            "filename": filename,
            "wav_file": wav_file,
            "ground_truth_file": pair.get("ground_truth_file"),
            "best_model": best_model,
            "best_word_count": best_wc,
        })

        print(f"   ✅ {filename} ({best_wc} words)")

    manifest_path = os.path.join(FINAL_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    # ── Final summary ──
    print(f"\n{'═' * 60}")
    print("✅ Transcription complete!")
    print(f"{'═' * 60}")

    print(f"\n   📊 Model benchmark:")
    print(f"   {'Model':<25} {'Files':>6} {'Time':>8} {'RTF':>8} {'Words':>8}")
    print(f"   {'─'*25} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
    for name, stats in benchmark.items():
        print(f"   {name:<25} {stats['files']:>6} {stats['total_proc_min']:>6.1f}m {stats['avg_rtf']:>8.4f} {stats['total_words']:>8}")

    print(f"\n   Final transcripts: {len(manifest_entries)}")
    print(f"   Saved to: {FINAL_DIR}")
    if resolutions:
        print(f"   Resolved: {len(resolutions)} unspecified form types")
    if multi_form:
        print(f"   Multi-form recordings: {len(multi_form)}")

    print(f"\n   Next: python 03_extract.py")


if __name__ == "__main__":
    main()
