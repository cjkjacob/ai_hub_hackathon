#!/usr/bin/env python3
"""
03_extract.py — Run 3 LLMs on transcripts to extract structured form data.

Models (via Ollama): Llama 3.2-3B, Qwen 2.5-7B, Phi-4 14B

Usage:
    python 03_extract.py

Reads from:  data/transcripts/final/, data/form_schemas.json
Writes to:   data/extractions/<model>/
"""
import os
import sys
import json
import time
import subprocess

from tqdm import tqdm

from config import DATA_DIR, EXTRACTIONS_DIR, LLM_MODELS, TRANSCRIPTS_DIR
from utils.extraction import build_extraction_prompt, extract_with_ollama


def ensure_ollama_running():
    """Start Ollama server if not already running."""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:
        print("   Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)


def main():
    print("═" * 60)
    print("🧠 Step 3: LLM Entity Extraction")
    print("═" * 60)

    ensure_ollama_running()

    # Load data
    schemas_path = os.path.join(DATA_DIR, "form_schemas.json")
    with open(schemas_path) as f:
        form_schemas = json.load(f)

    final_dir = os.path.join(TRANSCRIPTS_DIR, "final")
    manifest_path = os.path.join(final_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n   Transcripts to process: {len(manifest)}")
    print(f"   Models: {', '.join(m['name'] for m in LLM_MODELS)}")

    # Determine which STT model to use for extraction input
    # Use the best model from transcription benchmarks if available
    # bench_path = os.path.join(TRANSCRIPTS_DIR, "model_benchmark_stats.json")
    # if os.path.exists(bench_path):
    #     with open(bench_path) as f:
    #         bench = json.load(f)
    #     # Pick the model with most words (proxy for accuracy)
    #     best_stt = max(bench, key=lambda m: bench[m].get("total_words", 0))
    #     print(f"   Using transcripts from: {best_stt}")
    # else:
    #     best_stt = None
    #     print("   Using first available transcript per entry")
    best_stt = "whisper-large-v3-turbo"
    print(f"   Using STT model: {best_stt}")

    # ── Run each LLM ──
    all_extractions = {}

    for model_info in LLM_MODELS:
        model_name = model_info["name"]
        ollama_id = model_info["ollama_id"]

        model_dir = os.path.join(EXTRACTIONS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"🧠 {model_name} ({model_info['params']})")
        print(f"   {model_info['description']}")
        print(f"{'─' * 60}")

        model_start = time.time()
        model_results = {}
        success_count = 0
        fail_count = 0

        for i, entry in enumerate(tqdm(manifest, desc=f"   {model_name}", leave=False)):
            rec_id = entry["rec_id"]
            form_type = entry["form_type"]
            key = f"{rec_id}_{form_type}"

            # Load transcript
            transcript_path = os.path.join(final_dir, entry["filename"])
            with open(transcript_path) as f:
                transcript_data = json.load(f)

            # Pick the best STT model's transcript
            transcripts = transcript_data.get("transcripts", {})
            if best_stt and best_stt in transcripts:
                text = transcripts[best_stt].get("text", "")
            else:
                # Fallback: use first non-empty transcript
                text = ""
                for t in transcripts.values():
                    if t.get("text"):
                        text = t["text"]
                        break

            if not text or len(text.strip()) < 10:
                print(f"\n   ⚠️  {key}: transcript too short, skipping")
                continue

            # Build prompt and extract
            sys_prompt, user_prompt = build_extraction_prompt(form_type, text, form_schemas)
            result = extract_with_ollama(ollama_id, sys_prompt, user_prompt)

            # Save
            record = {
                "rec_id": rec_id,
                "form_type": form_type,
                "model": model_name,
                "ground_truth_file": entry.get("ground_truth_file"),
                "transcript_word_count": len(text.split()),
                "extraction": result["extracted_json"],
                "raw_response": result["raw_response"],
                "elapsed_sec": result["elapsed_sec"],
                "success": result["success"],
                "error": result["error"],
            }

            with open(os.path.join(model_dir, f"{key}.json"), "w") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            model_results[key] = record

            if result["success"]:
                success_count += 1
            else:
                fail_count += 1
                tqdm.write(f"   ❌ {key}: {result['error'][:60]}")

        model_elapsed = time.time() - model_start
        all_extractions[model_name] = model_results

        avg_time = model_elapsed / max(success_count, 1)
        print(f"\n   📊 {model_name}: {success_count} success, {fail_count} failed")
        print(f"      Total: {model_elapsed/60:.1f} min, avg: {avg_time:.1f}s/transcript")

        # Unload model to free VRAM
        subprocess.run(["ollama", "stop", ollama_id], capture_output=True)
        time.sleep(2)

    # ── Save summary ──
    summary = {"models": [m["name"] for m in LLM_MODELS], "total": len(manifest), "results": {}}
    for model_info in LLM_MODELS:
        name = model_info["name"]
        results = all_extractions.get(name, {})
        s = sum(1 for r in results.values() if r.get("success"))
        f = sum(1 for r in results.values() if not r.get("success"))
        avg = sum(r.get("elapsed_sec", 0) for r in results.values() if r.get("success")) / max(s, 1)
        summary["results"][name] = {"success": s, "failed": f, "avg_time_sec": round(avg, 1)}

    with open(os.path.join(EXTRACTIONS_DIR, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Enriched manifest ──
    enriched = []
    for entry in manifest:
        key = f"{entry['rec_id']}_{entry['form_type']}"
        row = {**entry}
        for mi in LLM_MODELS:
            r = all_extractions.get(mi["name"], {}).get(key, {})
            row[f"{mi['name']}_success"] = r.get("success", False)
            row[f"{mi['name']}_time_sec"] = r.get("elapsed_sec", 0)
        enriched.append(row)

    with open(os.path.join(EXTRACTIONS_DIR, "enriched_manifest.json"), "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"\n{'═' * 60}")
    print("✅ Extraction complete!")
    print(f"{'═' * 60}")
    for name, stats in summary["results"].items():
        print(f"   {name}: {stats['success']} success, {stats['failed']} failed, avg {stats['avg_time_sec']}s")
    print(f"\n   Next: python 04_evaluate.py")


if __name__ == "__main__":
    main()
