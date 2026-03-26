#!/usr/bin/env python3
"""
01_data_prep.py — Parse forms, convert audio, build dataset pairs.

Usage:
    python 01_data_prep.py

Reads from:  data/raw/  (your .docx forms and .ogg/.m4a audio files)
Writes to:   data/ground_truth/  data/audio_wav/  data/*.json
"""
import os
import json
import sys
from tqdm import tqdm

from config import (
    RAW_DIR, GROUND_TRUTH_DIR, AUDIO_WAV_DIR, DATA_DIR,
    AUDIO_EXTENSIONS,
)
from utils.docx_parser import parse_docx_form
from utils.audio import convert_to_wav
from utils.matching import build_dataset_map, build_pairs


def main():
    print("═" * 60)
    print("🏥 Step 1: Data Preparation")
    print("═" * 60)

    # ── Discover files ──
    if not os.path.exists(RAW_DIR):
        print(f"\n❌ Raw data directory not found: {RAW_DIR}")
        print(f"   Create it and copy your .docx and audio files there.")
        sys.exit(1)

    all_files = os.listdir(RAW_DIR)
    docx_files = sorted([f for f in all_files if f.endswith('.docx')])
    audio_files = sorted([f for f in all_files if f.lower().endswith(AUDIO_EXTENSIONS)])

    print(f"\n📁 Raw data: {RAW_DIR}")
    print(f"   📄 {len(docx_files)} .docx forms")
    print(f"   🎙️  {len(audio_files)} audio files")

    if not docx_files:
        print("\n❌ No .docx files found in data/raw/")
        sys.exit(1)
    if not audio_files:
        print("\n❌ No audio files found in data/raw/")
        sys.exit(1)

    # ── Parse all .docx forms ──
    print(f"\n{'─' * 60}")
    print("📄 Parsing .docx forms → JSON...")

    ground_truths = {}
    form_type_counts = {"initial": 0, "triage": 0, "follow_up": 0, "unknown": 0}

    for filename in tqdm(docx_files, desc="   Parsing"):
        filepath = os.path.join(RAW_DIR, filename)
        try:
            parsed = parse_docx_form(filepath)
            rec_id = parsed["metadata"]["recording_id"]
            form_type = parsed["metadata"]["form_type"]

            # Save individual JSON
            json_filename = filename.replace(".docx", ".json")
            json_path = os.path.join(GROUND_TRUTH_DIR, json_filename)
            with open(json_path, "w") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            ground_truths[filename] = parsed
            form_type_counts[form_type] += 1

        except Exception as e:
            print(f"\n   ❌ {filename}: {e}")

    print(f"\n   ✅ Parsed {len(ground_truths)} forms")
    for ft, count in form_type_counts.items():
        if count > 0:
            print(f"      {ft}: {count}")

    # ── Build schema templates per form type ──
    schemas_by_type = {}
    for filename, data in ground_truths.items():
        ft = data["metadata"]["form_type"]
        if ft not in schemas_by_type:
            schemas_by_type[ft] = {"sections": {}}
        for section, fields in data["sections"].items():
            if section not in schemas_by_type[ft]["sections"]:
                schemas_by_type[ft]["sections"][section] = []
            for field_name in fields:
                if field_name not in schemas_by_type[ft]["sections"][section]:
                    schemas_by_type[ft]["sections"][section].append(field_name)

    schemas_path = os.path.join(DATA_DIR, "form_schemas.json")
    with open(schemas_path, "w") as f:
        json.dump(schemas_by_type, f, indent=2, ensure_ascii=False)

    print(f"\n   📋 Schema templates saved: {schemas_path}")
    for ft, schema in schemas_by_type.items():
        total_fields = sum(len(fields) for fields in schema["sections"].values())
        print(f"      {ft}: {total_fields} fields across {len(schema['sections'])} sections")

    # ── Convert audio files ──
    print(f"\n{'─' * 60}")
    print("🎙️  Converting audio → 16kHz mono WAV...")

    audio_manifest = []
    for filename in tqdm(audio_files, desc="   Converting"):
        filepath = os.path.join(RAW_DIR, filename)
        try:
            info = convert_to_wav(filepath, AUDIO_WAV_DIR)
            audio_manifest.append(info)
        except Exception as e:
            print(f"\n   ❌ {filename}: {e}")

    manifest_path = os.path.join(DATA_DIR, "audio_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(audio_manifest, f, indent=2)

    total_min = sum(a["duration_minutes"] for a in audio_manifest)
    formats = set(a["original_format"] for a in audio_manifest)
    print(f"\n   ✅ Converted {len(audio_manifest)} files ({total_min:.1f} min total)")
    print(f"      Formats handled: {', '.join(sorted(formats))}")

    # ── Match audio ↔ forms ──
    print(f"\n{'─' * 60}")
    print("🔗 Matching audio ↔ forms...")

    dataset_map = build_dataset_map(audio_manifest, ground_truths)
    pairs = build_pairs(dataset_map)

    # Save
    map_path = os.path.join(DATA_DIR, "dataset_map.json")
    with open(map_path, "w") as f:
        json.dump(dataset_map, f, indent=2)

    pairs_path = os.path.join(DATA_DIR, "pairs.json")
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)

    # Report
    matched = [p for p in pairs if p["status"] == "matched"]
    missing_form = [p for p in pairs if p["status"] == "missing_form"]
    missing_audio = [p for p in pairs if p["status"] == "missing_audio"]
    needs_res = [p for p in pairs if p["status"] == "needs_resolution"]

    print(f"\n   ✅ Matched pairs: {len(matched)}")
    for p in matched:
        print(f"      {p['rec_id']} / {p['form_type']}")
    if needs_res:
        print(f"   🔍 Needs resolution: {len(needs_res)}")
    if missing_form:
        print(f"   ⚠️  Audio without form: {len(missing_form)}")
    if missing_audio:
        print(f"   ⚠️  Form without audio: {len(missing_audio)}")

    # ── Summary ──
    print(f"\n{'═' * 60}")
    print("✅ Data preparation complete!")
    print(f"{'═' * 60}")
    print(f"   Ground truth:  {GROUND_TRUTH_DIR}")
    print(f"   Audio WAVs:    {AUDIO_WAV_DIR}")
    print(f"   Schemas:       {schemas_path}")
    print(f"   Pairs:         {pairs_path}")
    print(f"\n   Next: python 02_transcribe.py")


if __name__ == "__main__":
    main()
