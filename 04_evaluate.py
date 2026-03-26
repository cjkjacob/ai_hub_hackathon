#!/usr/bin/env python3
"""
04_evaluate.py — Compare extracted fields vs ground truth.

Usage:
    python 04_evaluate.py

Reads from:  data/ground_truth/, data/extractions/
Writes to:   results/evaluation/
"""
import os
import json
from collections import defaultdict

from config import DATA_DIR, GROUND_TRUTH_DIR, EXTRACTIONS_DIR, EVAL_DIR, LLM_MODELS, TRANSCRIPTS_DIR
from utils.scoring import (
    values_match, find_extracted_value, is_critical_field, compute_accuracy,
)

MODEL_LABELS = [m["name"] for m in LLM_MODELS]


def main():
    print("═" * 60)
    print("📊 Step 4: Evaluation")
    print("═" * 60)

    # Load enriched manifest
    with open(os.path.join(EXTRACTIONS_DIR, "enriched_manifest.json")) as f:
        manifest = json.load(f)

    # Load ground truth
    ground_truths = {}
    for entry in manifest:
        gt_file = entry.get("ground_truth_file")
        if gt_file and gt_file not in ground_truths:
            gt_path = os.path.join(GROUND_TRUTH_DIR, gt_file)
            if os.path.exists(gt_path):
                with open(gt_path) as f:
                    ground_truths[gt_file] = json.load(f)

    # Load extractions
    extractions = {}
    for ml in MODEL_LABELS:
        extractions[ml] = {}
        model_dir = os.path.join(EXTRACTIONS_DIR, ml)
        if os.path.exists(model_dir):
            for fname in os.listdir(model_dir):
                if fname.endswith(".json"):
                    with open(os.path.join(model_dir, fname)) as f:
                        extractions[ml][fname.replace(".json", "")] = json.load(f)

    print(f"\n   Entries: {len(manifest)}")
    print(f"   Ground truth files: {len(ground_truths)}")
    print(f"   Models: {', '.join(MODEL_LABELS)}")

    # ── Field-by-field evaluation ──
    all_results = []

    for entry in manifest:
        rec_id = entry["rec_id"]
        form_type = entry["form_type"]
        key = f"{rec_id}_{form_type}"
        gt_file = entry.get("ground_truth_file")

        if not gt_file or gt_file not in ground_truths:
            continue

        gt_flat = ground_truths[gt_file].get("flat_fields", {})

        for ml in MODEL_LABELS:
            success = entry.get(f"{ml}_success", False)
            ext_data = extractions.get(ml, {}).get(key, {})
            ext_json = ext_data.get("extraction") if success else None

            for field_name, gt_value in gt_flat.items():
                if not success or not ext_json:
                    all_results.append({
                        "rec_id": rec_id, "form_type": form_type, "model": ml,
                        "field_name": field_name, "gt_value": gt_value,
                        "extracted_value": None, "confidence": None,
                        "match_type": "extraction_failed", "score": 0.0,
                        "is_critical": is_critical_field(field_name),
                    })
                    continue

                ext_val, conf = find_extracted_value(ext_json, field_name)
                if ext_val is None:
                    mt, sc = "not_found", 0.0
                else:
                    mt, sc = values_match(gt_value, ext_val)
                    if sc is None:
                        sc = 0.0

                all_results.append({
                    "rec_id": rec_id, "form_type": form_type, "model": ml,
                    "field_name": field_name, "gt_value": gt_value,
                    "extracted_value": ext_val, "confidence": conf,
                    "match_type": mt, "score": sc,
                    "is_critical": is_critical_field(field_name),
                })

    print(f"\n   Total field comparisons: {len(all_results)}")

    # ── Overall accuracy ──
    print(f"\n{'═' * 70}")
    print(f"📊 OVERALL ACCURACY BY MODEL")
    print(f"{'═' * 70}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Exact':>10} {'Partial':>10} {'Miss':>10} {'N':>8}")
    print(f"{'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    overall = {}
    for ml in MODEL_LABELS:
        stats = compute_accuracy([r for r in all_results if r["model"] == ml])
        overall[ml] = stats
        print(f"{ml:<20} {stats['accuracy']:>9.1f}% {stats['exact_rate']:>9.1f}% {stats['partial_rate']:>9.1f}% {stats['miss_rate']:>9.1f}% {stats['n']:>8}")

    # ── Critical fields ──
    print(f"\n{'═' * 70}")
    print(f"🔴 CRITICAL FIELD ACCURACY (meds, dosages, diagnoses, safety)")
    print(f"{'═' * 70}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Exact':>10} {'Miss':>10} {'N':>8}")
    print(f"{'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    critical = {}
    for ml in MODEL_LABELS:
        stats = compute_accuracy([r for r in all_results if r["model"] == ml and r["is_critical"]])
        critical[ml] = stats
        print(f"{ml:<20} {stats['accuracy']:>9.1f}% {stats['exact_rate']:>9.1f}% {stats['miss_rate']:>9.1f}% {stats['n']:>8}")

    # ── By form type ──
    print(f"\n{'═' * 70}")
    print(f"📋 ACCURACY BY FORM TYPE")
    print(f"{'═' * 70}")

    for ft in ["initial", "follow_up", "triage"]:
        print(f"\n  ── {ft.upper()} ──")
        print(f"  {'Model':<20} {'Accuracy':>10} {'Exact':>10} {'Miss':>10} {'N':>8}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
        for ml in MODEL_LABELS:
            stats = compute_accuracy([r for r in all_results if r["model"] == ml and r["form_type"] == ft])
            print(f"  {ml:<20} {stats['accuracy']:>9.1f}% {stats['exact_rate']:>9.1f}% {stats['miss_rate']:>9.1f}% {stats['n']:>8}")

    # ── Confidence calibration ──
    print(f"\n{'═' * 70}")
    print(f"🎯 CONFIDENCE CALIBRATION")
    print(f"{'═' * 70}")

    for ml in MODEL_LABELS:
        mr = [r for r in all_results if r["model"] == ml and r["confidence"] and r["match_type"] not in ("skip", "extraction_failed")]
        if not mr:
            continue
        print(f"\n  {ml}:")
        print(f"  {'Confidence':<15} {'Accuracy':>10} {'Count':>8}")
        print(f"  {'─'*15} {'─'*10} {'─'*8}")
        for level in ["high", "medium", "low", "unknown"]:
            sub = [r for r in mr if str(r["confidence"]).lower() == level]
            if sub:
                stats = compute_accuracy(sub)
                print(f"  {level:<15} {stats['accuracy']:>9.1f}% {stats['n']:>8}")

    # ── Most missed fields ──
    print(f"\n{'═' * 70}")
    print(f"❌ MOST MISSED FIELDS (top 10 per model)")
    print(f"{'═' * 70}")

    for ml in MODEL_LABELS:
        mr = [r for r in all_results if r["model"] == ml and r["match_type"] != "skip"]
        counts = defaultdict(lambda: {"miss": 0, "total": 0})
        for r in mr:
            counts[r["field_name"]]["total"] += 1
            if r["match_type"] in ("miss", "not_found", "extraction_failed"):
                counts[r["field_name"]]["miss"] += 1

        sorted_fields = sorted(counts.items(), key=lambda x: x[1]["miss"], reverse=True)
        print(f"\n  {ml}:")
        for field, c in sorted_fields[:10]:
            pct = c["miss"] / c["total"] * 100 if c["total"] > 0 else 0
            crit = " 🔴" if is_critical_field(field) else ""
            print(f"    {field[:50]:<52} {c['miss']:>3}/{c['total']:<3} ({pct:.0f}%){crit}")

    # ── Combined pipeline table ──
    stt_path = os.path.join(TRANSCRIPTS_DIR, "model_benchmark_stats.json")
    ext_path = os.path.join(EXTRACTIONS_DIR, "extraction_summary.json")

    if os.path.exists(stt_path) and os.path.exists(ext_path):
        with open(stt_path) as f:
            stt_stats = json.load(f)
        with open(ext_path) as f:
            ext_summary = json.load(f)

        print(f"\n{'═' * 70}")
        print(f"🏆 COMBINED PIPELINE: STT + LLM")
        print(f"{'═' * 70}")

        print(f"\n  STT:")
        for name, s in stt_stats.items():
            speed = f"{1/s['avg_rtf']:.0f}×" if s['avg_rtf'] > 0 else "N/A"
            print(f"    {name:<25} RTF={s['avg_rtf']:.4f} ({speed} real-time), {s['total_words']} words")

        print(f"\n  LLM:")
        for ml in MODEL_LABELS:
            s = ext_summary["results"].get(ml, {})
            a = overall.get(ml, {}).get("accuracy", 0)
            c = critical.get(ml, {}).get("accuracy", 0)
            print(f"    {ml:<25} {s.get('success',0)}/{ext_summary['total']} success, accuracy={a:.1f}%, critical={c:.1f}%")

    # ── Save everything ──
    os.makedirs(EVAL_DIR, exist_ok=True)

    with open(os.path.join(EVAL_DIR, "all_field_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(EVAL_DIR, "aggregate_stats.json"), "w") as f:
        json.dump({"overall": overall, "critical": critical}, f, indent=2)

    # ── Pitch stats ──
    best_ml = max(MODEL_LABELS, key=lambda m: overall.get(m, {}).get("accuracy", 0))
    ba = overall[best_ml]["accuracy"]
    ca = critical[best_ml]["accuracy"]

    print(f"\n{'═' * 70}")
    print(f"🎤 PITCH-READY STATS")
    print(f"{'═' * 70}")
    print(f"   Best LLM: {best_ml}")
    print(f"   Overall accuracy: {ba:.1f}%")
    print(f"   Critical field accuracy: {ca:.1f}%")
    print(f"   Total field comparisons: {len(all_results)}")
    print(f"   All models self-hosted, zero external API calls")

    print(f"\n💾 Results saved to: {EVAL_DIR}")
    print(f"   Next: python 05_demo.py")


if __name__ == "__main__":
    main()
