#!/usr/bin/env python3
"""
wer_checker.py — WER & field accuracy analysis.

Two types of comparison:

  TRANSCRIPT WER (Speech-to-Text quality):
    python wer_checker.py --cross-compare          # Models vs each other
    python wer_checker.py --gold-dir gold_txts/     # Models vs gold transcripts

  FIELD ACCURACY (Extraction quality — JSON vs JSON):
    python wer_checker.py --json-compare            # Extracted fields vs ground truth

Reads transcript folders from: data/transcripts/<model_name>/
Reads extraction folders from: data/extractions/<model_name>/
Reads ground truth from:       data/ground_truth/
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

try:
    import jiwer
except ImportError:
    print("❌ jiwer not installed. Run: pip install jiwer")
    sys.exit(1)


# ════════════════════════════════════════════════
# WER computation
# ════════════════════════════════════════════════

def compute_wer_detailed(reference: str, hypothesis: str) -> dict:
    """Compute detailed WER metrics."""
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

    ref_clean = transform(reference)
    hyp_clean = transform(hypothesis)

    if not ref_clean.strip():
        return {
            "wer": 1.0 if hyp_clean.strip() else 0.0,
            "substitutions": 0, "insertions": len(hyp_clean.split()),
            "deletions": 0, "hits": 0,
            "ref_words": 0, "hyp_words": len(hyp_clean.split()),
        }

    output = jiwer.process_words(ref_clean, hyp_clean)

    return {
        "wer": round(output.wer, 4),
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "hits": output.hits,
        "ref_words": len(ref_clean.split()),
        "hyp_words": len(hyp_clean.split()),
    }


# ════════════════════════════════════════════════
# Load transcripts from model folders
# ════════════════════════════════════════════════

def load_transcripts_from_folders(transcripts_dir: str) -> dict:
    """
    Load transcripts from individual model folders.

    Reads:  data/transcripts/<model_name>/<file>.json
    Each JSON has a "full_text" field.

    Returns: {model_name: {stem: full_text}}
    """
    models = {}

    for entry in os.scandir(transcripts_dir):
        if not entry.is_dir():
            continue
        if entry.name in ("final",):
            continue

        model_name = entry.name
        models[model_name] = {}

        for f in sorted(os.listdir(entry.path)):
            if not f.endswith(".json"):
                continue
            filepath = os.path.join(entry.path, f)
            try:
                with open(filepath) as fh:
                    data = json.load(fh)
                text = data.get("full_text", "")
                stem = Path(f).stem
                models[model_name][stem] = text
            except Exception:
                pass

    return models


# ════════════════════════════════════════════════
# Mode 1: Cross-compare STT models
# ════════════════════════════════════════════════

def cross_compare(transcripts_dir: str):
    """Compare all STT models against each other using transcript folders."""
    models = load_transcripts_from_folders(transcripts_dir)

    if len(models) < 2:
        print(f"❌ Found {len(models)} model(s). Need at least 2 to cross-compare.")
        print(f"   Looking in: {transcripts_dir}")
        print(f"   Found: {list(models.keys())}")
        sys.exit(1)

    # Word counts per model
    word_counts = {
        model: sum(len(t.split()) for t in texts.values())
        for model, texts in models.items()
    }

    # Reference = model with most words
    ref_model = max(word_counts, key=word_counts.get)

    # Common files across all models
    all_stems = set()
    for texts in models.values():
        all_stems.update(texts.keys())

    print(f"\n{'═' * 70}")
    print(f"📊 STT CROSS-COMPARISON (WER)")
    print(f"{'═' * 70}")
    print(f"\n  Pseudo-reference: {ref_model} ({word_counts[ref_model]} total words)")
    print(f"  (Model with most words assumed closest to ground truth)\n")

    print(f"  {'Model':<30} {'Total Words':>12} {'Files':>8}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 8}")
    for model in sorted(models.keys()):
        marker = " ← ref" if model == ref_model else ""
        print(f"  {model:<30} {word_counts[model]:>12} {len(models[model]):>8}{marker}")

    # Compare each model vs reference
    all_results = {}

    for hyp_model in sorted(models.keys()):
        if hyp_model == ref_model:
            continue

        per_file = []
        all_ref_texts = []
        all_hyp_texts = []

        # Find common files between ref and hyp
        common = sorted(set(models[ref_model].keys()) & set(models[hyp_model].keys()))

        for stem in common:
            ref_text = models[ref_model][stem]
            hyp_text = models[hyp_model][stem]

            if not ref_text.strip():
                continue

            metrics = compute_wer_detailed(ref_text, hyp_text)
            per_file.append({"file": stem, **metrics})
            all_ref_texts.append(ref_text)
            all_hyp_texts.append(hyp_text)

        if not all_ref_texts:
            print(f"\n  {hyp_model}: no common files with reference")
            continue

        # Aggregate WER
        agg = compute_wer_detailed(" ".join(all_ref_texts), " ".join(all_hyp_texts))

        print(f"\n  {'━' * 65}")
        print(f"  {hyp_model} vs {ref_model}")
        print(f"  {'━' * 65}")
        print(f"  Aggregate WER: {agg['wer']:.1%}  |  "
              f"Sub: {agg['substitutions']}  Ins: {agg['insertions']}  Del: {agg['deletions']}")
        print(f"  Files compared: {len(per_file)}")

        print(f"\n  {'File':<35} {'WER':>8} {'Ref':>6} {'Hyp':>6} {'Sub':>5} {'Ins':>5} {'Del':>5}")
        print(f"  {'─' * 35} {'─' * 8} {'─' * 6} {'─' * 6} {'─' * 5} {'─' * 5} {'─' * 5}")
        for e in sorted(per_file, key=lambda x: x["wer"], reverse=True):
            print(f"  {e['file']:<35} {e['wer']:>7.1%} "
                  f"{e['ref_words']:>6} {e['hyp_words']:>6} "
                  f"{e['substitutions']:>5} {e['insertions']:>5} {e['deletions']:>5}")

        all_results[hyp_model] = {
            "aggregate_wer": agg["wer"],
            "files_compared": len(per_file),
            "per_file": per_file,
        }

    # Save
    out_path = os.path.join(transcripts_dir, "wer_cross_comparison.json")
    with open(out_path, "w") as f:
        json.dump({
            "reference_model": ref_model,
            "word_counts": word_counts,
            "comparisons": {k: {"aggregate_wer": v["aggregate_wer"], "files": v["files_compared"]}
                            for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\n💾 Saved to: {out_path}")


# ════════════════════════════════════════════════
# Mode 2: Compare vs gold-standard transcripts
# ════════════════════════════════════════════════

def compare_gold(transcripts_dir: str, gold_dir: str):
    """Compare all STT models against gold-standard .txt transcripts."""
    models = load_transcripts_from_folders(transcripts_dir)

    gold_texts = {}
    for f in os.listdir(gold_dir):
        if f.endswith(".txt"):
            with open(os.path.join(gold_dir, f)) as fh:
                gold_texts[Path(f).stem] = fh.read().strip()

    if not gold_texts:
        print(f"❌ No .txt files in {gold_dir}")
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"📊 WER vs GOLD-STANDARD TRANSCRIPTS")
    print(f"{'═' * 70}")
    print(f"  Gold transcripts: {len(gold_texts)} files from {gold_dir}\n")

    for model_name in sorted(models.keys()):
        per_file = []
        all_ref = []
        all_hyp = []

        for stem, gold_text in gold_texts.items():
            hyp_text = models[model_name].get(stem, "")
            if not gold_text or not hyp_text:
                continue
            metrics = compute_wer_detailed(gold_text, hyp_text)
            per_file.append({"file": stem, **metrics})
            all_ref.append(gold_text)
            all_hyp.append(hyp_text)

        if not all_ref:
            print(f"  {model_name}: no matching gold files found")
            continue

        agg = compute_wer_detailed(" ".join(all_ref), " ".join(all_hyp))

        print(f"  {'━' * 65}")
        print(f"  {model_name}")
        print(f"  {'━' * 65}")
        print(f"  Aggregate WER: {agg['wer']:.1%}  |  "
              f"Sub: {agg['substitutions']}  Ins: {agg['insertions']}  Del: {agg['deletions']}")

        print(f"\n  {'File':<35} {'WER':>8} {'Ref':>6} {'Hyp':>6}")
        print(f"  {'─' * 35} {'─' * 8} {'─' * 6} {'─' * 6}")
        for e in sorted(per_file, key=lambda x: x["wer"], reverse=True):
            print(f"  {e['file']:<35} {e['wer']:>7.1%} {e['ref_words']:>6} {e['hyp_words']:>6}")
        print()


# ════════════════════════════════════════════════
# Mode 3: JSON field comparison (extraction vs ground truth)
# ════════════════════════════════════════════════

def json_compare(extractions_dir: str, ground_truth_dir: str):
    """
    Compare extracted JSON fields against ground truth JSON.
    This is NOT WER — it's field-level accuracy.
    Shows per-field match/mismatch for each model.
    """
    import re

    def norm(val):
        if val is None:
            return ""
        return re.sub(r'\s+', ' ', str(val).strip().lower())

    # Load ground truth
    gt_data = {}
    for f in os.listdir(ground_truth_dir):
        if f.endswith(".json"):
            with open(os.path.join(ground_truth_dir, f)) as fh:
                data = json.load(fh)
            gt_data[f] = data.get("flat_fields", {})

    # Discover extraction models
    model_dirs = sorted([
        d.name for d in os.scandir(extractions_dir)
        if d.is_dir()
    ])

    if not model_dirs:
        print(f"❌ No model folders in {extractions_dir}")
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"📊 FIELD ACCURACY: EXTRACTED JSON vs GROUND TRUTH JSON")
    print(f"{'═' * 70}")
    print(f"  Ground truth files: {len(gt_data)}")
    print(f"  Extraction models: {', '.join(model_dirs)}\n")

    # Enriched manifest to find gt_file for each extraction
    enriched_path = os.path.join(extractions_dir, "enriched_manifest.json")
    if os.path.exists(enriched_path):
        with open(enriched_path) as f:
            enriched = json.load(f)
    else:
        enriched = []

    # Build lookup: extraction key → ground truth filename
    key_to_gt = {}
    for entry in enriched:
        key = f"{entry['rec_id']}_{entry['form_type']}"
        gt_file = entry.get("ground_truth_file")
        if gt_file:
            key_to_gt[key] = gt_file

    summary_table = []

    for model_name in model_dirs:
        model_dir = os.path.join(extractions_dir, model_name)

        total_fields = 0
        exact_match = 0
        partial_match = 0
        missed = 0
        skipped = 0
        failed_extractions = 0

        per_file_results = []

        for f in sorted(os.listdir(model_dir)):
            if not f.endswith(".json"):
                continue

            key = Path(f).stem

            with open(os.path.join(model_dir, f)) as fh:
                ext_data = json.load(fh)

            if not ext_data.get("success", False):
                failed_extractions += 1
                continue

            # Find ground truth
            gt_file = key_to_gt.get(key) or ext_data.get("ground_truth_file")
            if not gt_file or gt_file not in gt_data:
                continue

            gt_fields = gt_data[gt_file]
            ext_json = ext_data.get("extraction", {})

            if not ext_json or "sections" not in ext_json:
                failed_extractions += 1
                continue

            # Build flat extracted fields
            ext_flat = {}
            for sec_fields in ext_json.get("sections", {}).values():
                if not isinstance(sec_fields, dict):
                    continue
                for fn, fd in sec_fields.items():
                    if isinstance(fd, dict):
                        ext_flat[norm(fn)] = norm(fd.get("value", ""))
                    else:
                        ext_flat[norm(fn)] = norm(fd)

            file_exact = 0
            file_total = 0

            for field_name, gt_val in gt_fields.items():
                gt_norm = norm(gt_val)
                ext_norm = ext_flat.get(norm(field_name), "")

                if ext_norm in ("not_mentioned", "uncertain", "n/a", "not mentioned", ""):
                    skipped += 1
                    continue

                total_fields += 1
                file_total += 1

                if gt_norm == ext_norm:
                    exact_match += 1
                    file_exact += 1
                elif gt_norm in ext_norm or ext_norm in gt_norm:
                    partial_match += 1
                else:
                    missed += 1

            if file_total > 0:
                file_acc = file_exact / file_total * 100
                per_file_results.append((key, file_exact, file_total, file_acc))

        # Print model results
        acc = (exact_match + partial_match * 0.5) / total_fields * 100 if total_fields > 0 else 0
        exact_pct = exact_match / total_fields * 100 if total_fields > 0 else 0

        print(f"  {'━' * 65}")
        print(f"  {model_name}")
        print(f"  {'━' * 65}")
        print(f"  Accuracy:    {acc:.1f}%  (exact: {exact_pct:.1f}%)")
        print(f"  Exact:       {exact_match}/{total_fields}")
        print(f"  Partial:     {partial_match}/{total_fields}")
        print(f"  Missed:      {missed}/{total_fields}")
        print(f"  Skipped:     {skipped} (NOT_MENTIONED)")
        print(f"  Failed JSON: {failed_extractions}")

        if per_file_results:
            print(f"\n  {'File':<35} {'Exact':>7} {'Total':>7} {'Acc':>8}")
            print(f"  {'─' * 35} {'─' * 7} {'─' * 7} {'─' * 8}")
            for key, ex, tot, a in sorted(per_file_results, key=lambda x: x[3]):
                print(f"  {key:<35} {ex:>7} {tot:>7} {a:>7.1f}%")

        print()

        summary_table.append({
            "model": model_name,
            "accuracy": round(acc, 1),
            "exact_match": exact_match,
            "partial_match": partial_match,
            "missed": missed,
            "total_fields": total_fields,
            "failed_extractions": failed_extractions,
        })

    # Summary comparison
    if len(summary_table) > 1:
        print(f"  {'━' * 65}")
        print(f"  SUMMARY")
        print(f"  {'━' * 65}")
        print(f"  {'Model':<25} {'Accuracy':>10} {'Exact':>10} {'Missed':>10} {'Fields':>8}")
        print(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")
        for row in sorted(summary_table, key=lambda x: x["accuracy"], reverse=True):
            print(f"  {row['model']:<25} {row['accuracy']:>9.1f}% "
                  f"{row['exact_match']:>10} {row['missed']:>10} {row['total_fields']:>8}")

    # Save
    out_path = os.path.join(extractions_dir, "json_field_comparison.json")
    with open(out_path, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\n💾 Saved to: {out_path}")


# ════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WER & field accuracy checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --cross-compare                     # Compare STT models vs each other
  %(prog)s --gold-dir data/gold_transcripts/   # Compare STT models vs gold .txt files
  %(prog)s --json-compare                      # Compare extraction JSON vs ground truth
  %(prog)s --ref "text a" --hyp "text b"       # Quick single comparison
  %(prog)s --all                               # Run everything
        """,
    )

    parser.add_argument("--cross-compare", action="store_true",
                        help="Cross-compare all STT models (transcript WER)")
    parser.add_argument("--gold-dir", type=str,
                        help="Compare STT models vs gold-standard .txt transcripts")
    parser.add_argument("--json-compare", action="store_true",
                        help="Compare extracted JSON fields vs ground truth JSON")
    parser.add_argument("--all", action="store_true",
                        help="Run both cross-compare and json-compare")
    parser.add_argument("--ref", type=str, help="Reference text (quick mode)")
    parser.add_argument("--hyp", type=str, help="Hypothesis text (quick mode)")

    args = parser.parse_args()

    # Import paths
    try:
        from config import TRANSCRIPTS_DIR, EXTRACTIONS_DIR, GROUND_TRUTH_DIR
    except ImportError:
        TRANSCRIPTS_DIR = "data/transcripts"
        EXTRACTIONS_DIR = "data/extractions"
        GROUND_TRUTH_DIR = "data/ground_truth"

    if args.all:
        cross_compare(TRANSCRIPTS_DIR)
        print("\n\n")
        json_compare(EXTRACTIONS_DIR, GROUND_TRUTH_DIR)

    elif args.cross_compare:
        cross_compare(TRANSCRIPTS_DIR)

    elif args.gold_dir:
        compare_gold(TRANSCRIPTS_DIR, args.gold_dir)

    elif args.json_compare:
        json_compare(EXTRACTIONS_DIR, GROUND_TRUTH_DIR)

    elif args.ref and args.hyp:
        metrics = compute_wer_detailed(args.ref, args.hyp)
        print(f"\n{'═' * 50}")
        print(f"📊 WER: {metrics['wer']:.1%}")
        print(f"   Ref words: {metrics['ref_words']}  |  Hyp words: {metrics['hyp_words']}")
        print(f"   Hits: {metrics['hits']}  Sub: {metrics['substitutions']}  "
              f"Ins: {metrics['insertions']}  Del: {metrics['deletions']}")

    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python wer_checker.py --all              # Everything")
        print("   python wer_checker.py --cross-compare    # Just STT WER")
        print("   python wer_checker.py --json-compare     # Just field accuracy")


if __name__ == "__main__":
    main()
