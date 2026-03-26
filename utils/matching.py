"""
Audio ↔ Form matching — extracts recording IDs from messy filenames
and builds 1:1 pairs of (audio, form) keyed by (rec_id, form_type).
"""
import os
import re


def extract_id_from_filename(filename: str) -> str | None:
    """
    Extract and normalise an AG-XXX recording ID from any filename.

    Handles: 'AG-001.ogg', 'Ag 1.ogg', 'Ag21.ogg',
    'Follow Up Consultation Form AG-014.m4a', etc.
    """
    name = os.path.splitext(filename)[0]

    # Pattern 1: AG-001, AG-021, ag-14, etc (with flexible separators)
    match = re.search(r'AG[-–—_\s]*(\d+)', name, re.IGNORECASE)
    if match:
        return f"AG-{int(match.group(1)):03d}"

    # Pattern 2: "Ag" followed by number
    match = re.search(r'Ag\s*(\d+)', name, re.IGNORECASE)
    if match:
        return f"AG-{int(match.group(1)):03d}"

    # Pattern 3: Bare number
    match = re.fullmatch(r'(\d+)', name.strip())
    if match:
        return f"AG-{int(match.group(1)):03d}"

    return None


def extract_form_type_from_filename(filename: str) -> str:
    """
    Detect form type from filename keywords.
    Returns: 'initial', 'triage', 'follow_up', or 'unspecified'
    """
    name_lower = os.path.splitext(filename)[0].lower()

    if any(kw in name_lower for kw in ['follow up', 'follow_up', 'followup', 'follow-up']):
        return 'follow_up'
    if 'triage' in name_lower:
        return 'triage'
    if 'initial' in name_lower:
        return 'initial'
    return 'unspecified'


def normalise_rec_id(raw_id: str) -> str | None:
    """Normalise any recording ID string to AG-XXX format."""
    if not raw_id or raw_id == "unknown":
        return None
    match = re.search(r'(\d+)', raw_id)
    if match:
        return f"AG-{int(match.group(1)):03d}"
    return raw_id.upper()


def build_dataset_map(audio_manifest: list, ground_truths: dict) -> dict:
    """
    Build a mapping of recording_id → {audio: {type: [files]}, forms: {type: data}}.

    Args:
        audio_manifest: list of dicts from audio conversion
        ground_truths: dict of {docx_filename: parsed_form_data}

    Returns:
        dataset_map dict
    """
    dataset_map = {}

    # Add audio files
    for audio_info in audio_manifest:
        rec_id = extract_id_from_filename(audio_info["original_file"])
        form_type = extract_form_type_from_filename(audio_info["original_file"])
        if not rec_id:
            continue
        if rec_id not in dataset_map:
            dataset_map[rec_id] = {"audio": {}, "forms": {}}
        if form_type not in dataset_map[rec_id]["audio"]:
            dataset_map[rec_id]["audio"][form_type] = []
        dataset_map[rec_id]["audio"][form_type].append(audio_info)

    # Add forms
    for filename, data in ground_truths.items():
        rec_id = normalise_rec_id(data["metadata"]["recording_id"])
        if not rec_id:
            rec_id = extract_id_from_filename(filename)
        if not rec_id:
            continue
        form_type = data["metadata"]["form_type"]
        if rec_id not in dataset_map:
            dataset_map[rec_id] = {"audio": {}, "forms": {}}
        dataset_map[rec_id]["forms"][form_type] = {
            "filename": filename,
            "form_type": form_type,
            "num_fields": len(data["flat_fields"]),
        }

    # Auto-resolve unspecified audio by elimination
    resolved = 0
    for rec_id, entry in dataset_map.items():
        if "unspecified" not in entry["audio"]:
            continue
        unspecified = entry["audio"]["unspecified"]
        matched_types = set(ft for ft in entry["audio"] if ft != "unspecified")
        unmatched_forms = [ft for ft in entry["forms"] if ft not in matched_types]

        if len(unspecified) == 1 and len(unmatched_forms) == 1:
            target = unmatched_forms[0]
            entry["audio"].setdefault(target, []).extend(unspecified)
            del entry["audio"]["unspecified"]
            resolved += 1
        elif len(unspecified) == len(unmatched_forms) and len(unspecified) > 1:
            for audio_item, target in zip(unspecified, sorted(unmatched_forms)):
                entry["audio"].setdefault(target, []).append(audio_item)
            del entry["audio"]["unspecified"]
            resolved += len(unspecified)

    return dataset_map


def build_pairs(dataset_map: dict) -> list:
    """
    Build a flat list of 1:1 (audio, form) pairs for processing.
    Each pair = one unit of work through the pipeline.
    """
    pairs = []
    for rec_id in sorted(dataset_map):
        entry = dataset_map[rec_id]
        all_types = set(list(entry["audio"]) + list(entry["forms"]))

        for ft in sorted(all_types):
            if ft == "unspecified":
                for audio_item in entry["audio"].get(ft, []):
                    pairs.append({
                        "rec_id": rec_id,
                        "form_type": "unspecified",
                        "wav_file": audio_item["wav_file"],
                        "ground_truth_file": None,
                        "status": "needs_resolution",
                    })
                continue

            has_audio = ft in entry["audio"] and len(entry["audio"][ft]) > 0
            has_form = ft in entry["forms"]

            if has_audio and has_form:
                audio_item = entry["audio"][ft][0]
                gt_json = entry["forms"][ft]["filename"].replace(".docx", ".json")
                pairs.append({
                    "rec_id": rec_id,
                    "form_type": ft,
                    "wav_file": audio_item["wav_file"],
                    "ground_truth_file": gt_json,
                    "status": "matched",
                })
            elif has_audio:
                audio_item = entry["audio"][ft][0]
                pairs.append({
                    "rec_id": rec_id,
                    "form_type": ft,
                    "wav_file": audio_item["wav_file"],
                    "ground_truth_file": None,
                    "status": "missing_form",
                })
            elif has_form:
                gt_json = entry["forms"][ft]["filename"].replace(".docx", ".json")
                pairs.append({
                    "rec_id": rec_id,
                    "form_type": ft,
                    "wav_file": None,
                    "ground_truth_file": gt_json,
                    "status": "missing_audio",
                })

    return pairs
