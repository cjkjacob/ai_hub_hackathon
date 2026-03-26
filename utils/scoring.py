"""
Evaluation scoring — field-by-field comparison of extracted vs ground truth.
"""
import re
from config import CRITICAL_FIELD_KEYWORDS


def normalise_field_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'[?!.,;:\-—–\s]+', ' ', name).strip()
    return name


def normalise_value(val) -> str:
    if val is None:
        return ""
    val = str(val).strip()
    return re.sub(r'\s+', ' ', val)


def values_match(gt_value: str, extracted_value: str) -> tuple:
    """
    Compare ground truth to extracted value.
    Returns (match_type, score):
      "exact"   1.0 | "partial" 0.5 | "miss" 0.0 | "skip" None
    """
    gt = normalise_value(gt_value).lower()
    ext = normalise_value(extracted_value).lower()

    if ext in ("not_mentioned", "uncertain", "n/a", "not mentioned", ""):
        return "skip", None
    if gt == ext:
        return "exact", 1.0

    # Yes/No — strict
    if gt in ("yes", "no"):
        if ext in ("yes", "no"):
            return ("exact", 1.0) if gt == ext else ("miss", 0.0)
        if "yes" in ext and gt == "yes":
            return "exact", 1.0
        if "no" in ext and gt == "no" and "yes" not in ext:
            return "exact", 1.0
        return "miss", 0.0

    # Numeric / dosage — wrong numbers = miss
    gt_nums = set(re.findall(r'\d+(?:\.\d+)?', gt))
    ext_nums = set(re.findall(r'\d+(?:\.\d+)?', ext))
    if gt_nums:
        if gt_nums == ext_nums:
            gt_words = set(re.findall(r'[a-z]{3,}', gt))
            ext_words = set(re.findall(r'[a-z]{3,}', ext))
            overlap = gt_words & ext_words
            return ("exact", 1.0) if len(overlap) >= len(gt_words) * 0.5 else ("partial", 0.5)
        elif gt_nums & ext_nums:
            return "partial", 0.5
        else:
            return "miss", 0.0

    # Long text — word overlap
    if len(gt) > 20:
        gt_words = set(re.findall(r'[a-z]{3,}', gt))
        ext_words = set(re.findall(r'[a-z]{3,}', ext))
        if not gt_words:
            return "miss", 0.0
        ratio = len(gt_words & ext_words) / len(gt_words)
        if ratio >= 0.7:
            return "exact", 1.0
        elif ratio >= 0.3:
            return "partial", 0.5
        return "miss", 0.0

    # Short text — containment
    if gt in ext or ext in gt:
        return "exact", 1.0

    gt_words = set(gt.split())
    ext_words = set(ext.split())
    if gt_words and ext_words:
        if len(gt_words & ext_words) / len(gt_words) >= 0.5:
            return "partial", 0.5

    return "miss", 0.0


def find_extracted_value(extracted_json: dict, target_field: str):
    """Search extraction JSON for a field by normalised name."""
    if not extracted_json or "sections" not in extracted_json:
        return None, None

    target = normalise_field_name(target_field)

    for section_fields in extracted_json.get("sections", {}).values():
        if not isinstance(section_fields, dict):
            continue
        for field_name, field_data in section_fields.items():
            if normalise_field_name(field_name) == target:
                if isinstance(field_data, dict):
                    return field_data.get("value", str(field_data)), field_data.get("confidence", "unknown")
                return str(field_data), "unknown"

    return None, None


def is_critical_field(field_name: str) -> bool:
    name_lower = field_name.lower()
    return any(kw in name_lower for kw in CRITICAL_FIELD_KEYWORDS)


def compute_accuracy(results: list) -> dict:
    """Compute accuracy from a list of eval result dicts."""
    evaluable = [r for r in results if r["match_type"] != "skip"]
    if not evaluable:
        return {"accuracy": 0, "exact_rate": 0, "partial_rate": 0, "miss_rate": 0, "n": 0}

    n = len(evaluable)
    total_score = sum(r["score"] for r in evaluable)
    exact = sum(1 for r in evaluable if r["match_type"] == "exact")
    partial = sum(1 for r in evaluable if r["match_type"] == "partial")
    miss = n - exact - partial

    return {
        "accuracy": round(total_score / n * 100, 1),
        "exact_rate": round(exact / n * 100, 1),
        "partial_rate": round(partial / n * 100, 1),
        "miss_rate": round(miss / n * 100, 1),
        "n": n,
    }
