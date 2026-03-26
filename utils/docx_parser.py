"""
DOCX Form Parser — extracts structured field:value pairs from Armour Group clinical forms.

The forms use a consistent pattern:
  - 1-column tables = section headers (e.g., "PATIENT DETAILS")
  - 2-column tables = field label (col 0) | value (col 1)
"""
import re
import os
from docx import Document


def detect_form_type(text: str) -> str:
    """Detect form type from header text."""
    text_lower = text.lower()
    if "initial" in text_lower:
        return "initial"
    elif "triage" in text_lower:
        return "triage"
    elif "follow" in text_lower:
        return "follow_up"
    return "unknown"


def extract_recording_id(text: str) -> str:
    """Extract recording ID (e.g., AG-001) from text."""
    match = re.search(r'(AG-\d+)', text)
    return match.group(1) if match else "unknown"


def extract_clinician(text: str) -> str:
    """Extract clinician info from header text."""
    match = re.search(r'Clinician[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "unknown"


def parse_docx_form(filepath: str) -> dict:
    """
    Parse an Armour Group clinical form .docx into structured JSON.

    Returns a dict with:
      - metadata: recording_id, form_type, clinician, source_file
      - sections: {section_name: {field_name: value}}
      - flat_fields: {field_name: value} (all fields flattened)
    """
    doc = Document(filepath)
    tables = doc.tables

    # Extract metadata from header
    header_text = tables[0].rows[0].cells[0].text if tables else ""
    form_type = detect_form_type(header_text)
    recording_id = extract_recording_id(header_text)
    clinician = extract_clinician(header_text)

    result = {
        "metadata": {
            "recording_id": recording_id,
            "form_type": form_type,
            "clinician": clinician,
            "source_file": os.path.basename(filepath),
        },
        "sections": {},
        "flat_fields": {},
    }

    current_section = "HEADER"

    for table in tables:
        rows = table.rows
        num_cols = len(rows[0].cells) if rows else 0

        if num_cols == 1:
            cell_text = rows[0].cells[0].text.strip()
            # Skip instruction/header/footer rows
            if any(cell_text.startswith(skip) for skip in ["HOW TO USE", "End of", "elios"]):
                continue
            current_section = cell_text
            if current_section not in result["sections"]:
                result["sections"][current_section] = {}

        elif num_cols == 2:
            if current_section not in result["sections"]:
                result["sections"][current_section] = {}
            for row in rows:
                field_name = row.cells[0].text.strip().replace("**", "")
                field_value = row.cells[1].text.strip()
                result["sections"][current_section][field_name] = field_value
                result["flat_fields"][field_name] = field_value

    return result
