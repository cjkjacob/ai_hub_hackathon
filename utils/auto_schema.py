"""
Auto-schema detection — extracts form schema from any .docx template.

Instead of hardcoding schemas for 3 form types, this module:
1. Takes any .docx clinical form as input
2. Auto-detects sections (1-column tables = headers)
3. Auto-detects fields (2-column tables = key-value pairs)
4. Outputs a JSON schema ready for the LLM extraction prompt

This makes the pipeline work with ANY clinical form — not just
Initial, Follow-Up, and Triage from the Armour Group dataset.
"""
import os
import json
import re
from docx import Document


def extract_schema_from_docx(filepath: str) -> dict:
    """
    Automatically extract a form schema from any .docx clinical form.

    Detects the pattern:
      - 1-column tables = section headers
      - 2-column tables = field_name | field_value pairs

    Returns:
    {
        "form_type": "auto_detected_type",
        "source_file": "filename.docx",
        "sections": {
            "SECTION NAME": ["field1", "field2", ...],
            ...
        },
        "total_fields": 43
    }
    """
    doc = Document(filepath)
    tables = doc.tables

    # Try to detect form type from header
    header_text = ""
    if tables:
        header_text = tables[0].rows[0].cells[0].text.strip()

    form_type = _detect_form_type(header_text)

    schema = {
        "form_type": form_type,
        "source_file": os.path.basename(filepath),
        "sections": {},
    }

    current_section = "HEADER"
    skip_prefixes = ("HOW TO USE", "End of", "elios")

    for table in tables:
        rows = table.rows
        if not rows:
            continue
        num_cols = len(rows[0].cells)

        if num_cols == 1:
            cell_text = rows[0].cells[0].text.strip()
            if any(cell_text.startswith(s) for s in skip_prefixes):
                continue
            current_section = cell_text
            if current_section not in schema["sections"]:
                schema["sections"][current_section] = []

        elif num_cols == 2:
            if current_section not in schema["sections"]:
                schema["sections"][current_section] = []
            for row in rows:
                field_name = row.cells[0].text.strip().replace("**", "")
                if field_name and field_name not in schema["sections"][current_section]:
                    schema["sections"][current_section].append(field_name)

    schema["total_fields"] = sum(len(f) for f in schema["sections"].values())
    return schema


def extract_schemas_from_directory(directory: str) -> dict:
    """
    Scan a directory of .docx forms and build schemas per form type.

    Returns a merged schema dict like form_schemas.json:
    {
        "initial": {"sections": {...}},
        "follow_up": {"sections": {...}},
        "triage": {"sections": {...}},
        "custom_form_name": {"sections": {...}},
    }
    """
    schemas = {}

    docx_files = sorted([f for f in os.listdir(directory) if f.endswith('.docx')])

    for filename in docx_files:
        filepath = os.path.join(directory, filename)
        try:
            schema = extract_schema_from_docx(filepath)
            form_type = schema["form_type"]

            if form_type not in schemas:
                schemas[form_type] = {"sections": {}}

            # Merge: take the union of all fields across forms of the same type
            for section, fields in schema["sections"].items():
                if section not in schemas[form_type]["sections"]:
                    schemas[form_type]["sections"][section] = []
                for field in fields:
                    if field not in schemas[form_type]["sections"][section]:
                        schemas[form_type]["sections"][section].append(field)

        except Exception as e:
            print(f"  ⚠️  Could not parse {filename}: {e}")

    return schemas


def schema_to_prompt_description(schema: dict) -> str:
    """
    Convert a schema dict into a human-readable description for the LLM prompt.
    Works with both auto-detected and manually defined schemas.

    Args:
        schema: either a full schema dict with "sections" key,
                or just the sections dict directly
    """
    sections = schema.get("sections", schema) if isinstance(schema, dict) else schema

    lines = []
    for section_name, fields in sections.items():
        if isinstance(fields, list):
            lines.append(f"\n### {section_name}")
            for field in fields:
                lines.append(f'  - "{field}"')
        elif isinstance(fields, dict):
            # Handle case where fields is a dict of field_name: value
            lines.append(f"\n### {section_name}")
            for field in fields.keys():
                lines.append(f'  - "{field}"')

    return "\n".join(lines)


def detect_form_type_from_schema(schema: dict) -> str:
    """
    Guess the form type from section names in the schema.
    Falls back to the auto-detected type from the header.
    """
    all_sections = " ".join(schema.get("sections", {}).keys()).lower()

    if "triage decision" in all_sections or ("triage" in all_sections and "clinical assessment" in all_sections):
        return "triage"
    if "follow up" in all_sections or "appointment details" in all_sections or "previous medication" in all_sections:
        return "follow_up"
    if "cannabis exposure" in all_sections or "patient capacity" in all_sections:
        return "initial"

    return schema.get("form_type", "unknown")


def _detect_form_type(text: str) -> str:
    """Detect form type from header text."""
    text_lower = text.lower()
    if "initial" in text_lower:
        return "initial"
    elif "triage" in text_lower:
        return "triage"
    elif "follow" in text_lower:
        return "follow_up"
    return "unknown"


# ════════════════════════════════════════════════════════════
# Convenience: generate schema from a single form for ad-hoc use
# ════════════════════════════════════════════════════════════

def generate_schema_json(docx_path: str, output_path: str = None) -> dict:
    """
    One-shot: parse a .docx form and optionally save the schema as JSON.

    Usage:
        schema = generate_schema_json("my_form.docx", "my_schema.json")
    """
    schema = extract_schema_from_docx(docx_path)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        print(f"Schema saved to {output_path}")
        print(f"  Form type: {schema['form_type']}")
        print(f"  Sections: {len(schema['sections'])}")
        print(f"  Fields: {schema['total_fields']}")

    return schema


if __name__ == "__main__":
    """
    CLI usage:
        python utils/auto_schema.py path/to/form.docx
        python utils/auto_schema.py path/to/forms_directory/
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python utils/auto_schema.py <form.docx>          # single form")
        print("  python utils/auto_schema.py <directory/>          # all forms in dir")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isfile(path) and path.endswith(".docx"):
        schema = extract_schema_from_docx(path)
        print(json.dumps(schema, indent=2))
    elif os.path.isdir(path):
        schemas = extract_schemas_from_directory(path)
        print(json.dumps(schemas, indent=2))
    else:
        print(f"❌ Not a .docx file or directory: {path}")