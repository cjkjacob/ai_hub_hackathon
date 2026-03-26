#!/usr/bin/env python3
"""
05_demo.py — Live Gradio demo: Upload audio → see form auto-populate.

Features:
  - Accepts any audio format (ogg, m4a, mp3, wav, flac, etc.)
  - Multilingual audio input (auto-detects language, extracts in English)
  - Optional: upload a .docx form template to auto-detect schema
  - Confidence scores per field
  - Ground truth comparison (where available)

Usage:
    python 05_demo.py
"""
import os
import sys
import json
import time
import re
import subprocess
import tempfile

import torch
import gradio as gr
from pydub import AudioSegment

from config import DATA_DIR, GROUND_TRUTH_DIR, TRANSCRIPTS_DIR, EVAL_DIR
from utils.extraction import build_extraction_prompt, extract_with_ollama
from utils.auto_schema import extract_schema_from_docx

# ════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════

BEST_STT_ENGINE = "faster-whisper"
BEST_STT_MODEL_ID = "large-v3-turbo"
BEST_STT_NAME = "whisper-large-v3-turbo"

BEST_LLM_OLLAMA_ID = "phi4:14b"
BEST_LLM_NAME = "phi4-14b"


# ════════════════════════════════════════════════
# Load project data
# ════════════════════════════════════════════════

print("🏥 Loading project data...")

schemas_path = os.path.join(DATA_DIR, "form_schemas.json")
with open(schemas_path) as f:
    default_schemas = json.load(f)

# Ground truth lookup
pairs_path = os.path.join(DATA_DIR, "pairs.json")
gt_lookup = {}
if os.path.exists(pairs_path):
    with open(pairs_path) as f:
        pairs = json.load(f)
    for p in pairs:
        if p.get("ground_truth_file"):
            gt_path = os.path.join(GROUND_TRUTH_DIR, p["ground_truth_file"])
            if os.path.exists(gt_path):
                with open(gt_path) as f:
                    gt_lookup[(p["rec_id"], p["form_type"])] = json.load(f)

print(f"   Schemas: {list(default_schemas.keys())}")
print(f"   Ground truth entries: {len(gt_lookup)}")


# ════════════════════════════════════════════════
# Load STT model
# ════════════════════════════════════════════════

print(f"🎙️  Loading STT model: {BEST_STT_NAME}...")

if BEST_STT_ENGINE == "faster-whisper":
    from faster_whisper import WhisperModel
    stt_model = WhisperModel(BEST_STT_MODEL_ID, device="cuda", compute_type="float16")

    def transcribe(wav_path):
        segments, info = stt_model.transcribe(
            wav_path,
            beam_size=5,
            language=None,  # Auto-detect language (multilingual support)
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list)
        detected_lang = info.language
        lang_prob = info.language_probability
        return text, detected_lang, lang_prob

elif BEST_STT_ENGINE == "qwen-asr":
    from qwen_asr import Qwen3ASRModel
    stt_model = Qwen3ASRModel.from_pretrained(
        BEST_STT_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=512,
    )

    def transcribe(wav_path):
        output = stt_model.transcribe(audio=wav_path, language=None)
        if isinstance(output, list) and len(output) > 0:
            text = output[0].text if hasattr(output[0], 'text') else str(output[0])
            lang = output[0].language if hasattr(output[0], 'language') else "unknown"
        else:
            text = str(output)
            lang = "unknown"
        return text.strip(), lang, 1.0

print("   ✅ STT loaded")

# Ensure Ollama is running
print(f"🧠 Checking LLM: {BEST_LLM_NAME}...")
try:
    import requests
    requests.get("http://localhost:11434/api/tags", timeout=3)
    print("   ✅ Ollama running")
except Exception:
    print("   Starting Ollama...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)


# ════════════════════════════════════════════════
# Pipeline functions
# ════════════════════════════════════════════════

CONF_ICONS = {"high": "🟢", "medium": "🟡", "low": "🔴", "unknown": "⚪"}


def detect_form_type(text):
    start = text[:500].lower()
    if any(kw in start for kw in ["initial consultation", "initial form"]):
        return "initial"
    if any(kw in start for kw in ["follow-up", "follow up", "followup"]):
        return "follow_up"
    if "triage" in start:
        return "triage"
    return "initial"


def extract_rec_id(text):
    m = re.search(r'AG[-\s]*(\d+)', text, re.IGNORECASE)
    return f"AG-{int(m.group(1)):03d}" if m else None


def run_pipeline(audio_path, template_path=None):
    """Full pipeline: audio → transcript → extraction → formatted output."""
    steps = []
    total_start = time.time()

    # Determine schemas to use
    if template_path:
        custom_schema = extract_schema_from_docx(template_path)
        detected_type = custom_schema["form_type"]
        schemas = {detected_type: custom_schema}
        steps.append(("Schema detection", 0,
                       f"Custom template: {custom_schema['total_fields']} fields, type: {detected_type}"))
    else:
        schemas = default_schemas

    # 1. Convert audio (handles ANY format via pydub/ffmpeg)
    t0 = time.time()
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    dur = len(audio) / 1000
    steps.append(("Audio conversion", round(time.time() - t0, 1), f"{dur:.1f}s → 16kHz WAV"))

    # 2. Transcribe (multilingual — auto-detects language)
    t0 = time.time()
    transcript, detected_lang, lang_prob = transcribe(tmp.name)
    steps.append(("Transcription", round(time.time() - t0, 1),
                   f"{len(transcript.split())} words, language: {detected_lang} ({lang_prob:.0%})"))

    # Clean up temp file
    try:
        os.unlink(tmp.name)
    except OSError:
        pass

    # 3. Detect form type + ID
    if template_path:
        form_type = detected_type
    else:
        form_type = detect_form_type(transcript)
    rec_id = extract_rec_id(transcript)
    steps.append(("Detection", 0, f"{form_type}, ID: {rec_id or 'N/A'}"))

    # 4. LLM extraction (uses the correct schemas — default or custom)
    t0 = time.time()
    sys_p, usr_p = build_extraction_prompt(form_type, transcript, schemas)
    result = extract_with_ollama(BEST_LLM_OLLAMA_ID, sys_p, usr_p)
    steps.append(("LLM extraction", round(time.time() - t0, 1),
                   "Success" if result["success"] else f"Failed: {result['error'][:40]}"))

    total = round(time.time() - total_start, 1)

    # Get ground truth (only for known recordings)
    gt = gt_lookup.get((rec_id, form_type)) if rec_id else None

    return {
        "steps": steps, "total_time": total, "transcript": transcript,
        "form_type": form_type, "rec_id": rec_id,
        "detected_language": detected_lang,
        "extraction": result.get("extracted_json"), "ground_truth": gt,
        "error": result.get("error"),
    }


def format_html(r):
    """Convert pipeline results to display HTML."""
    gt_flat = r["ground_truth"].get("flat_fields", {}) if r["ground_truth"] else {}
    badge_color = {"initial": "#2196F3", "follow_up": "#FF9800", "triage": "#4CAF50"}.get(r["form_type"], "#999")

    h = []

    # Timing table
    h.append("<h3>Pipeline steps</h3><table style='width:100%;border-collapse:collapse;margin-bottom:16px;'>")
    h.append("<tr style='background:var(--color-background-secondary);'>"
             "<th style='text-align:left;padding:8px;'>Step</th>"
             "<th style='text-align:right;padding:8px;'>Time</th>"
             "<th style='text-align:left;padding:8px;'>Detail</th></tr>")
    for name, t, detail in r["steps"]:
        h.append(f"<tr><td style='padding:6px 8px;'>{name}</td>"
                 f"<td style='padding:6px 8px;text-align:right;font-weight:bold;'>{t}s</td>"
                 f"<td style='padding:6px 8px;opacity:0.7;'>{detail}</td></tr>")
    h.append(f"<tr style='font-weight:bold;'><td style='padding:6px 8px;'>Total</td>"
             f"<td style='padding:6px 8px;text-align:right;'>{r['total_time']}s</td><td></td></tr>")
    h.append("</table>")

    # Badge row
    h.append(f"<span style='background:{badge_color};color:white;padding:4px 12px;"
             f"border-radius:12px;font-weight:bold;'>{r['form_type'].replace('_',' ').upper()}</span>")
    h.append(f"&nbsp;&nbsp;Recording ID: <strong>{r['rec_id'] or 'Not detected'}</strong>")
    lang = r.get("detected_language", "en")
    if lang and lang != "en":
        h.append(f"&nbsp;&nbsp;Language: <strong>{lang}</strong>")
    h.append("<br><br>")

    ext = r.get("extraction")
    if not ext or "sections" not in ext:
        h.append("<p style='color:red;'>Extraction failed.</p>")
        if r.get("error"):
            h.append(f"<p style='opacity:0.6;'>{r['error']}</p>")
        return "\n".join(h)

    # Form fields
    h.append("<h3>Extracted form data</h3>")
    for sec_name, sec_fields in ext["sections"].items():
        h.append(f"<div style='margin:12px 0 6px;padding:6px 12px;"
                 f"background:var(--color-background-secondary);"
                 f"border-left:4px solid {badge_color};font-weight:bold;'>{sec_name}</div>")
        h.append("<table style='width:100%;border-collapse:collapse;'>")
        if isinstance(sec_fields, dict):
            for fn, fd in sec_fields.items():
                val = fd.get("value", str(fd)) if isinstance(fd, dict) else str(fd)
                conf = str(fd.get("confidence", "unknown")).lower() if isinstance(fd, dict) else "unknown"
                icon = CONF_ICONS.get(conf, "⚪")
                gt_val = gt_flat.get(fn)
                gt_cell = ""
                if gt_val is not None:
                    v_lo = str(val).strip().lower()
                    g_lo = str(gt_val).strip().lower()
                    if v_lo == g_lo:
                        gt_cell = "<td style='padding:4px 8px;color:green;font-size:0.85em;'>match</td>"
                    elif v_lo in ("not_mentioned", "uncertain", ""):
                        gt_cell = "<td style='padding:4px 8px;color:gray;font-size:0.85em;'>skipped</td>"
                    else:
                        gt_cell = (f"<td style='padding:4px 8px;color:red;font-size:0.85em;'"
                                   f" title='GT: {str(gt_val)[:80]}'>mismatch</td>")
                disp = str(val)[:120] + ("..." if len(str(val)) > 120 else "")
                h.append(
                    f"<tr style='border-bottom:1px solid var(--color-border-tertiary);'>"
                    f"<td style='padding:6px 8px;width:35%;font-weight:500;'>{fn}</td>"
                    f"<td style='padding:6px 8px;'>{disp}</td>"
                    f"<td style='padding:4px;text-align:center;'>{icon}</td>"
                    f"{gt_cell}</tr>")
        h.append("</table>")

    h.append("<div style='margin-top:16px;font-size:0.85em;opacity:0.6;'>"
             "Confidence: 🟢 high 🟡 medium 🔴 low ⚪ unknown</div>")
    return "\n".join(h)


def process(audio_file, template_file=None):
    """Main Gradio handler."""
    if audio_file is None:
        return "Upload an audio file first.", "", ""

    # Handle template upload
    template_path = None
    if template_file is not None:
        # Gradio returns a file path string for gr.File
        template_path = template_file

    r = run_pipeline(audio_file, template_path=template_path)
    return r["transcript"] or "No transcript.", format_html(r), f"Done in {r['total_time']}s"


# ════════════════════════════════════════════════
# Gradio UI
# ════════════════════════════════════════════════

print("\n🚀 Building demo UI...")

with gr.Blocks(title="Patients Over Paperwork", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # Patients Over Paperwork
    ### Automating clinical workflows with speech AI
    **Pipeline:** {BEST_STT_NAME} → {BEST_LLM_NAME} → Structured clinical form

    Upload a consultation recording in **any audio format** and **any language**.
    Optionally upload a .docx form template to use a custom schema.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.File(
                label="Upload consultation audio (.ogg, .m4a, .mp3, .wav, etc.)",
                file_types=[".ogg", ".m4a", ".mp3", ".wav", ".flac", ".aac", ".wma", ".webm"],
                type="filepath",
            )
            template_in = gr.File(
                label="Form template (.docx) — optional",
                file_types=[".docx"],
                type="filepath",
            )
            btn = gr.Button("Process consultation", variant="primary", size="lg")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            with gr.Tab("Extracted form"):
                form_out = gr.HTML()
            with gr.Tab("Transcript"):
                transcript_out = gr.Textbox(
                    label="Raw transcript",
                    lines=15,
                    interactive=False,
                )

    gr.Markdown(f"""
    ---
    **Privacy:** All processing is local — {BEST_STT_NAME} and {BEST_LLM_NAME} run on-device.
    Zero data leaves this machine. GDPR/DTAC compliant by design.

    **Multilingual:** Audio in any language is auto-detected and transcribed.
    Form fields are always extracted in English.

    **Custom forms:** Upload any .docx clinical form as a template —
    the system auto-detects the schema and extracts matching fields.
    """)

    btn.click(
        fn=process,
        inputs=[audio_in, template_in],
        outputs=[transcript_out, form_out, status],
    )

print("✅ Ready! Launching...")
demo.launch(share=True, show_error=True)
