# AI Hub Hackathon — Patient Transcription Project

An end-to-end pipeline that converts clinical consultation audio recordings into structured medical forms using locally-hosted speech-to-text and large language models. Zero data leaves the machine.

**Pipeline:** Audio → STT (Whisper / Qwen-ASR) → Transcript → LLM (Phi-4 / Llama / Qwen / MedGemma) → Populated JSON Form

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Dataset](#dataset)
6. [Project Structure](#project-structure)
7. [Running the Pipeline](#running-the-pipeline)
8. [Pipeline Steps in Detail](#pipeline-steps-in-detail)
9. [WER & Accuracy Checker](#wer--accuracy-checker)
10. [Live Demo](#live-demo)
11. [Benchmark Results](#benchmark-results)
12. [Configuration Reference](#configuration-reference)
13. [MedGemma Installation](#medgemma-installation)
14. [Troubleshooting](#troubleshooting)

---

## Overview

Clinical consultations generate hours of audio that must be transcribed and manually entered into structured forms — a slow, error-prone process that takes clinicians away from patient care.

This project automates that workflow:

1. **Speech-to-Text**: Transcribes consultation audio using multiple STT models (benchmarked for accuracy and speed)
2. **Form Detection**: Auto-detects the form type (Initial, Follow-Up, or Triage) from the transcript content
3. **LLM Extraction**: Uses locally-hosted LLMs to extract structured field values from the transcript, guided by the form schema
4. **Evaluation**: Scores extracted fields against hand-filled ground truth forms (exact/partial/miss at both overall and critical-field level)
5. **Live Demo**: A Gradio web UI where you upload any audio file (in any language) and optionally a .docx form template, and watch the form auto-populate in real time

All models run locally on a single GPU. No cloud APIs, no data exfiltration, fully GDPR-compliant by design.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Audio File  │────▶│  STT Model       │────▶│  LLM + Schema    │────▶│  JSON Form  │
│  .ogg .m4a   │     │  (faster-whisper  │     │  (Ollama local)  │     │  Structured │
│  .mp3 .wav   │     │   or Qwen3-ASR)  │     │  Phi-4 / Llama   │     │  Output     │
└─────────────┘     └──────────────────┘     └──────────────────┘     └─────────────┘
       │                     │                        │
       │              Auto-detects               Few-shot prompted
       │              language                   per form type
       │
  pydub/ffmpeg
  → 16kHz mono WAV
```

**One pipeline, three schemas.** Form type is detected from transcript content (clinicians read the form title at the start of each recording). The system also supports custom schemas — upload any .docx form template and it auto-extracts the field structure.

---

## Prerequisites

- **OS**: WSL2 on Windows 11 (tested) or native Linux
- **GPU**: NVIDIA GPU with 12+ GB VRAM (tested on RTX 5070 Ti Laptop, 12.8 GB)
- **NVIDIA Drivers**: Installed on the Windows host (not inside WSL2)
- **Python**: 3.12+ (required for Blackwell SM_120 GPUs; 3.10+ for older GPUs)
- **Disk Space**: ~30 GB (models + audio + virtual environment)
- **Ollama**: For running LLMs locally

### GPU Compatibility Notes

| GPU Architecture | PyTorch Version Required |
|---|---|
| Ampere (RTX 30xx) | Stable PyTorch + CUDA 11.8/12.1 |
| Ada Lovelace (RTX 40xx) | Stable PyTorch + CUDA 12.1 |
| Blackwell (RTX 50xx) | **PyTorch Nightly + CUDA 12.8** |

Blackwell GPUs (SM_120) are not supported by stable PyTorch. The setup script handles this automatically.

---

## Setup

### Automated Setup

```bash
chmod +x setup.sh && ./setup.sh
```

This script:
1. Checks NVIDIA GPU access from WSL2
2. Installs system dependencies (ffmpeg, python3, git, build tools)
3. Creates a Python virtual environment
4. Installs PyTorch nightly with CUDA 12.8 (for Blackwell GPUs)
5. Installs all Python packages (faster-whisper, qwen-asr, jiwer, gradio, etc.)
6. Installs Ollama and pulls LLM models (Phi-4 14B, Qwen 2.5 7B, Llama 3.2 3B)

> **Note:** MedGemma 1.5 4B requires a separate installation from a GGUF file (not available via `ollama pull`). See the [MedGemma Installation](#medgemma-installation) section after running setup.

### Manual Setup (if setup.sh doesn't work)

```bash
# Create project directory
mkdir -p ~/project-dir && cd ~/project-dir

# Create venv
python3.12 -m venv venv && source venv/bin/activate

# PyTorch nightly (Blackwell) or stable (older GPUs)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# STT models
pip install faster-whisper qwen-asr

# Other dependencies
pip install python-docx pydub jiwer scikit-learn gradio ollama pandas tqdm requests

# Ollama (for LLMs)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull phi4:14b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b

# MedGemma 1.5 4B (medical-specialist LLM — see MedGemma section below)
pip install huggingface-hub
huggingface-cli download unsloth/medgemma-1.5-4b-it-GGUF medgemma-1.5-4b-it-Q4_K_M.gguf --local-dir ./models
cat > models/Modelfile << 'EOF'
FROM ./medgemma-1.5-4b-it-Q4_K_M.gguf
PARAMETER temperature 0.1
PARAMETER num_predict 8192
TEMPLATE """<start_of_turn>user
{{ .System }}

{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""
EOF
ollama create medgemma:4b -f models/Modelfile
```

### Verify GPU Works

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
x = torch.randn(100, 100, device='cuda')
print('GPU tensor test: PASSED')
"
```

---

## Dataset

Place all raw data files in `data/raw/`:

- **Audio files**: `.ogg`, `.m4a`, `.mp3`, `.wav`, or any ffmpeg-supported format
- **Form files**: `.docx` clinical forms (ground truth for evaluation)

The data prep script (`01_data_prep.py`) handles:
- Parsing .docx forms into structured JSON (1-column tables = section headers, 2-column tables = key-value fields)
- Converting all audio to 16kHz mono WAV
- Matching audio files to their corresponding forms using patient ID extraction
- Generating schemas, manifests, and pairs files

### ID Matching

The matching system handles messy filenames robustly:

| Audio Filename | Extracted ID | Matched Form |
|---|---|---|
| `Ag 1.ogg` | AG-001 | AG-001_Clinician-1_InitialForm.docx |
| `Ag21.ogg` | AG-021 | AG-021_Clinician-7_TriageForm.docx |
| `Follow Up Consultation Form AG-014.m4a` | AG-014 | AG-014_Clinician-5_FollowUpForm.docx |
| `Ag 16 17 18.ogg` | AG-016 | Multi-form: initial + follow_up + triage |

---

## Project Structure

```
project-root/
│
├── setup.sh                    # Automated environment setup
├── config.py                   # All paths, model configs, constants
├── README.md                   # This file
│
├── 01_data_prep.py             # Parse .docx forms, convert audio, match pairs
├── 02_transcribe.py            # Run STT models, save transcripts
├── 03_extract.py               # Run LLMs via Ollama, extract form fields
├── 04_evaluate.py              # Field-by-field accuracy scoring
├── 05_demo.py                  # Gradio live demo
├── wer_checker.py              # Standalone WER & accuracy analysis
│
├── utils/
│   ├── __init__.py
│   ├── docx_parser.py          # .docx form → JSON (section/field detection)
│   ├── audio.py                # Any audio format → 16kHz mono WAV
│   ├── matching.py             # Robust patient ID extraction & form matching
│   ├── extraction.py           # LLM prompts with few-shot examples, Ollama API
│   ├── scoring.py              # Field matching (exact/partial/miss), accuracy
│   └── auto_schema.py          # Auto-extract schema from any .docx template
│
├── data/                       # All data files (created by pipeline)
│   ├── raw/                    # ← PUT YOUR .docx AND AUDIO FILES HERE
│   ├── ground_truth/           # Parsed JSON from .docx forms
│   │   ├── AG-001_Clinician-1_InitialForm.json
│   │   └── ...
│   ├── audio_wav/              # Converted 16kHz mono WAV files
│   │   ├── Ag 1.wav
│   │   └── ...
│   ├── transcripts/            # Per-model transcript folders
│   │   ├── whisper-large-v3-turbo/
│   │   │   ├── Ag 1.json       # {full_text, segments, timing, ...}
│   │   │   └── ...
│   │   ├── whisper-large-v3/
│   │   ├── qwen3-asr-0.6b/
│   │   ├── qwen3-asr-1.7b/
│   │   ├── final/              # Merged transcripts (all models per file)
│   │   │   ├── AG-001_initial.json
│   │   │   ├── manifest.json
│   │   │   └── ...
│   │   ├── model_benchmark_stats.json
│   │   ├── transcription_summary.json
│   │   └── wer_cross_comparison.json
│   ├── extractions/            # Per-LLM extraction folders
│   │   ├── phi4-14b/
│   │   │   ├── AG-001_initial.json
│   │   │   └── ...
│   │   ├── llama3.2-3b/
│   │   ├── qwen2.5-7b/
│   │   ├── enriched_manifest.json
│   │   ├── extraction_summary.json
│   │   └── field_accuracy_comparison.json
│   ├── audio_manifest.json     # All audio files with durations
│   ├── dataset_map.json        # Full dataset: audio ↔ form mapping
│   ├── form_schemas.json       # Extracted schemas per form type
│   ├── pairs.json              # Matched audio-form pairs for evaluation
│   └── form_type_resolutions.json
│
└── results/
    └── evaluation/
        ├── all_field_results.json   # Every field comparison
        └── aggregate_stats.json     # Overall & critical accuracy
```

### What Each Step Produces

| Script | Reads From | Writes To |
|---|---|---|
| `01_data_prep.py` | `data/raw/*.docx`, `data/raw/*.ogg/.m4a/...` | `data/ground_truth/`, `data/audio_wav/`, `data/*.json` |
| `02_transcribe.py` | `data/audio_wav/`, `data/pairs.json` | `data/transcripts/<model>/`, `data/transcripts/final/` |
| `03_extract.py` | `data/transcripts/final/`, `data/form_schemas.json` | `data/extractions/<model>/` |
| `04_evaluate.py` | `data/extractions/`, `data/ground_truth/` | `results/evaluation/` |
| `05_demo.py` | `data/form_schemas.json`, `data/ground_truth/` | (live web UI) |
| `wer_checker.py` | `data/transcripts/`, `data/extractions/`, `data/ground_truth/` | JSON reports in respective dirs |

---

## Running the Pipeline

Always activate the virtual environment first:

```bash
cd ~/project-dir && source venv/bin/activate
```

### Step 1: Data Preparation

```bash
python 01_data_prep.py
```

Parses all .docx forms into JSON, converts audio to WAV, matches audio-form pairs. Run this once (or whenever new data is added to `data/raw/`).

### Step 2: Transcription

```bash
python 02_transcribe.py
```

Runs all configured STT models sequentially on every audio file. Each model loads, processes all files, then unloads to free GPU VRAM before the next model loads. Expect 5–15 minutes total for ~90 minutes of audio.

### Step 3: LLM Extraction

```bash
python 03_extract.py
```

Requires Ollama to be running (`ollama serve &`). Runs each LLM model on all transcripts. Uses the best STT model's transcript (most words) as input. The extraction prompt includes few-shot examples per form type for dramatically better accuracy.

### Step 4: Evaluation

```bash
python 04_evaluate.py
```

Compares extracted fields against ground truth. Reports overall accuracy, critical field accuracy, per-form-type accuracy, confidence calibration, and most-missed fields.

### Step 5: Live Demo

```bash
python 05_demo.py
```

Launches a Gradio web UI at `http://localhost:7860` (also creates a public share link for presentations). Upload any audio file in any language + optionally a .docx form template.

---

## Pipeline Steps in Detail

### Speech-to-Text (02_transcribe.py)

Four STT models are benchmarked:

| Model | Engine | Params | Description |
|---|---|---|---|
| Whisper Large V3 Turbo | faster-whisper | 809M | Fast baseline, best speed/accuracy tradeoff |
| Whisper Large V3 | faster-whisper | 1.55B | Accuracy ceiling, slowest |
| Qwen3-ASR 0.6B | qwen-asr | 0.6B | Ultra-lightweight, 52 languages |
| Qwen3-ASR 1.7B | qwen-asr | 1.7B | SOTA open-source, multi-accent |

Each transcript is saved as JSON with full text, word-level timestamps (Whisper), timing metrics, and language detection.

### LLM Extraction (03_extract.py)

Four LLMs are benchmarked via Ollama:

| Model | Ollama ID | Params | VRAM Usage | Notes |
|---|---|---|---|---|
| Llama 3.2 | `llama3.2:3b` | 3B | ~2 GB | Lightweight baseline |
| Qwen 2.5 | `qwen2.5:7b` | 7B | ~5 GB | Prompt-sensitive |
| Phi-4 | `phi4:14b` | 14B | ~9 GB | Best accuracy |
| MedGemma 1.5 | `medgemma:4b` | 4B | ~3 GB | Medical-specialist (Google Health AI) |

**MedGemma 1.5 4B** is a medical-domain LLM from Google's Health AI Developer Foundations. It is specifically trained on EHR data extraction (90% on EHRQA) and medical document understanding (78% F1 on lab reports). At 4B params it requires less VRAM than Phi-4 while being medically specialised. See the [MedGemma installation section](#medgemma-installation) below for setup instructions.

The extraction prompt includes:
- A system prompt with 11 critical rules (field name matching, date formatting, medication specificity, etc.)
- A few-shot example specific to the detected form type (initial, follow-up, or triage)
- The full target schema with exact field names
- The transcript

Output format per field:
```json
{
  "value": "Naproxen 500mg twice daily",
  "confidence": "high"
}
```

### Evaluation (04_evaluate.py)

Field-level scoring with three outcomes:

| Match Type | Score | Criteria |
|---|---|---|
| **Exact** | 1.0 | Values match (after normalisation) |
| **Partial** | 0.5 | Significant overlap (≥30% word overlap for long text, containment for short text) |
| **Miss** | 0.0 | Wrong value or not found |
| **Skip** | excluded | Extraction said NOT_MENTIONED (excluded from denominator) |

**Critical fields** are identified by keywords: medication, dosage, prescribed, diagnosis, allergy, psychosis, safeguarding, etc. These are reported separately because getting them wrong has direct patient safety implications.

### Auto-Schema Detection (utils/auto_schema.py)

Makes the pipeline work with any clinical form, not just the 3 hardcoded types:

```bash
# Extract schema from a single .docx
python utils/auto_schema.py path/to/form.docx

# Extract schemas from a directory of forms
python utils/auto_schema.py path/to/forms_directory/
```

The detection logic: 1-column tables in the .docx are section headers, 2-column tables are key-value field pairs. The form type is inferred from section names.

---

## WER & Accuracy Checker

`wer_checker.py` is a standalone analysis tool with three modes:

### Cross-Compare STT Models (Transcript WER)

```bash
python wer_checker.py --cross-compare
```

Uses the model with the most total words as a pseudo-reference and computes Word Error Rate for all other models against it. Reports aggregate WER and per-file breakdown.

### Compare Against Gold Transcripts

```bash
python wer_checker.py --gold-dir data/gold_transcripts/
```

Place manually-transcribed `.txt` files in the gold directory (named to match WAV files, e.g., `Ag 1.txt`). Computes true WER per model.

### Field Accuracy (JSON Extraction)

```bash
python wer_checker.py --json-compare
```

Compares extracted JSON field values against ground truth using exact/partial/miss scoring. Shows per-file accuracy breakdown per LLM model.

### Run Everything

```bash
python wer_checker.py --all
```

### Quick One-Off WER

```bash
python wer_checker.py --ref "the patient has pain" --hyp "the patient had pain"
```

---

## Live Demo

```bash
python 05_demo.py
```

Opens at `http://localhost:7860` with a public share link for presentations.

### Features

- **Any audio format**: OGG, M4A, MP3, WAV, FLAC, AAC, WMA — pydub/ffmpeg handles conversion
- **Multilingual**: Audio in any language is auto-detected and transcribed; form fields are always extracted in English
- **Custom form templates**: Upload any .docx clinical form and the system auto-detects the schema
- **Ground truth comparison**: For known recordings, extracted values are compared to ground truth with match/mismatch indicators
- **Confidence scores**: Each extracted field shows a confidence level (high/medium/low)
- **Pipeline timing**: Every step (conversion, transcription, detection, extraction) is timed

---

## Benchmark Results

Results from local testing on RTX 5070 Ti Laptop GPU (12.8 GB VRAM), ~89 minutes of clinical audio across 19 recordings.

### STT Benchmark

| Model | RTF | Speed | Total Words |
|---|---|---|---|
| Whisper Large V3 Turbo | 0.027 | 38× real-time | 9,713 |
| Whisper Large V3 | 0.174 | 6× real-time | 9,814 |
| Qwen3-ASR 0.6B | 0.044 | 23× real-time | 5,255 |
| Qwen3-ASR 1.7B | 0.045 | 22× real-time | 5,934 |

Whisper produces ~2× more words than Qwen-ASR, leading to higher downstream extraction accuracy.

### LLM Extraction (Best STT: Whisper Large V3 Turbo)

| Model | Success Rate | Overall Accuracy | Critical Field Accuracy |
|---|---|---|---|
| Phi-4 14B | 17/18 | **66.6%** | **63.8%** |
| Llama 3.2 3B | 13/18 | 40.9% | 36.3% |
| Qwen 2.5 7B | 17/18 | 15.3% | 12.4% |
| MedGemma 1.5 4B | tested | underperformed | — |

MedGemma was tested but did not outperform Phi-4 on this specific extraction task, likely because MedGemma's medical training focuses on radiology reports and lab data rather than cannabis clinic consultation forms. It remains a strong candidate for other medical extraction tasks and is included in the pipeline for benchmarking.

### Best Pipeline

**Whisper Large V3 Turbo → Phi-4 14B**: 66.6% overall accuracy, 63.8% critical field accuracy, fully self-hosted.

---

## Configuration Reference

All configuration lives in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `STT_MODELS` | 4 models | List of STT models to benchmark (name, engine, model_id, params) |
| `LLM_MODELS` | 3 models | List of LLMs to benchmark (name, ollama_id, params) |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `LLM_TEMPERATURE` | `0.1` | Low for factual extraction |
| `LLM_MAX_TOKENS` | `4096` | Max generation tokens (extraction.py overrides to 8192) |
| `LLM_TIMEOUT` | `600` | Seconds before extraction times out |
| `CRITICAL_FIELD_KEYWORDS` | 14 keywords | Fields containing these are flagged as safety-critical |
| `AUDIO_EXTENSIONS` | 7 formats | Accepted audio file extensions |

### Adding a New STT Model

Add an entry to `STT_MODELS` in `config.py`:

```python
{
    "name": "my-model-name",
    "engine": "faster-whisper",  # or "qwen-asr"
    "model_id": "model-id-for-engine",
    "params": "1.5B",
    "description": "Description for logs",
},
```

### Adding a New LLM Model

```bash
# Pull the model first
ollama pull model-name:tag
```

Then add to `LLM_MODELS` in `config.py`:

```python
{
    "name": "display-name",
    "ollama_id": "model-name:tag",
    "params": "7B",
    "description": "Description for logs",
},
```

---

## MedGemma Installation

[MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) is a medical-domain LLM from Google's Health AI Developer Foundations program, built on the Gemma 3 architecture and fine-tuned on medical text, EHR data, and clinical documents. It is free for both research and commercial use.

Key capabilities relevant to this project:
- **EHR understanding**: 90% accuracy on the EHRQA electronic health record QA dataset
- **Medical document understanding**: 78% F1 on lab report data extraction
- **Smaller than Phi-4**: 4B params (~3 GB VRAM) vs 14B params (~9 GB VRAM)

MedGemma is not available directly via `ollama pull`. It must be installed from a GGUF file:

### Step 1: Accept the License

Go to [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it), log in, and agree to the Health AI Developer Foundations terms of use.

### Step 2: Download the GGUF

```bash
cd ~/project-dir
pip install huggingface-hub
mkdir -p models

# Download the Q4_K_M quantisation (~2.5 GB)
huggingface-cli download unsloth/medgemma-1.5-4b-it-GGUF \
    medgemma-1.5-4b-it-Q4_K_M.gguf \
    --local-dir ./models
```

### Step 3: Create the Ollama Model

```bash
cat > models/Modelfile << 'EOF'
FROM ./medgemma-1.5-4b-it-Q4_K_M.gguf
PARAMETER temperature 0.1
PARAMETER num_predict 8192
TEMPLATE """<start_of_turn>user
{{ .System }}

{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""
EOF

ollama create medgemma:4b -f models/Modelfile
```

### Step 4: Verify

```bash
ollama run medgemma:4b "Extract the patient ID from this text: Patient ID AG-042, male, age 55."
```

### Step 5: Add to config.py

MedGemma should already be in `LLM_MODELS` in `config.py`:

```python
{
    "name": "medgemma-4b",
    "ollama_id": "medgemma:4b",
    "params": "4B",
    "description": "MedGemma 1.5 4B — medical-specialist, trained on EHR extraction",
},
```

Then re-run extraction and evaluation:

```bash
python 03_extract.py
python 04_evaluate.py
```

### Notes

- The `unsloth/medgemma-1.5-4b-it-GGUF` repo provides multiple quantisation levels. Q4_K_M is a good balance of speed and quality. Q8_0 is higher quality but uses ~5 GB VRAM.
- MedGemma is sensitive to prompt formatting. The Gemma 3 chat template uses `<start_of_turn>` / `<end_of_turn>` tokens, which are handled by the Modelfile above.
- MedGemma has not been optimised for multi-turn conversations. Each extraction is a single-turn request, which is the intended use pattern.

---

## Troubleshooting

### PyTorch doesn't detect GPU (Blackwell SM_120)

Stable PyTorch doesn't support SM_120. Install nightly:

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Ollama uses CPU instead of GPU

Known issue with Blackwell GPUs on WSL2. Two fixes:

1. **Force GPU**: `OLLAMA_NUM_GPU=999 CUDA_VISIBLE_DEVICES=0 ollama serve`
2. **Run Ollama on Windows** (native, not WSL2) — it has better Blackwell GPU support. From WSL2, update `OLLAMA_URL` in `config.py` to point at the Windows host IP.

### Ollama timeout errors

Increase `LLM_TIMEOUT` in `config.py` to `600` (10 minutes). The few-shot prompts are large (~3000 tokens), and complex forms with 40+ fields take longer to generate.

### "Address already in use" when starting Ollama

```bash
sudo kill -9 $(lsof -t -i:11434) 2>/dev/null
sleep 2
ollama serve > /dev/null 2>&1 &
```

### Qwen3-ASR import error

The correct import is `from qwen_asr import Qwen3ASRModel` (not `Qwen3ASR`). The model uses `Qwen3ASRModel.from_pretrained()` with `.transcribe()` returning result objects with `.text` and `.language` attributes.

### OGG files not accepted in demo

The demo uses `gr.File` (not `gr.Audio`) for upload to avoid browser-level format restrictions. All audio format conversion is handled server-side by pydub/ffmpeg.

### Low extraction accuracy

The biggest accuracy gains come from prompt engineering, not model swapping. Ensure `utils/extraction.py` includes the few-shot examples. Key improvements:
- Few-shot example per form type showing complete input→output
- Explicit field name matching rules (copy field names exactly)
- Date formatting rules ("14th of March 1968" → "14-03-1968")
- Medication specificity (drug name + dose + frequency)
- Token limit set to 8192 (forms with 40+ fields need this)

---

## License & Privacy

All processing is performed locally. No audio, transcripts, or patient data is transmitted to external servers. The STT models (Whisper, Qwen3-ASR) and LLMs (Phi-4, Llama, Qwen) all run on-device via GPU inference.
