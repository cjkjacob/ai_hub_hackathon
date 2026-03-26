"""
Armour Group Challenge — Centralized Configuration
All paths, model definitions, and constants in one place.
"""
import os

# ── Project Paths ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "ground_truth")
AUDIO_WAV_DIR = os.path.join(DATA_DIR, "audio_wav")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
EXTRACTIONS_DIR = os.path.join(DATA_DIR, "extractions")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")

# Create all directories
for d in [RAW_DIR, GROUND_TRUTH_DIR, AUDIO_WAV_DIR, TRANSCRIPTS_DIR,
          EXTRACTIONS_DIR, RESULTS_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Supported Audio Formats ──
AUDIO_EXTENSIONS = ('.ogg', '.m4a', '.mp3', '.wav', '.flac', '.aac', '.wma')

# ── STT Models to Benchmark ──
STT_MODELS = [
    {
        "name": "whisper-large-v3-turbo",
        "engine": "faster-whisper",
        "model_id": "large-v3-turbo",
        "params": "809M",
        "description": "Whisper Large V3 Turbo — fast baseline from round 1",
    },
    {
        "name": "qwen3-asr-0.6b",
        "engine": "qwen-asr",
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "params": "0.6B",
        "description": "Qwen3-ASR 0.6B — ultra-lightweight, multi-accent specialist",
    },
    {
        "name": "qwen3-asr-1.7b",
        "engine": "qwen-asr",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "params": "1.7B",
        "description": "Qwen3-ASR 1.7B — SOTA open-source, should beat Whisper on accents",
    },
    {
        "name": "whisper-large-v3",
        "engine": "faster-whisper",
        "model_id": "large-v3",
        "params": "1.55B",
        "description": "Whisper Large V3 — accuracy ceiling, slowest",
    },
]

# ── LLM Models to Benchmark ──
LLM_MODELS = [
    {
        "name": "llama3.2-3b",
        "ollama_id": "llama3.2:3b",
        "params": "3B",
        "description": "Lightweight baseline — fastest",
    },
    {
        "name": "qwen2.5-7b",
        "ollama_id": "qwen2.5:7b",
        "params": "7B",
        "description": "Re-test with improved prompts (underperformed round 1)",
    },
    {
        "name": "phi4-14b",
        "ollama_id": "phi4:14b",
        "params": "14B",
        "description": "Accuracy ceiling — won round 1 at 61%",
    },
    {
        "name": "medgemma-4b",
        "ollama_id": "medgemma:4b",
        "params": "4B",
        "description": "MedGemma 1.5 4B — medical-specialist, trained on EHR extraction",
    },
]

# ── Critical Field Keywords ──
# Fields containing these keywords are clinically high-risk if wrong
CRITICAL_FIELD_KEYWORDS = [
    "medication", "dosage", "prescribed", "product", "quantity",
    "condition", "diagnosis", "allergy", "allergies",
    "psychosis", "schizophrenia", "cvd", "arrhythmia",
    "liver disease", "drug abuse", "safeguarding",
    "cannabis exposure", "amount of flower", "frequency",
]

# ── Ollama Settings ──
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_TEMPERATURE = 0.1      # Low for factual extraction
LLM_MAX_TOKENS = 4096
LLM_TIMEOUT = 300           # 5 min max per extraction
