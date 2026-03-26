#!/bin/bash
# ============================================================
# Armour Group Challenge — Environment Setup for WSL2
# ============================================================
# Run this INSIDE your WSL2 Ubuntu terminal:
#   chmod +x setup.sh && ./setup.sh
#
# Prerequisites:
#   - WSL2 with Ubuntu installed
#   - NVIDIA GPU drivers installed on WINDOWS (not WSL)
#     Download from: https://www.nvidia.com/drivers
#   - At least 30GB free disk space
# ============================================================

set -e  # Exit on any error

echo "══════════════════════════════════════════════════"
echo "🏥 Armour Group — Environment Setup"
echo "══════════════════════════════════════════════════"

# ── Step 1: Check NVIDIA driver is visible from WSL2 ──
echo ""
echo "Step 1: Checking NVIDIA GPU access..."
if nvidia-smi --query-gpu=name --format=csv,noheader,nounits > /dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "✅ GPU detected from WSL2"
else
    echo "❌ nvidia-smi not found or GPU not accessible!"
    echo ""
    echo "   This means WSL2 can't see your GPU."
    echo "   Fix: Install the LATEST NVIDIA driver on WINDOWS (not inside WSL)."
    echo "   Download from: https://www.nvidia.com/drivers"
    echo "   After installing, restart WSL2: wsl --shutdown (from PowerShell)"
    echo "   Then re-run this script."
    exit 1
fi

# ── Step 2: Install system dependencies ──
echo ""
echo "Step 2: Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg \
    python3-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    zstd \
    2>/dev/null
echo "✅ System packages installed"

# ── Step 3: Check Python version (need 3.12+ for Blackwell) ──
echo ""
echo "Step 3: Checking Python version..."
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
echo "   Found Python $PYTHON_MAJOR.$PYTHON_MINOR"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "   ⚠️  Python 3.12+ is required for Blackwell (SM_120) support."
    echo "   Installing Python 3.12..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y 2>/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev 2>/dev/null
    # Use python3.12 explicitly for the venv
    PYTHON_BIN="python3.12"
    echo "   ✅ Python 3.12 installed"
else
    PYTHON_BIN="python3"
    echo "   ✅ Python version is compatible"
fi

# ── Step 4: Create project directory ──
echo ""
echo "Step 4: Setting up project structure..."
PROJECT_DIR="/mnt/c/Users/chris/dev/armour-group"
mkdir -p "$PROJECT_DIR"/{data/{raw,ground_truth,audio_wav,transcripts,extractions},results/{evaluation,demo},utils}

echo "✅ Project directory: $PROJECT_DIR"

# ── Step 5: Create Python virtual environment ──
echo ""
echo "Step 5: Creating Python virtual environment..."
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    $PYTHON_BIN -m venv venv
    echo "✅ Virtual environment created (using $PYTHON_BIN)"
else
    echo "✅ Virtual environment already exists"
fi

source venv/bin/activate

# ── Step 6: Install Python packages ──
echo ""
echo "Step 6: Installing Python packages..."
echo ""
echo "   ⚠️  IMPORTANT: Your RTX 5070 Ti uses Blackwell architecture (SM_120)."
echo "   PyTorch stable does NOT support SM_120 yet."
echo "   We MUST install PyTorch nightly with CUDA 12.8+ support."

# Check Python version (need 3.12+)
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "   Python version: $PYTHON_VERSION"

# Upgrade pip first
pip install --upgrade pip -q

# ── PyTorch NIGHTLY with CUDA 12.8 for Blackwell SM_120 ──
echo "   Installing PyTorch nightly (cu128) for Blackwell support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -q

# STT: faster-whisper for Whisper models
pip install faster-whisper -q

# STT: Qwen3-ASR
pip install qwen-asr -q

# Document parsing
pip install python-docx -q

# Audio processing
pip install pydub -q

# Evaluation
pip install jiwer scikit-learn -q

# Demo UI
pip install gradio -q

# LLM inference (Ollama Python client)
pip install ollama -q

# Utilities
pip install pandas tqdm -q

echo "✅ Python packages installed"

# ── Step 7: Verify CUDA works with PyTorch (SM_120) ──
echo ""
echo "Step 7: Verifying PyTorch + CUDA + Blackwell SM_120..."
python3 -c "
import torch
print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    cap = torch.cuda.get_device_capability(0)
    print(f'   Compute capability: {cap[0]}.{cap[1]}')
    if cap[0] >= 12:
        print('   ✅ Blackwell SM_120 is supported!')
    else:
        print('   ⚠️  GPU detected but SM_120 may not be fully supported')
        print('   Try: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128')
    # Quick tensor test on GPU
    try:
        x = torch.randn(100, 100, device='cuda')
        y = x @ x.T
        print(f'   ✅ GPU tensor operations working!')
    except Exception as e:
        print(f'   ❌ GPU tensor test failed: {e}')
else:
    print('   ❌ CUDA not available — models will run on CPU (very slow)')
    print('   Check your NVIDIA driver on Windows and restart WSL2')
    print('   Then try: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128')
"

# ── Step 8: Install Ollama (for LLM inference) ──
echo ""
echo "Step 8: Installing Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama already installed"
else
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed"
fi

# ── Step 9: Pull LLM models ──
echo ""
echo "Step 9: Pulling LLM models (this may take 10-20 min)..."

# Start Ollama server in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &>/dev/null &
    sleep 3
    echo "   Waiting for Ollama server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "   ✅ Ollama server ready"
            break
        fi
        sleep 1
    done
    if [ $i -eq 30 ]; then
        echo "   ❌ Ollama server failed to start after 30 seconds"
        exit 1
    fi
fi

echo "   Pulling Phi-4 14B (best accuracy from round 1)..."
ollama pull phi4:14b

echo "   Pulling Qwen 2.5 7B (re-testing with better prompts)..."
ollama pull qwen2.5:7b

echo "   Pulling Llama 3.2 3B (lightweight baseline)..."
ollama pull llama3.2:3b

echo "✅ All LLM models ready"
ollama list

# ── Step 10: Verify Qwen3-ASR ──
echo ""
echo "Step 10: Verifying Qwen3-ASR installation..."
python3 -c "
try:
    import qwen_asr
    print('   ✅ qwen-asr package installed')
    print('   Models will be downloaded on first use (~1-3 GB)')
except ImportError:
    print('   ⚠️  qwen-asr not found, will install manually')
"

# ── Done ──
echo ""
echo "══════════════════════════════════════════════════"
echo "✅ SETUP COMPLETE"
echo "══════════════════════════════════════════════════"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "  1. Copy your .docx forms and audio files into:"
echo "     $PROJECT_DIR/data/raw/"
echo ""
echo "  2. Activate the environment:"
echo "     cd $PROJECT_DIR && source venv/bin/activate"
echo ""
echo "  3. Run the pipeline:"
echo "     python 01_data_prep.py"
echo "     python 02_transcribe.py"
echo "     python 03_extract.py"
echo "     python 04_evaluate.py"
echo "     python 05_demo.py"
echo ""