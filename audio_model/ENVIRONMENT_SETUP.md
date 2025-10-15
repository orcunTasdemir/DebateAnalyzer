# Whisper Training Environment Setup

This document describes how to recreate the `whisper-training` conda environment for the audio_model pipeline.

## Quick Restore from environment.yml

If you have the `environment.yml` file, you can restore the entire environment with:

```bash
conda env create -f environment.yml
```

## Manual Setup (if environment.yml is missing)

### Step 1: Create Environment
```bash
conda create -n whisper-training python=3.11 -y
```

### Step 2: Install PyTorch 2.2.0 with CUDA 11.8 (for GTX 980 compatibility)
```bash
conda activate whisper-training
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Core Dependencies
```bash
pip install "numpy<2.0,>=1.24" \
    transformers \
    datasets \
    accelerate \
    huggingface_hub \
    librosa \
    soundfile \
    evaluate \
    jiwer \
    yt-dlp \
    youtube-transcript-api \
    tqdm \
    tensorboard
```

### Step 4: Verify Installation
```bash
python -c "import torch; import transformers; import librosa; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

You should see:
- PyTorch: 2.2.0+cu118
- CUDA: True

## Key Version Constraints

**CRITICAL:** Your GTX 980 GPU only supports up to PyTorch 2.2.0 with CUDA 11.8.

**Do NOT upgrade:**
- torch > 2.2.0
- numpy >= 2.0 (keep at 1.26.4)

## Running the Pipeline

Once the environment is activated, you can run:

```bash
# Activate environment
conda activate whisper-training

# Navigate to audio_model
cd /home/ot/Work/DebateAnalyzer/audio_model

# Run full pipeline
python main.py

# Or run specific stages
python main.py --only-find
python main.py --only-scrape
python main.py --only-preprocess
python main.py --only-train

# For GTX 980 training
python main.py --only-train --batch-size 1 --gradient-accumulation 16 --epochs 3
```

## Backup This Configuration

Always keep a backup of:
1. `environment.yml` - Full environment specification
2. This `ENVIRONMENT_SETUP.md` - Manual setup instructions

To regenerate `environment.yml`:
```bash
conda env export -n whisper-training > environment.yml
```
