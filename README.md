# DebateAnalyzer

A complete pipeline for analyzing political debates using speech recognition, speaker diarization, and inconsistency detection.

## Project Structure

```
DebateAnalyzer/
├── audio_model/          # Fine-tune Whisper on debate data
│   ├── youtube-scraper/  # Find and download debate videos
│   ├── whisper-train/    # Fine-tune Whisper ASR models
│   └── main.py          # Complete pipeline orchestrator
│
└── debate_analyzer/      # Analyze transcribed debates
    ├── diarization/      # Speaker identification
    ├── transcription/    # Speech-to-text using fine-tuned Whisper
    ├── inconsistency_detection/  # Detect logical contradictions
    └── weak_supervision/ # Label aggregation and training
```

## Components

### 1. Audio Model Pipeline

Fine-tune Whisper models on political debate data for improved transcription accuracy.

**Features:**
- YouTube video search and download
- Automatic transcript validation
- Whisper model fine-tuning (tiny, base, small, medium)
- GPU memory optimization for limited VRAM
- Training subset functionality for testing

**Quick Start:**
```bash
cd audio_model
conda activate whisper-training
python main.py --help
```

See [audio_model/README.md](audio_model/README.md) for details.

### 2. Debate Analyzer

Analyze transcribed debates with speaker diarization and inconsistency detection.

**Features:**
- Speaker diarization with pyannote.audio
- Whisper-based transcription
- Inconsistency detection across time
- Weak supervision for labeling

**Quick Start:**
```bash
cd debate_analyzer
conda activate debate-analyzer
python main.py
```

## Setup

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional but recommended)
- Conda or Miniconda

### Audio Model Environment

```bash
cd audio_model
conda create -n whisper-training python=3.11
conda activate whisper-training
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

Or use the saved environment:
```bash
conda env create -f audio_model/environment.yml
```

### Debate Analyzer Environment

```bash
conda create -n debate-analyzer python=3.11
conda activate debate-analyzer
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyannote.audio transformers datasets
```

## Hardware Requirements

**Minimum:**
- 4GB GPU VRAM (GTX 980 or equivalent)
- 16GB RAM
- 50GB free disk space

**Recommended:**
- 8GB+ GPU VRAM
- 32GB RAM
- 100GB+ free disk space

See [audio_model/GPU_MEMORY_GUIDE.md](audio_model/GPU_MEMORY_GUIDE.md) for detailed memory requirements.

## Usage Examples

### Train Whisper on Debates

```bash
cd audio_model

# Full pipeline: find, scrape, preprocess, train
python main.py

# Train only (use existing data)
python main.py --only-train --model openai/whisper-small --epochs 3

# Test on 20% of data (faster iteration)
python main.py --only-train --train-subset 20
```

### Analyze a Debate

```bash
cd debate_analyzer
python main.py --audio debate.mp3 --output analysis/
```

## Documentation

- [Audio Model Setup](audio_model/ENVIRONMENT_SETUP.md)
- [GPU Memory Guide](audio_model/GPU_MEMORY_GUIDE.md)
- [Training Subset Feature](audio_model/TRAIN_SUBSET_FEATURE.md)
- [Gradient Checkpointing Fix](audio_model/GRADIENT_CHECKPOINTING_FIX.md)

## Model Performance

**Whisper Model Comparison:**

| Model | Parameters | GPU Memory | WER (approx) | Training Time |
|-------|-----------|------------|--------------|---------------|
| tiny  | 39M       | ~500MB     | ~10%         | 30 min        |
| base  | 74M       | ~800MB     | ~7%          | 45 min        |
| small | 244M      | ~2GB       | ~5%          | 1.5 hours     |
| medium| 769M      | ~4GB       | ~4%          | 4-6 hours     |

*Times based on ~2000 training samples on GTX 980*

## Features

### Audio Model
- ✅ Automated YouTube debate search
- ✅ Transcript validation (manual only)
- ✅ Whisper fine-tuning pipeline
- ✅ Gradient checkpointing for memory efficiency
- ✅ Training subset for quick testing
- ✅ 4-stage modular pipeline

### Debate Analyzer
- 🚧 Speaker diarization (in progress)
- 🚧 Transcript alignment
- 🚧 Inconsistency detection
- 🚧 Weak supervision training

## Contributing

This is a research project for analyzing political debates. Contributions welcome!

## License

MIT

## Acknowledgments

- OpenAI Whisper for ASR models
- pyannote.audio for speaker diarization
- HuggingFace Transformers for model training infrastructure

## Contact

For questions or issues, please open a GitHub issue.
