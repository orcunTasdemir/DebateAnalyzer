# Audio Model - Complete Pipeline

End-to-end pipeline for scraping political debate videos and fine-tuning Whisper models for debate transcription.

## Features

- 🔍 **Find debates** - Search YouTube for debate videos with transcripts (15 default queries)
- 📥 **Scrape data** - Download audio and transcripts automatically
- 🎤 **Fine-tune Whisper** - Train custom models on debate data
- 🚀 **One command** - Complete pipeline from search to trained model

## Quick Start

### Run Full Pipeline

```bash
cd /home/ot/Work/DebateAnalyzer/audio_model

# Activate appropriate conda environments as needed:
# - For scraping: conda activate env-data-manipulation
# - For training: conda activate whisper-training

# Find 1 video per query (15 total), scrape, and train whisper-tiny
python main.py
```

This will:
1. Search 15 queries for debates with transcripts
2. Find 1 eligible video per query (~15 videos total)
3. Download audio and transcripts
4. Preprocess data for Whisper
5. Fine-tune whisper-tiny model (3 epochs)

**Expected time**: ~2-3 hours total

## Usage Examples

### Full Pipeline (Default)

```bash
python main.py
```

### Custom Configuration

```bash
# Get 2 videos per query (30 total debates)
python main.py --videos-per-query 2

# Use larger model with more training
python main.py --model openai/whisper-base --epochs 10

# Allow auto-generated transcripts (more videos available)
python main.py --allow-auto-transcripts --videos-per-query 3
```

### Scrape Only (No Training)

```bash
# Just find and download debates
python main.py --scrape-only --videos-per-query 1
```

### Train Only (Skip Scraping)

```bash
# Use existing data, just train the model
python main.py --skip-scraping --epochs 5
```

## Command-Line Options

```
Pipeline Control:
  --scrape-only              Only find and scrape, skip training
  --skip-scraping            Skip finding/scraping, use existing data

Finding/Scraping:
  --videos-per-query N       Videos to find per query (default: 1)
  --allow-auto-transcripts   Allow auto-generated transcripts

Training:
  --model MODEL              Whisper model (default: whisper-tiny)
                            Options: tiny, base, small, medium, large-v2, large-v3
  --epochs N                 Training epochs (default: 3)
  --batch-size N             Batch size (default: 1)
```

## How It Works

### Step 1: Find Debates (DebateFinder)

Searches YouTube using 15 curated queries for political debates.

**Eligibility criteria:**
- Minimum duration: 20 minutes
- Has transcript (manual or auto-generated)
- Not already in database

**Smart searching:** Searches enough videos to find exactly N **eligible** videos per query (skipping videos without transcripts).

### Step 2: Scrape Data (DebateScraper)

For each found video:
- Downloads high-quality WAV audio
- Extracts transcript with timestamps
- Saves metadata

Output: `Data/debate_dataset/`

### Step 3: Fine-Tune Whisper (WhisperTrainer)

**Preprocessing:**
- Segments audio based on transcript timestamps
- Filters segments (duration, silence, text length)
- Extracts Whisper features using librosa
- Creates train/validation/test splits (80%/10%/10%)

Output: `Data/debate_dataset_processed/`

**Training:**
- Fine-tunes Whisper model on debate audio
- Evaluates Word Error Rate (WER)
- Saves model for deployment

Output: `Model/whisper-debate-finetuned/`

## Project Structure

```
audio_model/
├── main.py                     # ← Complete pipeline orchestration
├── __init__.py
├── pyproject.toml
├── project_config.py           # Shared paths
│
├── youtube-scraper/            # Finding and scraping
│   ├── debate_finder/
│   └── debate_scraper/
│
├── whisper-train/              # Preprocessing and training
│   └── whisper_train/
│
├── Data/                       # Generated data
│   ├── debate_dataset/
│   └── debate_dataset_processed/
│
└── Model/                      # Trained models
    └── whisper-debate-finetuned/
```

## Expected Timeline

For default config (`--videos-per-query 1`, `--epochs 3`, `whisper-tiny`):

1. **Finding**: ~2-5 minutes
2. **Scraping**: ~10-20 minutes
3. **Preprocessing**: ~5-10 minutes
4. **Training**: ~1-2 hours (GTX 980)

**Total: ~2-3 hours**

## Hardware Requirements

- **GPU**: GTX 980 (4GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Disk**: ~10GB for 15 debates + model
- **PyTorch**: 2.2.0 (for GTX 980 compatibility)

### Model Size vs VRAM

- **whisper-tiny**: Fits in 4GB ✅ (Recommended for GTX 980)
- **whisper-base**: Fits in 4GB ✅
- **whisper-small**: May OOM on 4GB ⚠️
- **whisper-medium/large**: Requires >4GB ❌

## Using the Fine-Tuned Model

```python
from transformers import pipeline

# Load your fine-tuned model
pipe = pipeline(
    'automatic-speech-recognition',
    model='audio_model/Model/whisper-debate-finetuned'
)

# Transcribe audio
result = pipe('debate_audio.wav')
print(result['text'])
```

## Programmatic Usage

```python
from audio_model import run_full_pipeline

# Run with custom settings
run_full_pipeline(
    num_videos_per_query=2,
    model_name="openai/whisper-base",
    num_epochs=5
)
```

## Default Queries

15 curated queries optimized for finding political debates:

1. presidential debate full
2. political debate 2024
3. intelligence squared debate
4. oxford union debate
5. congressional hearing full
6. senate hearing full
7. GOP debate
8. democratic debate
9. biden trump debate
10. primary debate full
11. town hall debate
12. policy debate full
13. campaign debate
14. election debate 2024
15. vice presidential debate

Edit `youtube-scraper/data/finder_data/default_queries.json` to customize.

## Troubleshooting

### "No debates found"
- Try `--allow-auto-transcripts`
- Increase `--videos-per-query`
- Check internet connection

### "Out of memory"
- Use `--model openai/whisper-tiny`
- Reduce `--videos-per-query`
- Close other GPU applications

### Slow training
- GTX 980 is limited (~2 hours for 3 epochs)
- Use `--videos-per-query 1` for less data
- Consider more powerful GPU

## Performance Expectations

### Word Error Rate (WER)

With 15 debates and whisper-tiny (3 epochs):
- **Before**: ~20-30% WER on debates
- **After**: ~10-20% WER on debates

More data + larger models = better WER

## Sub-Package Documentation

- [youtube-scraper README](youtube-scraper/README.md) - Finding and scraping details
- [whisper-train README](whisper-train/README.md) - Training details
- [CONFIGURATION.md](youtube-scraper/CONFIGURATION.md) - Configuration architecture

## License

MIT License - Part of the DebateAnalyzer project
