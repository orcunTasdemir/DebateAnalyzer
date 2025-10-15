# YouTube Debate Scraper

Complete pipeline for finding and downloading political debate videos from YouTube with transcripts.

## Features

- 🔍 **Find debates** - Search YouTube for debate videos with transcripts
- 📥 **Download audio** - Extract high-quality audio in WAV format
- 📝 **Get transcripts** - Download manually-created transcripts
- 📊 **Metadata tracking** - Track all processed videos
- 🎯 **Production-ready** - Saves to `audio_model/Data` directory

## Quick Start

### 1. Setup Environment

```bash
conda activate env-data-manipulation
cd youtube-scraper
```

### 2. Run Full Pipeline

```bash
# Run with defaults (saves to audio_model/Data/debate_dataset)
python main.py

# With custom settings
python main.py --min-duration 1800 --max-results 20
```

## Usage Examples

### Full Pipeline (Find + Scrape)

```bash
# Default: Find debates and download everything
python main.py

# Custom search queries
python main.py --queries "presidential debate 2024" "senate hearing"

# Allow auto-generated transcripts
python main.py --allow-auto-transcripts

# Custom output directory
python main.py --output-dir /custom/path
```

### Find Only (No Download)

```bash
# Only find debates, don't download
python main.py --find-only

# Find with custom parameters
python main.py --find-only --min-duration 1800 --max-results 15
```

### Scrape Only (Download Previously Found)

```bash
# Download debates that were previously found
python main.py --scrape-only
```

## Command Line Options

```
usage: main.py [-h] [--output-dir OUTPUT_DIR] [--min-duration MIN_DURATION]
               [--max-results MAX_RESULTS] [--allow-auto-transcripts]
               [--find-only] [--scrape-only] [--queries QUERIES [QUERIES ...]]

Find and scrape political debate videos from YouTube

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory to save audio and transcripts
                        (default: audio_model/Data/debate_dataset)
  --min-duration MIN_DURATION
                        Minimum video duration in seconds
                        (default: 1200 = 20 minutes)
  --max-results MAX_RESULTS
                        Maximum results per search query (default: 10)
  --allow-auto-transcripts
                        Allow auto-generated transcripts (default: manual only)
  --find-only           Only find debates, don't download
  --scrape-only         Only scrape previously found debates
  --queries QUERIES [QUERIES ...]
                        Custom search queries
```

## Output Structure

Data is saved to `audio_model/Data/debate_dataset/`:

```
debate_dataset/
├── audio/                    # WAV audio files
│   ├── VIDEO_ID_1.wav
│   ├── VIDEO_ID_2.wav
│   └── ...
├── transcripts/              # JSON transcript files
│   ├── VIDEO_ID_1.json
│   ├── VIDEO_ID_2.json
│   └── ...
└── metadata.json             # Video metadata & statistics
```

## Programmatic Usage

You can also use the modules directly in Python:

```python
from debate_finder import DebateFinder
from debate_scraper import DebateScraper
from project_config import DEBATE_DATASET_DIR

# Step 1: Find debates
finder = DebateFinder(min_duration=1200, max_results_per_query=10)
finder.search_debates()
finder.save_results()

# Step 2: Scrape debates
scraper = DebateScraper(output_dir=DEBATE_DATASET_DIR)
finder.import_to_scraper(scraper)
scraper.get_statistics()
```

## Module Architecture

```
youtube-scraper/
├── debate_finder/            # Find debates on YouTube
│   ├── finder.py            # Main finder class
│   ├── youtube.py           # YouTube search
│   ├── transcript.py        # Transcript validation
│   └── config.py            # Configuration
│
├── debate_scraper/          # Download audio & transcripts
│   ├── scraper.py          # Main scraper class
│   ├── downloader.py       # Audio download
│   ├── transcript_fetcher.py  # Transcript download
│   ├── metadata.py         # Metadata management
│   └── config.py           # Configuration
│
├── main.py                 # Complete pipeline script
└── scripts/                # Test suites
    ├── finder-scripts/
    └── scraper-scripts/
```

## Configuration

Configuration is managed through:
- `debate_finder/config.py` - Finder settings
- `debate_scraper/config.py` - Scraper settings
- `project_config.py` (parent) - Project-wide paths

See [CONFIGURATION.md](CONFIGURATION.md) for details.

## Testing

Run comprehensive test suites:

```bash
# Test finder module
conda run -n env-data-manipulation python scripts/finder-scripts/test_debate_finder.py

# Test scraper module
conda run -n env-data-manipulation python scripts/scraper-scripts/test_debate_scraper.py

# Verify configuration
python verify_config.py
```

## Requirements

- Python 3.11+
- Conda environment: `env-data-manipulation`
- FFmpeg (for audio conversion)
- Dependencies: `yt-dlp`, `youtube-transcript-api`, `tqdm`

## Troubleshooting

### No debates found
- Try increasing `--max-results`
- Use `--allow-auto-transcripts` if manual transcripts are scarce
- Try different search queries with `--queries`

### Download errors
- Check your internet connection
- Some videos may have restrictions
- Try running again later (rate limiting)

### Path issues
- Ensure you're running from the correct directory
- Check that `audio_model/Data` exists
- Verify conda environment is activated

## Examples

### Quick Test (5 results)
```bash
python main.py --max-results 5 --queries "presidential debate"
```

### Production Run (Many results)
```bash
python main.py --max-results 20 --min-duration 2400
```

### Custom Output
```bash
python main.py --output-dir /mnt/external/debates --max-results 50
```

## License

Part of the DebateAnalyzer project.
