"""
Audio Model Package

Complete pipeline for:
1. Finding political debate videos on YouTube
2. Scraping audio and transcripts
3. Fine-tuning Whisper models for debate transcription

This package integrates:
- youtube-scraper: Find and download debate videos
- whisper-train: Preprocess data and fine-tune Whisper models

Usage:
    # Command line
    python -m audio_model.main

    # Or after installation
    audio-model

    # Programmatic
    from audio_model import run_full_pipeline
    run_full_pipeline(num_videos_per_query=1, num_epochs=3)
"""

__version__ = "0.1.0"

# Re-export main function for convenience
from .main import run_full_pipeline

__all__ = ["run_full_pipeline", "__version__"]
