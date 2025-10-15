"""
Debate Scraper Module

Download debate audio and transcripts from YouTube.

This module provides tools to download audio files and transcripts from YouTube
debate videos, managing metadata and supporting bulk operations on playlists
and channels.

Main classes:
    - DebateScraper: Main orchestrator for downloading debates
    - AudioDownloader: Audio download functionality
    - TranscriptFetcher: Transcript fetching and saving
    - MetadataManager: Metadata management for processed videos

Example:
    >>> from debate_scraper import DebateScraper
    >>> scraper = DebateScraper(output_dir="debate_dataset")
    >>> scraper.process_video("https://youtube.com/watch?v=VIDEO_ID")
    >>> scraper.get_statistics()
"""

from .scraper import DebateScraper
from .downloader import AudioDownloader
from .transcript_fetcher import TranscriptFetcher
from .metadata import MetadataManager
from .config import (
    DATA_REPOSITORY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_AUDIO_QUALITY,
    DEFAULT_MANUAL_TRANSCRIPTS_ONLY,
)

__version__ = "0.1.0"
__all__ = [
    "DebateScraper",
    "AudioDownloader",
    "TranscriptFetcher",
    "MetadataManager",
    "DATA_REPOSITORY",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_AUDIO_FORMAT",
    "DEFAULT_AUDIO_QUALITY",
    "DEFAULT_MANUAL_TRANSCRIPTS_ONLY",
]
