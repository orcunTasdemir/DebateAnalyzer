"""
Configuration for debate scraper module.

This module defines default paths and settings for downloading and processing
debate audio and transcripts from YouTube.

Paths:
    - DATA_REPOSITORY: For testing/development (youtube-scraper/data/scraper_data)
    - DEFAULT_OUTPUT_DIR: Production output (audio_model/Data/debate_dataset)
"""

from importlib.resources import files
from pathlib import Path
import sys

# Import shared configuration from debate_finder
from debate_finder import DEFAULT_MANUAL_TRANSCRIPTS_ONLY

# Get the package root
package_root = files('debate_scraper').parent

# Data repository for testing/development (module-local storage)
DATA_REPOSITORY: Path = package_root.joinpath('data/scraper_data')

# Default output directory for production (project-level Data directory)
try:
    # Try to import from project config
    sys.path.insert(0, str(package_root.parent))
    from project_config import DEBATE_DATASET_DIR
    DEFAULT_OUTPUT_DIR: Path = DEBATE_DATASET_DIR
except (ImportError, ModuleNotFoundError):
    # Fallback to local directory if project_config not available
    DEFAULT_OUTPUT_DIR: Path = package_root.joinpath("debate_dataset")

# Audio settings
DEFAULT_AUDIO_FORMAT: str = "wav"
DEFAULT_AUDIO_QUALITY: str = "192"

# Re-export the shared constant for backwards compatibility
__all__ = [
    'DATA_REPOSITORY',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_AUDIO_FORMAT',
    'DEFAULT_AUDIO_QUALITY',
    'DEFAULT_MANUAL_TRANSCRIPTS_ONLY',
]
