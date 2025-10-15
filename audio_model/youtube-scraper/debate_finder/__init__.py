"""
Debate Finder Module

Find political debate videos on YouTube with transcripts.

This module provides tools to search for and validate political debate videos
on YouTube, ensuring they have transcripts available for further processing.

Main classes:
    - DebateFinder: Main orchestrator for finding debates
    - YouTubeSearcher: YouTube search functionality
    - TranscriptChecker: Transcript validation
    - FoundDebatesManager: Database management for found debates

Example:
    >>> from debate_finder import DebateFinder
    >>> finder = DebateFinder()
    >>> debates = finder.search_debates()
    >>> finder.save_results()
"""

from .finder import DebateFinder
from .youtube import YouTubeSearcher
from .transcript import TranscriptChecker
from .config import (
    DATA_REPOSITORY,
    DEFAULT_QUERIES,
    DEFAULT_MANUAL_TRANSCRIPTS_ONLY,
    found_debates_manager,  # Export the manager instance
    FoundDebatesManager     # Export the class too
)

__version__ = "0.1.0"
__all__ = [
    "DebateFinder",
    "YouTubeSearcher",
    "TranscriptChecker",
    "DATA_REPOSITORY",
    "DEFAULT_QUERIES",
    "DEFAULT_MANUAL_TRANSCRIPTS_ONLY",
    "found_debates_manager",
    "FoundDebatesManager"
]