#!/usr/bin/env python3
"""
Verify configuration paths for debate_scraper module
"""

from debate_scraper.config import (
    DATA_REPOSITORY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_AUDIO_QUALITY,
    DEFAULT_MANUAL_TRANSCRIPTS_ONLY,
)

print("=" * 70)
print("DEBATE_SCRAPER CONFIGURATION")
print("=" * 70)

print("\n📁 Path Configuration:")
print(f"  DATA_REPOSITORY (testing):   {DATA_REPOSITORY}")
print(f"  DEFAULT_OUTPUT_DIR (prod):   {DEFAULT_OUTPUT_DIR}")

print("\n🎵 Audio Settings:")
print(f"  Format:  {DEFAULT_AUDIO_FORMAT}")
print(f"  Quality: {DEFAULT_AUDIO_QUALITY}")

print("\n📝 Transcript Settings:")
print(f"  Manual Only: {DEFAULT_MANUAL_TRANSCRIPTS_ONLY}")

print("\n✅ Verification:")
print(f"  DATA_REPOSITORY exists:     {DATA_REPOSITORY.exists()}")
print(f"  DEFAULT_OUTPUT_DIR parent:  {DEFAULT_OUTPUT_DIR.parent.exists()}")

print("\n" + "=" * 70)
print("Configuration loaded successfully!")
print("=" * 70)
