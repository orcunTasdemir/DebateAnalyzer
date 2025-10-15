"""
Transcript fetching functionality for YouTube videos.

This module handles fetching and saving transcripts from YouTube videos
using the youtube-transcript-api.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import JSONFormatter

from .config import DATA_REPOSITORY, DEFAULT_MANUAL_TRANSCRIPTS_ONLY


class TranscriptFetcher:
    """
    Handle fetching and saving transcripts from YouTube videos.

    Supports both manually created and auto-generated transcripts.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = DATA_REPOSITORY,
        manual_only: bool = DEFAULT_MANUAL_TRANSCRIPTS_ONLY
    ) -> None:
        """
        Initialize the transcript fetcher.

        Args:
            output_dir: Directory to save transcript files (default: from config)
            manual_only: Only fetch manually created transcripts (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manual_only = manual_only
        self.formatter = JSONFormatter()

    def fetch(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch transcript for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            List of transcript segments, or None if failed
        """
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)

            # Get transcript based on preference
            if self.manual_only:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            else:
                transcript = transcript_list.find_generated_transcript(['en'])

            return transcript.fetch()

        except TranscriptsDisabled:
            print(f"Error: Transcripts are disabled for this video")
            return None
        except NoTranscriptFound:
            print(f"Error: No {'manual ' if self.manual_only else ''}English transcript found")
            return None
        except Exception as e:
            print(f"Error: Unexpected error fetching transcript: {e}")
            return None

    def save(self, video_id: str, transcript: List[Dict[str, Any]]) -> Optional[str]:
        """
        Save transcript to JSON file.

        Args:
            video_id: YouTube video ID (used for filename)
            transcript: List of transcript segments

        Returns:
            Path to saved transcript file, or None if failed
        """
        try:
            output_path = self.output_dir / f"{video_id}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self.formatter.format_transcript(transcript))

            print(f"  ✓ Transcript saved to: {output_path}")
            return str(output_path)
        except (OSError, IOError) as e:
            print(f"Error: Failed to save transcript: {e}")
            return None
        except Exception as e:
            print(f"Error: Unexpected error saving transcript: {e}")
            return None