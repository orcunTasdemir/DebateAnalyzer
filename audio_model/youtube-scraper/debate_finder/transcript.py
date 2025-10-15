"""
Transcript checking functionality
"""

from typing import List, Callable
from youtube_transcript_api import TranscriptList, YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from .config import DEFAULT_MANUAL_TRANSCRIPTS_ONLY


class TranscriptChecker:
    """
    Check if YouTube videos have transcripts available.

    Supports checking for both manually created and auto-generated transcripts
    in specified languages.
    """

    def __init__(self, languages: List[str] = None) -> None:
        """
        Initialize the transcript checker.

        Args:
            languages: List of language codes to check (default: ["en"])
        """
        self.languages = languages or ["en"]

    # Check if the video has a transcript given the video_id and flag for whether
    # to look for only manual or manual + auto-generated transcripts
    def has_transcript(
        self, video_id: str, manual_only: bool = DEFAULT_MANUAL_TRANSCRIPTS_ONLY
    ) -> bool:
        """
        Check if video has a transcript available.

        Args:
            video_id: YouTube video ID
            manual_only: If True, only check for manual transcripts

        Returns:
            True if transcript exists, False otherwise
        """
        if manual_only:
            return self._check_transcript(video_id, self._find_manual_transcript)
        else:
            return self._check_transcript(video_id, self._find_generated_transcript)

    # Generic method to check existence of manual/generated debate given the function
    def _check_transcript(self, video_id: str, finder_method: Callable) -> bool:
        """
        Generic method to check for transcript existence.

        Args:
            video_id: YouTube video ID
            finder_method: Method to find the transcript

        Returns:
            True if transcript exists, False otherwise
        """
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            finder_method(transcript_list)
            return True
        except (TranscriptsDisabled, NoTranscriptFound):
            return False
        except Exception as e:
            # Log unexpected errors but don't crash
            print(f"Warning: Unexpected error checking transcript for {video_id}: {e}")
            return False

    def _find_manual_transcript(self, transcript_list: TranscriptList) -> None:
        """
        Find manually created transcript in the transcript list.

        Args:
            transcript_list: YouTube transcript list object

        Raises:
            NoTranscriptFound: If no manual transcript exists
        """
        transcript_list.find_manually_created_transcript(self.languages)

    def _find_generated_transcript(self, transcript_list: TranscriptList) -> None:
        """
        Find auto-generated transcript in the transcript list.

        Args:
            transcript_list: YouTube transcript list object

        Raises:
            NoTranscriptFound: If no generated transcript exists
        """
        transcript_list.find_generated_transcript(self.languages)
