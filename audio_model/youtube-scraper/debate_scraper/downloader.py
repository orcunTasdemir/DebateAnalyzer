"""
Audio download functionality for YouTube videos.

This module handles downloading audio from YouTube videos using yt-dlp and
converting to the specified format using FFmpeg.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import yt_dlp
from yt_dlp.utils import DownloadError
from .config import DATA_REPOSITORY, DEFAULT_AUDIO_FORMAT, DEFAULT_AUDIO_QUALITY

# Responsible for downloading audio for the scraper
class AudioDownloader:
    """
    Handle downloading and converting audio from YouTube videos.

    Uses yt-dlp to download audio and FFmpeg to convert to the desired format.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = DATA_REPOSITORY,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
        quality: str = DEFAULT_AUDIO_QUALITY
    ) -> None:
        """
        Initialize the audio downloader.

        Args:
            output_dir: Directory to save audio files (default: from config)
            audio_format: Audio format (default: "wav")
            quality: Audio quality/bitrate (default: "192")
        """
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        self.quality = quality

    def download(self, video_url: str, video_id: str) -> Optional[str]:
        """
        Download audio from YouTube video.

        Args:
            video_url: YouTube video URL
            video_id: Video ID for filename

        Returns:
            Path to downloaded audio file, or None if failed
        """
        output_path = self.output_dir / f"{video_id}.{self.audio_format}"

        if output_path.exists():
            print(f"  ✓ Audio already exists")
            return str(output_path)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": self.audio_format,
                "preferredquality": self.quality,
            }],
            "outtmpl": str(self.output_dir / f"{video_id}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            print(f"  ✓ Audio downloaded")
            return str(output_path)
        except yt_dlp.utils.DownloadError as e:
            print(f"Error: Download failed: {e}")
            return None
        except Exception as e:
            print(f"Error: Unexpected error downloading audio: {e}")
            return None

    def get_video_info(self, video_url: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata without downloading.

        Args:
            video_url: YouTube video URL

        Returns:
            Dictionary containing video metadata, or None if failed
        """
        ydl_opts = {"quiet": True, "no_warnings": True}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info
        except DownloadError as e:
            print(f"Error: Failed to get video info: {e}")
            return None
        except Exception as e:
            print(f"Error: Unexpected error getting video info: {e}")
            return None