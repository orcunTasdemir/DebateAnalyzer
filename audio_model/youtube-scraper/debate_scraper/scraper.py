"""
Main DebateScraper class for downloading debate videos and transcripts.

This module orchestrates the downloading of audio and transcripts from YouTube
videos, managing metadata and handling playlists/channels.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import time
from tqdm import tqdm
import yt_dlp

from .downloader import AudioDownloader
from .transcript_fetcher import TranscriptFetcher
from .metadata import MetadataManager
from .config import DEFAULT_OUTPUT_DIR, DATA_REPOSITORY, DEFAULT_MANUAL_TRANSCRIPTS_ONLY

class DebateScraper:
    """
    Main class for scraping debate videos from YouTube.

    Orchestrates downloading audio, fetching transcripts, and managing metadata
    for debate videos from YouTube.
    """

    def __init__(
        self,
        output_dir: Union[str, Path, None] = DATA_REPOSITORY,
        manual_transcripts_only: Optional[bool] = DEFAULT_MANUAL_TRANSCRIPTS_ONLY
    ) -> None:
        """
        Initialize DebateScraper.

        Args:
            output_dir: Directory to save downloads (default from config)
            manual_transcripts_only: Only use manually created transcripts (default: True)
        """
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        if manual_transcripts_only is None:
            manual_transcripts_only = DEFAULT_MANUAL_TRANSCRIPTS_ONLY

        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.transcript_dir = self.output_dir / "transcripts"
        self.manual_transcripts_only = manual_transcripts_only

        # Initialize components
        self.downloader = AudioDownloader(self.audio_dir)
        self.transcript_fetcher = TranscriptFetcher(
            self.transcript_dir,
            manual_only=self.manual_transcripts_only
        )
        self.metadata = MetadataManager(self.output_dir / "metadata.json")
    
    def process_video(self, video_url: str) -> bool:
        """
        Process a single video - download audio and transcript.

        Args:
            video_url: YouTube video URL

        Returns:
            True if successful, False otherwise
        """
        video_id = self._extract_video_id(video_url)

        if not video_id:
            print(f"Error: Invalid URL: {video_url}")
            return False

        if self.metadata.video_exists(video_id):
            print(f"  Skipping {video_id} (already processed)")
            return True

        print(f"\nProcessing: {video_id}")

        # Get video info
        info = self.downloader.get_video_info(video_url)
        if not info:
            print(f"Error: Failed to get video info")
            return False

        print(f"  Title: {info.get('title', 'Unknown')}")
        print(f"  Duration: {info.get('duration', 0) // 60} minutes")

        # Get transcript
        print(f"  Fetching transcript...")
        transcript = self.transcript_fetcher.fetch(video_id)

        if not transcript:
            return False

        # Save transcript
        transcript_path = self.transcript_fetcher.save(video_id, transcript)
        if not transcript_path:
            return False

        print(f"  Transcript saved ({len(transcript)} segments)")

        # Download audio
        print(f"  Downloading audio...")
        audio_path = self.downloader.download(video_url, video_id)

        if not audio_path:
            return False

        # Save metadata
        video_metadata = {
            "video_id": video_id,
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "upload_date": info.get("upload_date", "unknown"),
            "channel": info.get("uploader", "unknown"),
            "url": video_url,
            "audio_path": audio_path,
            "transcript_path": transcript_path,
            "transcript_segments": len(transcript),
        }
        self.metadata.add_video(video_metadata)

        print(f"  ✓ Successfully processed!")
        return True
    
    def process_playlist(
        self, playlist_url: str, max_videos: Optional[int] = None
    ) -> None:
        """
        Process all videos in a playlist.

        Args:
            playlist_url: YouTube playlist URL
            max_videos: Maximum number of videos to process (None = all)
        """
        print(f"\n📋 Processing playlist: {playlist_url}")
        self._process_video_collection(playlist_url, max_videos, "playlist")

    def process_channel(
        self, channel_url: str, max_videos: Optional[int] = None
    ) -> None:
        """
        Process videos from a channel.

        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum number of videos to process (None = all)
        """
        print(f"\n📺 Processing channel: {channel_url}")
        self._process_video_collection(channel_url, max_videos, "channel")

    def _process_video_collection(
        self, url: str, max_videos: Optional[int], collection_type: str
    ) -> None:
        """
        Generic method to process multiple videos from a collection.

        Args:
            url: YouTube playlist or channel URL
            max_videos: Maximum number of videos to process
            collection_type: Type of collection ("playlist" or "channel")
        """
        videos = self._get_videos_from_url(url, max_videos)

        if not videos:
            print(f"No videos found in {collection_type}")
            return

        print(f"Found {len(videos)} videos")

        success_count = 0
        for video in tqdm(videos, desc="Processing videos"):
            if video and video.get('id'):
                video_url = f"https://www.youtube.com/watch?v={video['id']}"
                if self.process_video(video_url):
                    success_count += 1
                time.sleep(2)  # Rate limiting

        print(f"\n✓ Successfully processed {success_count}/{len(videos)} videos")
    
    def get_statistics(self) -> None:
        """Print dataset statistics."""
        self.metadata.print_statistics()

    def _extract_video_id(self, url: str) -> str:
        """
        Extract video ID from YouTube URL.

        Args:
            url: YouTube video URL or video ID

        Returns:
            Extracted video ID, or original string if not a URL
        """
        if "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
        elif "youtu.be" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        # Assume it's already a video ID
        return url

    def _get_videos_from_url(
        self, url: str, max_videos: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get videos from a playlist or channel URL.

        Args:
            url: YouTube playlist or channel URL
            max_videos: Maximum number of videos to return

        Returns:
            List of video dictionaries
        """
        ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if not info or "entries" not in info:
                    return []

                videos = [v for v in info["entries"] if v]
                if max_videos:
                    videos = videos[:max_videos]

                return videos
        except Exception as e:
            print(f"Error: Failed to get videos from URL: {e}")
            return []