"""
Metadata management for scraped debate videos.

This module handles loading, saving, and querying metadata for processed videos.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from .config import DATA_REPOSITORY


class MetadataManager:
    """
    Manage dataset metadata for scraped debate videos.

    Handles storing and retrieving information about processed videos including
    video details, file paths, and transcript information.
    """

    def __init__(self, metadata_file: Union[str, Path] = "metadata.json") -> None:
        """
        Initialize the metadata manager.

        Args:
            metadata_file: Filename or path for metadata storage (default: "metadata.json")
        """
        if isinstance(metadata_file, str):
            self.metadata_file: Path = DATA_REPOSITORY / metadata_file
        else:
            self.metadata_file = Path(metadata_file)
        self.metadata: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        """
        Load existing metadata or create new empty list.

        Returns:
            List of video metadata dictionaries
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    print(f"Warning: Metadata file contains non-list data, resetting")
                    return []
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in metadata file: {e}")
                return []
            except (OSError, IOError) as e:
                print(f"Warning: Error reading metadata file: {e}")
                return []
        return []

    def save(self) -> None:
        """
        Save metadata to disk.

        Creates parent directories if they don't exist.
        """
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
        except (OSError, IOError) as e:
            print(f"Error: Failed to save metadata: {e}")
            raise

    def add_video(self, video_info: Dict[str, Any]) -> None:
        """
        Add video to metadata and save.

        Args:
            video_info: Dictionary containing video information
        """
        self.metadata.append(video_info)
        self.save()

    def video_exists(self, video_id: str) -> bool:
        """
        Check if video already exists in metadata.

        Args:
            video_id: YouTube video ID to check

        Returns:
            True if video exists, False otherwise
        """
        return any(item.get("video_id") == video_id for item in self.metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary containing:
                - total_videos: Number of videos
                - total_duration: Total duration in seconds
                - total_duration_hours: Total duration in hours
                - total_segments: Total transcript segments
                - avg_segments: Average segments per video
        """
        if not self.metadata:
            return {
                "total_videos": 0,
                "total_duration": 0,
                "total_duration_hours": 0.0,
                "total_segments": 0,
                "avg_segments": 0
            }

        total_duration = sum(item.get("duration", 0) for item in self.metadata)
        total_segments = sum(item.get("transcript_segments", 0) for item in self.metadata)

        return {
            "total_videos": len(self.metadata),
            "total_duration": total_duration,
            "total_duration_hours": total_duration / 3600,
            "total_segments": total_segments,
            "avg_segments": total_segments // len(self.metadata) if self.metadata else 0
        }

    def print_statistics(self) -> None:
        """Print formatted dataset statistics to stdout."""
        stats = self.get_statistics()

        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total videos: {stats['total_videos']}")
        print(f"Total duration: {stats['total_duration_hours']:.1f} hours")
        print(f"Total transcript segments: {stats['total_segments']:,}")
        print(f"Average segments per video: {stats['avg_segments']}")
        print("=" * 50)