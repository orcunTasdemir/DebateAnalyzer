"""
Main DebateFinder class that orchestrates debate discovery
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

from .youtube import YouTubeSearcher
from .transcript import TranscriptChecker
from .config import (
    found_debates_manager,
    found_debates_json,
    DEFAULT_QUERIES,
    DEFAULT_MANUAL_TRANSCRIPTS_ONLY
)

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from debate_scraper import DebateScraper

# Class for using youtube and transcript classes to find viable candidate debates for the whisper-finetuning
class DebateFinder:
    """
    Find political debate videos on YouTube with transcripts.

    This class orchestrates the search for debate videos by coordinating
    YouTube searches and transcript validation.
    """
    # Initialize the DebateFinder, delegating the selection of the output_file location
    # and the enforcement of manual/generated transcripts to the .config
    def __init__(
        self,
        output_file: Optional[Path] = None,
        min_duration: int = 1200,
        max_results_per_query: int = 10,
        enforce_manual_transcript: Optional[bool] = None
    ) -> None:
        """
        Initialize DebateFinder.

        Args:
            output_file: Path to save results (uses default if None)
            min_duration: Minimum video duration in seconds (default: 1200 = 20 minutes)
            max_results_per_query: Max videos to check per search query
            enforce_manual_transcript: Require manual transcripts (uses default if None)
        """
        self.output_file = output_file or found_debates_json
        self.min_duration = min_duration
        self.max_results_per_query = max_results_per_query

        if enforce_manual_transcript is None:
            enforce_manual_transcript = DEFAULT_MANUAL_TRANSCRIPTS_ONLY
        self.enforce_manual_transcript = enforce_manual_transcript

        # Initialize components here to attach to this class
        self.searcher = YouTubeSearcher()
        self.transcript_checker = TranscriptChecker()

        # Results storage (in memory for current session)
        self.videos: List[Dict[str, Any]] = []
    
    # Function to search for debates for the query, returns a list of a dictionary of videos 
    def search_debates(
        self, queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for debate videos with transcripts.

        Args:
            queries: List of search queries (uses defaults if None)

        Returns:
            List of video dictionaries with debate information
        """
        if queries is None:
            queries = DEFAULT_QUERIES

        print("🔍 Searching for political debates...\n")

        for query in queries:
            print(f"Searching: {query}")
            self._process_query(query)
            print()

        self._print_summary()
        return self.videos
    
    # Processes a single query from a list of queries from input or {package_root}/data/finder_data/default_queries.json
    def _process_query(self, query: str) -> None:
        """
        Process a single search query.

        Searches YouTube and finds exactly max_results_per_query number of ELIGIBLE videos
        (videos that pass duration and transcript checks). Will keep searching more videos
        until the desired number is found or max_search_attempts is reached.

        Example: If max_results_per_query=3 and first 9 videos yield only 2 eligible,
        it will continue searching videos 10, 11, 12, etc. until finding the 3rd eligible video.

        Args:
            query: Search query string
        """
        found_count = 0
        videos_checked = 0
        max_search_attempts = 50  # Safety limit to prevent infinite loops
        search_increment = 10  # How many videos to search in each batch

        while found_count < self.max_results_per_query and videos_checked < max_search_attempts:
            # Search next batch of videos (videos_checked + search_increment total)
            next_search_size = videos_checked + search_increment
            results = self.searcher.search(query, next_search_size)

            # Process only the new videos we haven't checked yet
            new_videos = results[videos_checked:] if len(results) > videos_checked else []

            if not new_videos:
                # No more videos available for this query
                print(f"  ⚠️  No more videos available for query")
                break

            for video in new_videos:
                videos_checked += 1

                if not video:
                    continue

                video_info = self._extract_video_info(video)

                # Skip if already in the manager's database
                if found_debates_manager.exists(video_info['video_id']):
                    continue

                if self._is_valid_debate(video_info):
                    self.videos.append(video_info)
                    found_count += 1
                    print(f"✓ {video_info['title'][:60]}... ({video_info['duration_min']} min)")

                    # Check if we have enough eligible videos
                    if found_count >= self.max_results_per_query:
                        break

            # Loop continues if we still need more eligible videos

        # Report results
        if found_count < self.max_results_per_query:
            if videos_checked >= max_search_attempts:
                print(f"  ⚠️  Reached search limit ({max_search_attempts} videos). Found {found_count}/{self.max_results_per_query} eligible")
            else:
                print(f"  ⚠️  Found {found_count}/{self.max_results_per_query} eligible videos after checking {videos_checked} videos")

    # Extracts info about video from the video metadata given by YoutubeSearcher.search()
    def _extract_video_info(self, video: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant info from video result.

        Args:
            video: Video dictionary from YouTube search

        Returns:
            Dictionary with standardized video information
        """
        video_id = video["id"]
        title = video["title"]
        duration = video.get("duration", 0)

        return {
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": duration,
            "duration_min": duration // 60,
        }

    # Check if it is a valid debate dependent on video duration
    # and whether it has a manual or any-type transcript
    def _is_valid_debate(self, video_info: Dict[str, Any]) -> bool:
        """
        Check if video meets criteria for debates.

        Args:
            video_info: Dictionary containing video information

        Returns:
            True if video is a valid debate, False otherwise
        """
        # Check duration
        if video_info["duration"] < self.min_duration:
            return False

        # Check for transcript
        has_transcript = self.transcript_checker.has_transcript(
            video_info["video_id"],
            self.enforce_manual_transcript
        )

        return has_transcript
    
    # Append the currently found results for eligible debates to the found_debates.json
    def save_results(self) -> None:
        """Save current session results to the debates database."""
        if not self.videos:
            print("No videos to save")
            return

        # Add all videos to the manager (which handles deduplication internally)
        saved_count = 0
        for video in self.videos:
            if not found_debates_manager.exists(video['video_id']):
                found_debates_manager.add(video)
                saved_count += 1

        print(f"✓ Saved {saved_count} new debates to database")
        print(f"  Total debates in database: {len(found_debates_manager.get_all())}")

    # Load all debates that are currently in the database
    def load_results(self) -> List[Dict[str, Any]]:
        """
        Load all debates from the database.

        Returns:
            List of all debate dictionaries from database
        """
        self.videos = found_debates_manager.get_all()

        if not self.videos:
            print("No saved results found in database")
            return []

        print(f"✓ Loaded {len(self.videos)} videos from database")
        return self.videos

    def get_all_debates(self) -> List[Dict[str, Any]]:
        """
        Get all debates from the database without loading into instance.

        Returns:
            List of all debate dictionaries
        """
        return found_debates_manager.get_all()

    def clear_database(self) -> None:
        """Clear all debates from the database (use with caution!)."""
        found_debates_manager.clear()
        self.videos = []
        print("✓ Database cleared")

    def _print_summary(self) -> None:
        """Print summary statistics for current search session."""
        if not self.videos:
            print("No debates found")
            return

        total_duration = sum(v['duration'] for v in self.videos)

        print(f"\n{'='*60}")
        print(f"Found {len(self.videos)} new debates with transcripts")
        print(f"Total duration: {total_duration / 3600:.1f} hours")
        print(f"{'='*60}")

    def import_to_scraper(self, scraper: "DebateScraper") -> None:
        """
        Import found debates into a DebateScraper instance.

        Args:
            scraper: DebateScraper instance with process_video() method
        """
        debates_to_import = self.videos if self.videos else self.get_all_debates()

        if not debates_to_import:
            print("No debates to import")
            return

        print(f"\n--> Importing {len(debates_to_import)} debates...")

        for video in debates_to_import:
            scraper.process_video(video["url"])

        scraper.get_statistics()