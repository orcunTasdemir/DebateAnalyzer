"""
YouTube search functionality
"""

from typing import List, Dict, Any
import yt_dlp


class YouTubeSearcher:
    """
    Handle YouTube video searches using yt-dlp.

    This class wraps yt-dlp functionality to provide a simple interface
    for searching YouTube videos.
    """

    def __init__(self, quiet: bool = True) -> None:
        """
        Initialize the YouTube searcher.

        Configuration settings:
            - quiet: Do not print to stdout
            - no_warnings: Do not print warnings
            - extract_flat: Do not resolve/process url_results further
                (we handle playlists ourselves)

        Args:
            quiet: If True, suppress yt-dlp output (default: True)
        """
        self.ydl_opts = {
            "quiet": quiet,
            "no_warnings": True,
            "extract_flat": True,
        }

    # Search given the query and the max results for videos for that query
    def search(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search YouTube for videos.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 20)

        Returns:
            List of video dictionaries containing video metadata
        """
        search_query = f"ytsearch{max_results}:{query}"

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                result = ydl.extract_info(search_query, download=False)

                if result and "entries" in result:
                    return result["entries"]
                return []
        except Exception as e:
            print(f"Error searching YouTube for '{query}': {e}")
            return []