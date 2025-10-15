from importlib.resources import files
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# The package root is /youtube-scraper
package_root = files('debate_finder').parent

# This is the individual data repository for the debate_finder module
DATA_REPOSITORY = package_root.joinpath('data/finder_data')

# These are the default queries a nd found_debates paths
queries_json = DATA_REPOSITORY.joinpath('default_queries.json')
found_debates_json = DATA_REPOSITORY.joinpath('found_debates.json')

# Static configuration for default queries are loaded once with error handling
try:
    with queries_json.open('r', encoding='utf-8') as f:
        DEFAULT_QUERIES = json.load(f)
except FileNotFoundError:
    print(f"Warning: default_queries.json not found at {queries_json}")
    DEFAULT_QUERIES = []
except json.JSONDecodeError as e:
    print(f"Warning: Invalid JSON in default_queries.json: {e}")
    DEFAULT_QUERIES = []

# By default the module gives True for enforcing only the videos that have manual transcripts
DEFAULT_MANUAL_TRANSCRIPTS_ONLY = True

# Class to manage the found_debates.json file so we can add debates or clear the file.
class FoundDebatesManager:
    """
    Manages the found_debates.json file for storing discovered debate videos.

    This class handles all CRUD operations for the debates database, including
    loading, saving, adding, checking existence, and clearing debates.

    Attributes:
        json_path: Path to the JSON file storing debate data
    """

    # Initialize with path
    def __init__(self, json_path: Optional[Path] = None) -> None:
        """
        Initialize the manager with a JSON file path.

        Args:
            json_path: Path to the debates JSON file. Uses default if None.
        """
        self.json_path = json_path or found_debates_json

    # Load from file
    def load(self) -> List[Dict[str, Any]]:
        """
        Load debates from file.

        Returns:
            List of debate dictionaries. Empty list if file not found.
        """
        try:
            with self.json_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {self.json_path}: {e}")
            return []

    # Given a list of debate dicts, add these to the found_debates.json
    def save(self, debates: List[Dict[str, Any]]) -> None:
        """
        Save debates to file.

        Args:
            debates: List of debate dictionaries to save
        """
        try:
            # Ensure parent directory exists
            self.json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(debates, f, indent=2)
        except (OSError, IOError) as e:
            print(f"Error: Failed to save debates to {self.json_path}: {e}")
            raise

    # Add a single list
    def add(self, debate: Dict[str, Any]) -> None:
        """
        Add a debate to the list.

        Args:
            debate: Dictionary containing debate information
        """
        debates = self.load()
        debates.append(debate)
        self.save(debates)

    # Check if debate already exists in the file
    def exists(self, video_id: str) -> bool:
        """
        Check if a debate already exists.

        Args:
            video_id: YouTube video ID to check

        Returns:
            True if debate exists, False otherwise
        """
        debates = self.load()
        return any(d.get('video_id') == video_id for d in debates)

    # return the current list of all debates in the file
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all debates.

        Returns:
            List of all debate dictionaries
        """
        return self.load()

    # clear all debates from the database
    def clear(self) -> None:
        """Clear all debates from the database."""
        self.save([])


# Create a default instance
found_debates_manager = FoundDebatesManager()