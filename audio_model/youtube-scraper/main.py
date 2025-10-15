"""
Complete pipeline for finding and scraping political debate videos.

This script orchestrates the full workflow:
1. Find debates on YouTube with transcripts (DebateFinder)
2. Download audio and transcripts (DebateScraper)
3. Save everything to audio_model/Data directory

Usage:
    python main.py                          # Run full pipeline with defaults
    python main.py --find-only              # Only find debates
    python main.py --scrape-only            # Only scrape found debates
    python main.py --queries "debate 2024"  # Custom search
    python main.py --help                   # See all options
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to path (audio_model directory)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DEBATE_DATASET_DIR
from debate_finder import DebateFinder, DEFAULT_QUERIES
from debate_scraper import DebateScraper


def find_debates(
    queries: Optional[List[str]] = None,
    min_duration: int = 1200,
    max_results_per_query: int = 10,
    enforce_manual_transcript: bool = True
) -> DebateFinder:
    """
    Step 1: Find debate videos with transcripts.

    Args:
        queries: List of search queries (uses defaults if None)
        min_duration: Minimum video duration in seconds (default: 1200 = 20 min)
        max_results_per_query: Max videos to check per query (default: 10)
        enforce_manual_transcript: Require manual transcripts only (default: True)

    Returns:
        DebateFinder instance with results
    """
    print("\n" + "="*70)
    print("STEP 1: FINDING DEBATES")
    print("="*70)
    
    finder = DebateFinder(
        min_duration=min_duration,
        max_results_per_query=max_results_per_query,
        enforce_manual_transcript=enforce_manual_transcript
    )
    
    # Search for debates (passing None will use defaults in the method)
    if queries is None:
        finder.search_debates()  # Uses DEFAULT_QUERIES
    else:
        finder.search_debates(queries=queries)
    
    # Save results to database
    finder.save_results()
    
    return finder


def scrape_debates(
    output_dir: Optional[str] = None,
    manual_transcripts_only: bool = True,
    use_finder_results: Optional[DebateFinder] = None
) -> DebateScraper:
    """
    Step 2: Download audio and transcripts for found debates.

    Args:
        output_dir: Directory to save downloads (default: audio_model/Data/debate_dataset)
        manual_transcripts_only: Only use manual transcripts (default: True)
        use_finder_results: DebateFinder instance to import from (optional)

    Returns:
        DebateScraper instance
    """
    if output_dir is None:
        output_dir = str(DEBATE_DATASET_DIR)
    
    print("\n" + "="*70)
    print("STEP 2: SCRAPING DEBATES")
    print("="*70)
    
    scraper = DebateScraper(
        output_dir=output_dir,
        manual_transcripts_only=manual_transcripts_only
    )
    
    if use_finder_results:
        # Import debates from finder
        use_finder_results.import_to_scraper(scraper)
    else:
        print("No finder results provided. Use finder.import_to_scraper(scraper) to import.")
    
    return scraper


def full_pipeline(
    queries: Optional[List[str]] = None,
    min_duration: int = 1200,
    max_results_per_query: int = 10,
    output_dir: Optional[str] = None,
    enforce_manual_transcript: bool = True
) -> Tuple[DebateFinder, Optional[DebateScraper]]:
    """
    Complete pipeline: Find debates and scrape them.

    Args:
        queries: Search queries (uses defaults if None)
        min_duration: Minimum video duration in seconds (default: 1200 = 20 min)
        max_results_per_query: Max videos per query (default: 10)
        output_dir: Where to save audio/transcripts (default: audio_model/Data/debate_dataset)
        enforce_manual_transcript: Require manual transcripts only (default: True)

    Returns:
        Tuple of (DebateFinder, DebateScraper or None)
    """
    if output_dir is None:
        output_dir = str(DEBATE_DATASET_DIR)
    
    print("\n" + "="*70)
    print("DEBATE SCRAPER PIPELINE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Minimum duration: {min_duration // 60} minutes")
    print(f"Manual transcripts only: {enforce_manual_transcript}")
    print(f"Max results per query: {max_results_per_query}")
    
    # Step 1: Find debates
    finder = find_debates(
        queries=queries,
        min_duration=min_duration,
        max_results_per_query=max_results_per_query,
        enforce_manual_transcript=enforce_manual_transcript
    )
    
    if not finder.videos:
        print("\n⚠️  No debates found. Pipeline complete.")
        return finder, None
    
    # Step 2: Scrape debates
    scraper = scrape_debates(
        output_dir=output_dir,
        manual_transcripts_only=enforce_manual_transcript,
        use_finder_results=finder
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)

    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Found: {len(finder.videos)} debates")
    if scraper:
        scraper.get_statistics()
        print(f"\n  Output directory: {scraper.output_dir}")
        print(f"  Audio: {scraper.audio_dir}")
        print(f"  Transcripts: {scraper.transcript_dir}")

    return finder, scraper


def main():
    """CLI interface for the debate scraper pipeline"""
    parser = argparse.ArgumentParser(
        description="Find and scrape political debate videos from YouTube"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to save audio and transcripts (default: {DEBATE_DATASET_DIR})"
    )
    
    parser.add_argument(
        "--min-duration",
        type=int,
        default=1200,
        help="Minimum video duration in seconds (default: 1200 = 20 minutes)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum results per search query (default: 10)"
    )
    
    parser.add_argument(
        "--allow-auto-transcripts",
        action="store_true",
        help="Allow auto-generated transcripts (default: manual only)"
    )
    
    parser.add_argument(
        "--find-only",
        action="store_true",
        help="Only find debates, don't download (skips scraping step)"
    )
    
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape previously found debates (skips finding step)"
    )
    
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Custom search queries (uses defaults if not provided)"
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if args.find_only and args.scrape_only:
        print("Error: Cannot use both --find-only and --scrape-only")
        return
    
    enforce_manual = not args.allow_auto_transcripts
    
    if args.scrape_only:
        # Only scrape previously found debates
        print("\n📥 Scraping previously found debates...")
        finder = DebateFinder()
        finder.load_results()
        
        scraper = scrape_debates(
            output_dir=args.output_dir,
            manual_transcripts_only=enforce_manual,
            use_finder_results=finder
        )
        
    elif args.find_only:
        # Only find debates
        finder = find_debates(
            queries=args.queries,
            min_duration=args.min_duration,
            max_results_per_query=args.max_results,
            enforce_manual_transcript=enforce_manual
        )
        
    else:
        # Full pipeline
        finder, scraper = full_pipeline(
            queries=args.queries,
            min_duration=args.min_duration,
            max_results_per_query=args.max_results,
            output_dir=args.output_dir,
            enforce_manual_transcript=enforce_manual
        )


if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, show example
        print("="*70)
        print("DEBATE SCRAPER - Example Usage")
        print("="*70)
        print("\nRunning with default settings...\n")
        
        # Run full pipeline with defaults
        finder, scraper = full_pipeline(
            queries=DEFAULT_QUERIES[:2],  # Use first 2 default queries as example
            max_results_per_query=5,      # Limit to 5 per query for demo
        )
        
        print("\n" + "="*70)
        print("To customize, run with arguments:")
        print("="*70)
        print("python main.py --output-dir /custom/path --min-duration 1800")
        print("python main.py --find-only  # Only find debates")
        print("python main.py --scrape-only  # Only download found debates")
        print("python main.py --queries 'presidential debate' 'senate debate'")
        print("python main.py --help  # See all options")
        
    else:
        # Run with command line arguments
        main()