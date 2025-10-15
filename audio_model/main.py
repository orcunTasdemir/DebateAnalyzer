#!/usr/bin/env python3
"""
Audio Model - Complete Pipeline

This script orchestrates the entire workflow:
1. Find debates on YouTube (1 video per query from 15 default queries)
2. Scrape audio and transcripts for found debates
3. Preprocess data for Whisper training
4. Fine-tune Whisper model on debate data

Usage:
    # Run full pipeline (find 1 video/query, train with whisper-tiny)
    python main.py

    # Custom number of videos per query
    python main.py --videos-per-query 2

    # Custom model and epochs
    python main.py --model openai/whisper-base --epochs 5

    # Skip finding/scraping (use existing data)
    python main.py --skip-scraping

    # Only find and scrape (skip training)
    python main.py --scrape-only
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add audio_model to path
AUDIO_MODEL_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(AUDIO_MODEL_ROOT))

# Import from project_config
from project_config import DEBATE_DATASET_DIR, DEBATE_DATASET_PROCESSED_DIR, WHISPER_MODEL_DIR

# Import youtube-scraper modules
sys.path.insert(0, str(AUDIO_MODEL_ROOT / "youtube-scraper"))
from debate_finder import DebateFinder, DEFAULT_QUERIES
from debate_scraper import DebateScraper

# Import whisper-train modules
sys.path.insert(0, str(AUDIO_MODEL_ROOT / "whisper-train"))
from whisper_train import DebatePreprocessor, WhisperTrainer, AVAILABLE_MODELS


def run_full_pipeline(
    num_videos_per_query: int = 1,
    model_name: str = "openai/whisper-tiny",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 1e-5,
    train_subset_percent: Optional[int] = None,
    only_find: bool = False,
    only_scrape: bool = False,
    only_preprocess: bool = False,
    only_train: bool = False,
    enforce_manual_transcript: bool = True
) -> None:
    """
    Run the complete audio model pipeline (or individual stages).

    Args:
        num_videos_per_query: Number of eligible videos to find per query (default: 1)
        model_name: Whisper model to fine-tune (default: whisper-tiny)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size for training (default: 1)
        gradient_accumulation_steps: Gradient accumulation steps (default: 16)
        learning_rate: Learning rate (default: 1e-5)
        train_subset_percent: Percentage of training data to use (1-100, None=use all)
        only_find: Only find debates, don't scrape or train
        only_scrape: Only scrape debates (requires found debates)
        only_preprocess: Only preprocess data (requires scraped data)
        only_train: Only train model (requires preprocessed data)
        enforce_manual_transcript: Require manual transcripts only (default: True)
    """
    # Determine which stages to run
    stages_specified = only_find or only_scrape or only_preprocess or only_train
    run_find = only_find or not stages_specified
    run_scrape = only_scrape or (not stages_specified and not only_find)
    run_preprocess = only_preprocess or (not stages_specified and not only_find and not only_scrape)
    run_train = only_train or (not stages_specified and not only_find and not only_scrape and not only_preprocess)

    print("="*80)
    print("AUDIO MODEL PIPELINE")
    print("="*80)
    print(f"Stages to run:")
    print(f"  1. Find debates: {'✓' if run_find else '✗'}")
    print(f"  2. Scrape audio/transcripts: {'✓' if run_scrape else '✗'}")
    print(f"  3. Preprocess data: {'✓' if run_preprocess else '✗'}")
    print(f"  4. Train Whisper: {'✓' if run_train else '✗'}")
    print(f"\nConfiguration:")
    print(f"  Videos per query: {num_videos_per_query}")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    if train_subset_percent:
        print(f"  Training subset: {train_subset_percent}% of data")
    else:
        print(f"  Training subset: 100% (full dataset)")
    print(f"  Manual transcripts only: {enforce_manual_transcript}")
    print("="*80)

    # STAGE 1: Find debates on YouTube
    if run_find:
        print("\n" + "="*80)
        print("STAGE 1/4: FINDING DEBATES ON YOUTUBE")
        print("="*80)

        finder = DebateFinder(
            min_duration=1200,  # 20 minutes minimum
            max_results_per_query=num_videos_per_query,
            enforce_manual_transcript=enforce_manual_transcript
        )

        # Search using all default queries
        finder.search_debates(queries=DEFAULT_QUERIES)
        finder.save_results()

        found_count = len(finder.videos)
        print(f"\n✅ Stage 1 complete: Found {found_count} eligible debate videos")

        if found_count == 0:
            print("❌ No debates found. Exiting.")
            return

    # STAGE 2: Scrape audio and transcripts
    if run_scrape:
        print("\n" + "="*80)
        print("STAGE 2/4: SCRAPING AUDIO AND TRANSCRIPTS")
        print("="*80)

        scraper = DebateScraper(
            output_dir=str(DEBATE_DATASET_DIR),
            manual_transcripts_only=enforce_manual_transcript
        )

        # If we just ran find, import from finder
        # Otherwise, scraper will use existing found_debates.json
        if run_find and 'finder' in locals():
            finder.import_to_scraper(scraper)
        else:
            print("Using existing found debates from database...")

        # Show statistics
        scraper.get_statistics()
        print(f"\n✅ Stage 2 complete: Scraped debates to {DEBATE_DATASET_DIR}")

    # STAGE 3: Preprocess data
    if run_preprocess:
        print("\n" + "="*80)
        print("STAGE 3/4: PREPROCESSING DATA FOR WHISPER")
        print("="*80)

        preprocessor = DebatePreprocessor(
            dataset_dir=str(DEBATE_DATASET_DIR),
            output_dir=str(DEBATE_DATASET_PROCESSED_DIR)
        )

        # Create and prepare dataset
        dataset = preprocessor.create_dataset()
        dataset_processed = preprocessor.prepare_for_whisper(dataset, model_name=model_name)
        dataset_path = preprocessor.save_dataset(dataset_processed)

        print(f"\n✅ Stage 3 complete: Preprocessed data saved to {dataset_path}")

    # STAGE 4: Train Whisper model
    if run_train:
        print("\n" + "="*80)
        print("STAGE 4/4: TRAINING WHISPER MODEL")
        print("="*80)

        trainer = WhisperTrainer(
            model_name=model_name,
            dataset_path=str(DEBATE_DATASET_PROCESSED_DIR / "whisper_ready"),
            output_dir=str(WHISPER_MODEL_DIR),
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            train_subset_percent=train_subset_percent
        )

        trainer.setup()
        trainer.train()

        # Evaluate
        results = trainer.evaluate_all()

        print(f"\n✅ Stage 4 complete: Model saved to {trainer.output_dir}")
        print(f"   Validation WER: {results['validation']['eval_wer']:.2f}%")
        print(f"   Test WER: {results['test']['eval_wer']:.2f}%")

    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    if run_train and 'trainer' in locals():
        print(f"\nModel saved to: {trainer.output_dir}")
        print(f"\nTo use your fine-tuned model:")
        print(f"  from transformers import pipeline")
        print(f"  pipe = pipeline('automatic-speech-recognition', model='{trainer.output_dir}')")
        print(f"  result = pipe('your_audio.wav')")
    print("="*80)


def main() -> None:
    """CLI entry point for audio_model pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete pipeline for finding debates and training Whisper models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all 4 stages)
  python main.py

  # Run only specific stages
  python main.py --only-find              # Stage 1: Find debates
  python main.py --only-scrape            # Stage 2: Scrape found debates
  python main.py --only-preprocess        # Stage 3: Preprocess for training
  python main.py --only-train             # Stage 4: Train model

  # Custom configuration
  python main.py --videos-per-query 2 --epochs 5
  python main.py --model openai/whisper-base --learning-rate 5e-6

  # Hardware-constrained training (for GTX 980 or similar)
  python main.py --only-train --batch-size 1 --gradient-accumulation 8 --epochs 1

  # Test medium model on 20% of data (safer for limited GPU memory)
  python main.py --only-train --model openai/whisper-medium --train-subset 20

  # Allow auto-generated transcripts
  python main.py --allow-auto-transcripts
        """
    )

    # Pipeline control (4 stages)
    parser.add_argument(
        "--only-find",
        action="store_true",
        help="Only find debates on YouTube (Stage 1 only)"
    )
    parser.add_argument(
        "--only-scrape",
        action="store_true",
        help="Only scrape audio/transcripts (Stage 2 only, requires found debates)"
    )
    parser.add_argument(
        "--only-preprocess",
        action="store_true",
        help="Only preprocess data for training (Stage 3 only, requires scraped data)"
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help="Only train Whisper model (Stage 4 only, requires preprocessed data)"
    )

    # Finding/scraping options
    parser.add_argument(
        "--videos-per-query",
        type=int,
        default=1,
        help="Number of eligible videos to find per query (default: 1)"
    )
    parser.add_argument(
        "--allow-auto-transcripts",
        action="store_true",
        help="Allow auto-generated transcripts (default: manual only)"
    )

    # Training options
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-tiny",
        choices=AVAILABLE_MODELS,
        help="Whisper model to fine-tune (default: whisper-tiny)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size per device (default: 1, keep at 1 for GTX 980)"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16, effective batch = batch_size * this)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        metavar="PERCENT",
        help="Use only a subset of training data (1-100 percent, useful for testing large models)"
    )

    args = parser.parse_args()

    # Validate arguments
    stage_flags = [args.only_find, args.only_scrape, args.only_preprocess, args.only_train]
    if sum(stage_flags) > 1:
        print("Error: Cannot specify multiple --only-* flags together")
        print("Use one of: --only-find, --only-scrape, --only-preprocess, --only-train")
        print("Or omit all flags to run the full pipeline")
        sys.exit(1)

    if args.train_subset is not None:
        if args.train_subset < 1 or args.train_subset > 100:
            print("Error: --train-subset must be between 1 and 100")
            sys.exit(1)

    # Run pipeline
    try:
        run_full_pipeline(
            num_videos_per_query=args.videos_per_query,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            train_subset_percent=args.train_subset,
            only_find=args.only_find,
            only_scrape=args.only_scrape,
            only_preprocess=args.only_preprocess,
            only_train=args.only_train,
            enforce_manual_transcript=not args.allow_auto_transcripts
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
