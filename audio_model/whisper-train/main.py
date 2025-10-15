"""
Complete pipeline for preprocessing and training Whisper

This module provides CLI and programmatic interfaces for:
1. Preprocessing debate data (segmentation, feature extraction)
2. Training Whisper models
3. Running full pipeline (preprocess + train)

NOTE: Designed for PyTorch 2.2.0 (GTX 980 GPU constraint)
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add project root to path (audio_model directory)
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DEBATE_DATASET_DIR, DEBATE_DATASET_PROCESSED_DIR, WHISPER_MODEL_DIR

# Add parent directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

try:
    from whisper_train import (
        DebatePreprocessor,
        WhisperTrainer,
        DEFAULT_MODEL_NAME,
        AVAILABLE_MODELS
    )
except ModuleNotFoundError:
    # Fallback for direct execution
    from preprocessor import DebatePreprocessor
    from trainer import WhisperTrainer
    from config import DEFAULT_MODEL_NAME, AVAILABLE_MODELS


def preprocess_only(
    dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    max_debates: Optional[int] = None
) -> Path:
    """
    Only preprocess the dataset

    Args:
        dataset_dir: Path to debate_dataset from youtube-scraper (uses project Data if None)
        output_dir: Where to save processed data (uses project Data if None)
        model_name: Whisper model (for tokenizer)
        max_debates: Limit number of debates (for testing)

    Returns:
        Path to saved dataset

    Raises:
        FileNotFoundError: If dataset_dir doesn't exist
        ValueError: If no segments created
    """
    if dataset_dir is None:
        dataset_dir = str(DEBATE_DATASET_DIR)
    if output_dir is None:
        output_dir = str(DEBATE_DATASET_PROCESSED_DIR)

    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING DATA")
    print("="*70)

    preprocessor = DebatePreprocessor(
        dataset_dir=dataset_dir,
        output_dir=output_dir
    )

    # Create dataset
    dataset = preprocessor.create_dataset(max_debates=max_debates)

    # Prepare for Whisper
    dataset_processed = preprocessor.prepare_for_whisper(
        dataset,
        model_name=model_name or DEFAULT_MODEL_NAME
    )

    # Save
    save_path = preprocessor.save_dataset(dataset_processed)

    print("\n✅ Preprocessing complete!")
    print(f"Dataset ready at: {save_path}")

    return save_path


def train_only(
    dataset_path: Optional[str] = None,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    freeze_encoder: bool = False
) -> WhisperTrainer:
    """
    Only train (assumes preprocessing is done)

    Args:
        dataset_path: Path to preprocessed dataset
        model_name: Whisper model to fine-tune
        output_dir: Where to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        freeze_encoder: Freeze encoder (train decoder only)

    Returns:
        WhisperTrainer instance

    Raises:
        FileNotFoundError: If dataset not found
        Exception: If training fails
    """
    print("\n" + "="*70)
    print("STEP 2: TRAINING WHISPER")
    print("="*70)

    trainer = WhisperTrainer(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_encoder=freeze_encoder
    )

    # Setup and train
    trainer.setup()
    trainer.train()

    # Evaluate
    results = trainer.evaluate_all()

    print("\n✅ Training complete!")
    print(f"Model saved to: {trainer.output_dir}")

    return trainer


def full_pipeline(
    dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    model_output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    freeze_encoder: bool = False,
    max_debates: Optional[int] = None
) -> WhisperTrainer:
    """
    Complete pipeline: Preprocess and train

    Args:
        dataset_dir: Path to debate_dataset from youtube-scraper (uses project Data if None)
        output_dir: Where to save processed data (uses project Data if None)
        model_name: Whisper model to fine-tune
        model_output_dir: Where to save fine-tuned model (uses project Model if None)
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        freeze_encoder: Freeze encoder (train decoder only)
        max_debates: Limit number of debates (for testing)

    Returns:
        WhisperTrainer instance

    Raises:
        FileNotFoundError: If dataset_dir doesn't exist
        ValueError: If no segments created
        Exception: If training fails
    """
    if dataset_dir is None:
        dataset_dir = str(DEBATE_DATASET_DIR)
    if output_dir is None:
        output_dir = str(DEBATE_DATASET_PROCESSED_DIR)
    if model_output_dir is None:
        model_output_dir = str(WHISPER_MODEL_DIR)

    print("\n" + "="*70)
    print("WHISPER TRAINING PIPELINE")
    print("="*70)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Model: {model_name or DEFAULT_MODEL_NAME}")
    print(f"Output directory: {model_output_dir}")

    # Step 1: Preprocess
    dataset_path = preprocess_only(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model_name=model_name,
        max_debates=max_debates
    )

    # Step 2: Train
    trainer = train_only(
        dataset_path=str(dataset_path),
        model_name=model_name,
        output_dir=model_output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_encoder=freeze_encoder
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nYour fine-tuned model is ready at: {trainer.output_dir}")
    print("\nTo use it:")
    print(f"  from transformers import pipeline")
    print(f"  pipe = pipeline('automatic-speech-recognition', model='{trainer.output_dir}')")
    print(f"  result = pipe('your_audio.wav')")

    return trainer


def main() -> None:
    """
    CLI interface for Whisper training

    Parses command-line arguments and executes the appropriate pipeline function.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess and train Whisper on debate data"
    )
    
    # General options
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=f"Path to debate_dataset from youtube-scraper (default: {DEBATE_DATASET_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Where to save processed data (default: {DEBATE_DATASET_PROCESSED_DIR})"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=AVAILABLE_MODELS,
        help=f"Whisper model to use (default: {DEFAULT_MODEL_NAME})"
    )
    
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=None,
        help=f"Where to save fine-tuned model (default: {WHISPER_MODEL_DIR})"
    )
    
    # Pipeline control
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only preprocess data (skip training)"
    )
    
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train (assumes preprocessing is done)"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to preprocessed dataset (for --train-only)"
    )
    
    # Training options
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device (default: 1)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder (train decoder only, faster)"
    )
    
    parser.add_argument(
        "--max-debates",
        type=int,
        help="Limit number of debates to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.preprocess_only and args.train_only:
        print("Error: Cannot use both --preprocess-only and --train-only")
        return
    
    if args.train_only and not args.dataset_path:
        print("Error: --dataset-path required when using --train-only")
        return
    
    # Run pipeline
    if args.preprocess_only:
        # Only preprocess
        preprocess_only(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_debates=args.max_debates
        )
        
    elif args.train_only:
        # Only train
        train_only(
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            output_dir=args.model_output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            freeze_encoder=args.freeze_encoder
        )
        
    else:
        # Full pipeline
        full_pipeline(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            model_output_dir=args.model_output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            freeze_encoder=args.freeze_encoder,
            max_debates=args.max_debates
        )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments, show example
        print("="*70)
        print("WHISPER TRAINING - Example Usage")
        print("="*70)
        print("\nRunning with default settings (1 debate for demo)...\n")
        
        # Run full pipeline with limited data for demo
        full_pipeline(max_debates=1)
        
        print("\n" + "="*70)
        print("To customize, run with arguments:")
        print("="*70)
        print("python main.py --model-name openai/whisper-small --num-epochs 5")
        print("python main.py --preprocess-only  # Only preprocess")
        print("python main.py --train-only --dataset-path /path/to/processed/data")
        print("python main.py --freeze-encoder  # Faster training")
        print("python main.py --help  # See all options")
        
    else:
        # Run with command line arguments
        main()