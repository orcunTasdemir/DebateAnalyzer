"""
Configuration for Whisper training

This module provides configuration constants for the Whisper fine-tuning pipeline.
It attempts to import absolute paths from project_config.py, falling back to
relative paths if unavailable.

NOTE: This module is designed to work with PyTorch 2.2.0 (GTX 980 GPU constraint).
PyTorch 2.2.0 does not support torchcodec, requiring manual audio loading via librosa.
"""

from pathlib import Path
from typing import List
import sys

# Try to import from project config (absolute paths)
try:
    # Add project root to path (audio_model directory)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.resolve()
    sys.path.insert(0, str(PROJECT_ROOT))

    from project_config import (
        DEBATE_DATASET_DIR,
        DEBATE_DATASET_PROCESSED_DIR,
        WHISPER_MODEL_DIR
    )

    # Use absolute paths from project config
    DEFAULT_DATASET_DIR: Path = DEBATE_DATASET_DIR
    DEFAULT_OUTPUT_DIR: Path = DEBATE_DATASET_PROCESSED_DIR
    DEFAULT_MODEL_OUTPUT_DIR: Path = WHISPER_MODEL_DIR

except ImportError:
    # Fallback to relative paths if project_config not available
    DEFAULT_DATASET_DIR: Path = Path("../../Data/debate_dataset")
    DEFAULT_OUTPUT_DIR: Path = Path("../../Data/debate_dataset_processed")
    DEFAULT_MODEL_OUTPUT_DIR: Path = Path("../../Model/whisper-debate-finetuned")

# Model configuration
DEFAULT_MODEL_NAME: str = "openai/whisper-tiny"
AVAILABLE_MODELS: List[str] = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3"
]

# Subdirectories
DEFAULT_SEGMENTS_DIR: Path = DEFAULT_OUTPUT_DIR / "segments"
DEFAULT_WHISPER_READY_DIR: Path = DEFAULT_OUTPUT_DIR / "whisper_ready"

# Training hyperparameters
DEFAULT_NUM_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 16
DEFAULT_LEARNING_RATE: float = 1e-5
DEFAULT_WARMUP_STEPS: int = 200
DEFAULT_EVAL_STEPS: int = 100
DEFAULT_SAVE_STEPS: int = 100
DEFAULT_LOGGING_STEPS: int = 25

# Audio processing
DEFAULT_SAMPLING_RATE: int = 16000
DEFAULT_MAX_DURATION: int = 30  # seconds
DEFAULT_MIN_DURATION: int = 1   # seconds
DEFAULT_SILENCE_THRESHOLD: float = 0.01

# Dataset splits
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_VALIDATION_SPLIT: float = 0.5  # Of the test set
DEFAULT_SEED: int = 42