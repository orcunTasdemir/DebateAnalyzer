"""
Whisper Training Module
Fine-tune Whisper models on debate data
"""

from .preprocessor import DebatePreprocessor
from .trainer import WhisperTrainer
from .config import (
    DEFAULT_MODEL_NAME,
    AVAILABLE_MODELS,
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE
)

__version__ = "0.1.0"
__all__ = [
    "DebatePreprocessor",
    "WhisperTrainer",
    "DEFAULT_MODEL_NAME",
    "AVAILABLE_MODELS",
    "DEFAULT_DATASET_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_MODEL_OUTPUT_DIR",
    "DEFAULT_NUM_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE"
]