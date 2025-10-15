"""
Shared project configuration for all modules
Defines absolute paths for Data and Model directories
"""

from pathlib import Path

# Project root directory (where this file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Shared directories
DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR = PROJECT_ROOT / "Model"

# Specific data paths
DEBATE_DATASET_DIR = DATA_DIR / "debate_dataset"
DEBATE_DATASET_PROCESSED_DIR = DATA_DIR / "debate_dataset_processed"

# Model paths
WHISPER_MODEL_DIR = MODEL_DIR / "whisper-debate-finetuned"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)