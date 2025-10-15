"""
Fine-tune Whisper on political debate data

This module provides the WhisperTrainer class for fine-tuning Whisper models
on preprocessed debate data. It handles model setup, training, and evaluation.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EvalPrediction,
)
import evaluate

from .config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_SAVE_STEPS,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_WHISPER_READY_DIR
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that pads audio features and labels"""
    
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class WhisperTrainer:
    """Train Whisper model on debate data"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        dataset_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        logging_steps: Optional[int] = None,
        freeze_encoder: bool = False,
        train_subset_percent: Optional[int] = None,
    ) -> None:
        """
        Initialize trainer

        Args:
            model_name: Whisper model to fine-tune
            dataset_path: Path to preprocessed dataset
            output_dir: Where to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            eval_steps: Evaluation frequency
            save_steps: Save checkpoint frequency
            logging_steps: Logging frequency
            freeze_encoder: Freeze encoder (train decoder only)
            train_subset_percent: Percentage of training data to use (1-100, None=use all)
        """
        self.model_name: str = model_name or DEFAULT_MODEL_NAME
        self.dataset_path: Path = Path(dataset_path or DEFAULT_WHISPER_READY_DIR)
        self.output_dir: Path = Path(output_dir or DEFAULT_MODEL_OUTPUT_DIR)

        # Training hyperparameters
        self.num_epochs: int = num_epochs or DEFAULT_NUM_EPOCHS
        self.batch_size: int = batch_size or DEFAULT_BATCH_SIZE
        self.gradient_accumulation_steps: int = (
            gradient_accumulation_steps or DEFAULT_GRADIENT_ACCUMULATION_STEPS
        )
        self.learning_rate: float = learning_rate or DEFAULT_LEARNING_RATE
        self.warmup_steps: int = warmup_steps or DEFAULT_WARMUP_STEPS
        self.eval_steps: int = eval_steps or DEFAULT_EVAL_STEPS
        self.save_steps: int = save_steps or DEFAULT_SAVE_STEPS
        self.logging_steps: int = logging_steps or DEFAULT_LOGGING_STEPS
        self.freeze_encoder: bool = freeze_encoder
        self.train_subset_percent: Optional[int] = train_subset_percent

        # Device
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Will be initialized in setup()
        self.model: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperProcessor] = None
        self.dataset: Optional[DatasetDict] = None
        self.trainer: Optional[Seq2SeqTrainer] = None
        self.metric: Optional[Any] = None
    
    def setup(self) -> None:
        """
        Setup model, processor, and dataset

        Raises:
            FileNotFoundError: If dataset not found
            Exception: If model/processor loading fails
        """
        self._print_header()
        self._check_device()
        self._load_model()
        self._load_dataset()
        self._setup_metric()

    def train(self) -> None:
        """
        Train the model

        Raises:
            Exception: If training fails
        """
        if self.trainer is None:
            self._setup_trainer()

        self._print_training_info()

        print("\n🚀 Starting training...\n")
        try:
            self.trainer.train()
        except Exception as e:
            raise Exception(f"Training failed: {e}")

        print("\n💾 Saving model...")
        try:
            self.trainer.save_model(str(self.output_dir))
            self.processor.save_pretrained(str(self.output_dir))
            print(f"✓ Model saved to: {self.output_dir}")
        except Exception as e:
            raise Exception(f"Failed to save model: {e}")

    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        """
        Evaluate the model

        Args:
            split: Dataset split to evaluate ('validation' or 'test')

        Returns:
            Dictionary of evaluation results

        Raises:
            ValueError: If split is invalid
        """
        if split not in self.dataset:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(self.dataset.keys())}")

        if self.trainer is None:
            self._setup_trainer()

        print(f"\n📊 Evaluating on {split} set...")
        results = self.trainer.evaluate(self.dataset[split])

        print(f"Word Error Rate: {results['eval_wer']:.2f}%")

        return results

    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate on both validation and test sets

        Returns:
            Dictionary with validation and test results
        """
        print("\n" + "="*60)
        print("📊 FINAL EVALUATION")
        print("="*60)

        print("\nValidation set:")
        val_results = self.evaluate("validation")

        print("\nTest set:")
        test_results = self.evaluate("test")

        print("\n" + "="*60)

        return {
            'validation': val_results,
            'test': test_results
        }
    
    def _print_header(self) -> None:
        """Print training header"""
        print("="*60)
        print("🎤 FINE-TUNING WHISPER FOR DEBATE TRANSCRIPTION")
        print("="*60)
        print(f"Base model: {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dataset: {self.dataset_path}")
        print("="*60)

    def _check_device(self) -> None:
        """Check and print device info"""
        print(f"\n🖥️  Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {mem_gb:.2f} GB")

    def _load_model(self) -> None:
        """
        Load model and processor

        Raises:
            Exception: If model or processor fails to load
        """
        print(f"\n📦 Loading model: {self.model_name}...")

        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name,
                language="English",
                task="transcribe"
            )
        except Exception as e:
            raise Exception(f"Failed to load model/processor: {e}")

        if self.freeze_encoder:
            self.model.freeze_encoder()
            print("🔒 Encoder frozen (training decoder only)")

        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        print("✓ Model loaded")

    def _load_dataset(self) -> None:
        """
        Load preprocessed dataset

        Raises:
            FileNotFoundError: If dataset not found at path
        """
        print(f"\n📊 Loading dataset from: {self.dataset_path}...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Run preprocessing first with DebatePreprocessor."
            )

        try:
            self.dataset = load_from_disk(str(self.dataset_path))
        except Exception as e:
            raise Exception(f"Failed to load dataset: {e}")

        # Apply subset if specified
        if self.train_subset_percent is not None:
            original_train_size = len(self.dataset['train'])
            subset_size = int(original_train_size * self.train_subset_percent / 100)

            print(f"\n⚠️  Using {self.train_subset_percent}% subset of training data")
            print(f"   Original: {original_train_size:,} samples")
            print(f"   Subset: {subset_size:,} samples")

            # Select random subset with a fixed seed for reproducibility
            self.dataset['train'] = self.dataset['train'].shuffle(seed=42).select(range(subset_size))

        print(f"✓ Train samples: {len(self.dataset['train']):,}")
        print(f"✓ Validation samples: {len(self.dataset['validation']):,}")
        print(f"✓ Test samples: {len(self.dataset['test']):,}")

    def _setup_metric(self) -> None:
        """
        Setup evaluation metric

        Raises:
            Exception: If metric fails to load
        """
        try:
            self.metric = evaluate.load("wer")
        except Exception as e:
            raise Exception(f"Failed to load WER metric: {e}")

    def _compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute Word Error Rate

        Args:
            pred: Evaluation prediction object from trainer

        Returns:
            Dictionary with WER metric
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and references
        pred_str = self.processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )
        label_str = self.processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True
        )

        # Compute WER
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def _setup_trainer(self) -> None:
        """Setup Seq2SeqTrainer with all configuration"""
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.num_epochs,
            gradient_checkpointing=True,  # Enable to save GPU memory
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Fix for PyTorch 2.x
            fp16=torch.cuda.is_available(),
            eval_strategy="steps",
            per_device_eval_batch_size=self.batch_size,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            logging_steps=self.logging_steps,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            save_total_limit=2,
        )

        # Initialize trainer
        self.trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

    def _print_training_info(self) -> None:
        """Print training configuration"""
        print("\n" + "="*60)
        print("📋 TRAINING CONFIGURATION")
        print("="*60)
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Warmup steps: {self.warmup_steps}")
        print("="*60)