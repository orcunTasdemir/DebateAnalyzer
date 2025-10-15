"""
Preprocess debate data for Whisper fine-tuning
Segments audio and aligns with transcripts

NOTE: This module uses librosa for audio loading instead of HuggingFace's Audio column.
This is required for PyTorch 2.2.0 (GTX 980 GPU constraint) which lacks torchcodec support.
"""

import json
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm
import numpy as np

from .config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENTS_DIR,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_DURATION,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_TEST_SIZE,
    DEFAULT_VALIDATION_SPLIT,
    DEFAULT_SEED,
    DEFAULT_MODEL_NAME
)


class DebatePreprocessor:
    """Preprocess debate audio and transcripts for Whisper training"""
    
    def __init__(
        self,
        dataset_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        sampling_rate: Optional[int] = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        silence_threshold: Optional[float] = None
    ) -> None:
        """
        Initialize preprocessor

        Args:
            dataset_dir: Path to debate_dataset (from youtube-scraper)
            output_dir: Where to save processed data
            sampling_rate: Audio sampling rate in Hz
            max_duration: Maximum segment length in seconds
            min_duration: Minimum segment length in seconds
            silence_threshold: Threshold for silence detection

        Raises:
            FileNotFoundError: If metadata.json not found in dataset_dir
            json.JSONDecodeError: If metadata.json is invalid
        """
        self.dataset_dir: Path = Path(dataset_dir or DEFAULT_DATASET_DIR)
        self.output_dir: Path = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.segments_dir: Path = self.output_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)

        self.sampling_rate: int = sampling_rate or DEFAULT_SAMPLING_RATE
        self.max_duration: int = max_duration or DEFAULT_MAX_DURATION
        self.min_duration: int = min_duration or DEFAULT_MIN_DURATION
        self.silence_threshold: float = silence_threshold or DEFAULT_SILENCE_THRESHOLD

        # Load metadata
        metadata_path: Path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                "Run youtube-scraper first to create the debate dataset."
            )

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata: List[Dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in metadata file: {e.msg}", e.doc, e.pos
            )

        print(f"✓ Loaded {len(self.metadata)} debates from {self.dataset_dir}")
    
    def segment_audio(
        self, audio_path: Path, transcript: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Segment audio based on transcript timestamps

        Args:
            audio_path: Path to full audio file
            transcript: List of transcript segments with 'text', 'start', 'duration'

        Returns:
            List of segments with audio paths and text

        Raises:
            IOError: If audio file cannot be loaded
        """
        # Load full audio
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sampling_rate)
        except Exception as e:
            raise IOError(f"Failed to load audio file {audio_path}: {e}")

        video_id: str = Path(audio_path).stem

        segments: List[Dict[str, Any]] = []
        segment_counter: int = 0
        
        for entry in transcript:
            text = entry['text'].strip()
            start_time = entry['start']
            duration = entry['duration']
            
            # Skip if text is empty or too short
            if not text or len(text) < 3:
                continue
            
            # Skip if duration is outside bounds
            if duration < self.min_duration or duration > self.max_duration:
                continue
            
            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int((start_time + duration) * sr)
            
            # Check bounds
            if start_sample >= len(audio) or end_sample > len(audio):
                continue
            
            segment_audio = audio[start_sample:end_sample]
            
            # Skip if audio is too quiet (likely silence)
            if np.max(np.abs(segment_audio)) < self.silence_threshold:
                continue
            
            # Save segment
            segment_filename = f"{video_id}_seg_{segment_counter:06d}.wav"
            segment_path = self.segments_dir / segment_filename
            
            sf.write(segment_path, segment_audio, sr)
            
            segments.append({
                'audio': str(segment_path),
                'text': text,
                'duration': duration,
                'video_id': video_id,
            })
            
            segment_counter += 1
        
        return segments
    
    def process_all_debates(
        self, max_debates: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all debates in the dataset

        Args:
            max_debates: Limit number of debates to process (for testing)

        Returns:
            List of all segments

        Raises:
            json.JSONDecodeError: If transcript file is invalid JSON
        """
        all_segments: List[Dict[str, Any]] = []

        print("\n🎬 Segmenting debate audio...\n")

        debates_to_process = self.metadata[:max_debates] if max_debates else self.metadata

        for debate in tqdm(debates_to_process, desc="Processing debates"):
            # Load transcript
            transcript_path = Path(debate['transcript_path'])
            if not transcript_path.exists():
                print(f"  ⚠️  Transcript not found: {transcript_path}")
                continue

            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Invalid JSON in transcript {transcript_path}: {e}")
                continue

            # Check audio exists
            audio_path = Path(debate['audio_path'])
            if not audio_path.exists():
                print(f"  ⚠️  Audio not found: {audio_path}")
                continue

            # Segment audio
            try:
                segments = self.segment_audio(audio_path, transcript)
                all_segments.extend(segments)
                print(f"  ✓ {debate['title'][:50]}... -> {len(segments)} segments")
            except IOError as e:
                print(f"  ⚠️  {e}")
                continue

        print(f"\n✅ Total segments created: {len(all_segments)}")

        return all_segments
    
    def create_dataset(
        self, max_debates: Optional[int] = None
    ) -> DatasetDict:
        """
        Create HuggingFace dataset from segments

        Args:
            max_debates: Limit number of debates (for testing)

        Returns:
            DatasetDict with train/validation/test splits

        Raises:
            ValueError: If no segments were created from the debates
        """
        # Process all debates
        segments = self.process_all_debates(max_debates=max_debates)

        if not segments:
            raise ValueError(
                "No segments created! Check your audio/transcript files. "
                f"Expected format:\n"
                f"  Audio: {self.dataset_dir}/audio/*.wav\n"
                f"  Transcripts: {self.dataset_dir}/transcripts/*.json"
            )

        print("\n📊 Creating HuggingFace dataset...")

        # Create dataset
        dataset = Dataset.from_list(segments)

        # NOTE: Audio column casting is commented out due to PyTorch 2.2.0 / GTX 980 constraint
        # PyTorch 2.2.0 lacks torchcodec support required for Audio column
        # Instead, we use librosa in prepare_for_whisper() method
        # dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

        # Split train/validation/test
        print("Splitting dataset...")
        train_test = dataset.train_test_split(
            test_size=DEFAULT_TEST_SIZE,
            seed=DEFAULT_SEED
        )
        test_valid = train_test['test'].train_test_split(
            test_size=DEFAULT_VALIDATION_SPLIT,
            seed=DEFAULT_SEED
        )

        dataset_dict = DatasetDict({
            'train': train_test['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })

        # Print statistics
        self._print_dataset_stats(dataset_dict, segments)

        return dataset_dict
   
    def prepare_for_whisper(
        self, dataset_dict: DatasetDict, model_name: Optional[str] = None
    ) -> DatasetDict:
        """
        Prepare dataset for Whisper training using librosa for audio loading

        NOTE: This method uses librosa.load() instead of HuggingFace Audio column.
        This is required for PyTorch 2.2.0 (GTX 980 GPU constraint) which lacks
        torchcodec support. The audio is loaded on-the-fly during dataset.map().

        Args:
            dataset_dict: DatasetDict to prepare
            model_name: Whisper model name (default: from config)

        Returns:
            Processed DatasetDict with input_features and labels

        Raises:
            Exception: If feature extractor or tokenizer fails to load
        """
        model_name = model_name or DEFAULT_MODEL_NAME

        print(f"\n🎯 Preparing for Whisper ({model_name})...")

        try:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
            tokenizer = WhisperTokenizer.from_pretrained(
                model_name,
                language="English",
                task="transcribe"
            )
        except Exception as e:
            raise Exception(f"Failed to load model components: {e}")

        def prepare_sample(batch: Dict[str, Any]) -> Dict[str, Any]:
            """
            Load audio with librosa and extract features

            This approach replaces the standard HuggingFace Audio column method:
            - Standard: batch["audio"]["array"] (requires torchcodec)
            - Our approach: librosa.load(audio_path) (works with PyTorch 2.2.0)
            """
            audio_path = batch["audio"]  # String path, not Audio object
            audio_array, sr = librosa.load(audio_path, sr=self.sampling_rate)

            batch["input_features"] = feature_extractor(
                audio_array, sampling_rate=sr
            ).input_features[0]

            batch["labels"] = tokenizer(batch["text"]).input_ids
            return batch

        print("Processing audio features...")
        dataset_dict = dataset_dict.map(
            prepare_sample,
            remove_columns=dataset_dict["train"].column_names,
            num_proc=1,  # Set >1 if multiprocessing works on your system
            desc="Extracting features",
            load_from_cache_file=False
        )

        return dataset_dict

    def save_dataset(
        self, dataset_dict: DatasetDict, output_path: Optional[Path] = None
    ) -> Path:
        """
        Save processed dataset

        Args:
            dataset_dict: DatasetDict to save
            output_path: Where to save (optional, uses default if None)

        Returns:
            Path where dataset was saved
        """
        if output_path is None:
            output_path = self.output_dir / "whisper_ready"

        output_path = Path(output_path)
        dataset_dict.save_to_disk(str(output_path))
        print(f"\n💾 Dataset saved to: {output_path}")
        return output_path

    def _print_dataset_stats(
        self, dataset_dict: DatasetDict, segments: List[Dict[str, Any]]
    ) -> None:
        """
        Print dataset statistics

        Args:
            dataset_dict: The dataset dictionary with splits
            segments: List of all segments
        """
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        print(f"Train:      {len(dataset_dict['train']):,} segments")
        print(f"Validation: {len(dataset_dict['validation']):,} segments")
        print(f"Test:       {len(dataset_dict['test']):,} segments")
        print(f"Total:      {len(segments):,} segments")

        total_duration = sum(s['duration'] for s in segments)
        print(f"\nTotal audio: {total_duration / 3600:.2f} hours")
        print("="*60)