"""
Speaker Identification and Diarized Transcription
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    warnings.warn("pyannote.audio not installed. Install with: pip install pyannote.audio")


class Diarizer:
    """
    Speaker diarization + transcription for debate audio files.
    Combines pyannote.audio (speaker detection) and Whisper (speech-to-text).
    """

    def __init__(
        self,
        hf_auth_token: Optional[str] = None,
        whisper_model_path: Optional[Union[str, Path]] = None,
        min_speakers: int = 2,
        max_speakers: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            hf_auth_token: Hugging Face token for pyannote models
            whisper_model_path: Path to fine-tuned Whisper model
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            device: Device ("cuda" or "cpu")
        """
        self.device = device
        self.hf_auth_token = hf_auth_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.whisper_model_path = Path(whisper_model_path) if whisper_model_path else None

        # Load Whisper model
        if not self.whisper_model_path or not self.whisper_model_path.exists():
            raise FileNotFoundError(
                f"Whisper model not found at: {self.whisper_model_path}"
            )

        print(f"🎙️ Loading Whisper model from {self.whisper_model_path} ...")
        self.processor = WhisperProcessor.from_pretrained(self.whisper_model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_path).to(self.device)

        # Load pyannote pipeline
        if PYANNOTE_AVAILABLE:
            try:
                print("🗣️  Loading pyannote speaker diarization pipeline...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=self.hf_auth_token
                ).to(self.device)
            except Exception as e:
                self.pipeline = None
                warnings.warn(f"Failed to load pyannote pipeline: {e}")
        else:
            self.pipeline = None

    # --------------------------
    # Whisper transcription
    # --------------------------
    def transcribe(self, audio_path: Union[str, Path]) -> List[Dict]:
        """
        Generate timestamped transcription using Whisper.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of dicts with start, end, text.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        print(f"🎧 Transcribing audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                return_timestamps=True
            )

        # Decode into segments with timestamps
        result = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # NOTE: Hugging Face `return_timestamps=True` returns structure only if processor supports it
        # so we fallback to phrase-level timestamps if needed
        if isinstance(result, list):
            text = result[0]
            return [{"start": 0.0, "end": None, "text": text}]
        else:
            return result["chunks"]

    # --------------------------
    # Pyannote diarization
    # --------------------------
    def diarize(self, audio_path: Union[str, Path]) -> List[Dict]:
        """
        Run speaker diarization with pyannote.

        Returns:
            List of {speaker, start, end}
        """
        if not self.pipeline:
            raise RuntimeError("pyannote pipeline not loaded.")
        audio_path = Path(audio_path)
        print(f"🗣️ Running speaker diarization on {audio_path}...")
        diarization = self.pipeline(
            audio_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            })
        print(f"✅ Diarization found {len(set(s['speaker'] for s in segments))} speakers.")
        return segments

    # --------------------------
    # Merge results
    # --------------------------
    def merge_transcript_and_diarization(
        self, whisper_segments: List[Dict], diar_segments: List[Dict]
    ) -> List[Dict]:
        """
        Merge Whisper transcription segments with pyannote speaker segments.

        Args:
            whisper_segments: [{'start','end','text'}]
            diar_segments: [{'speaker','start','end'}]

        Returns:
            [{'speaker','start','end','text'}]
        """
        merged = []
        for w in whisper_segments:
            w_start = w.get("start", 0.0)
            w_end = w.get("end", w_start + 5.0)  # fallback 5 sec
            w_text = w.get("text", "")

            overlapping = [
                d for d in diar_segments
                if not (d["end"] <= w_start or d["start"] >= w_end)
            ]
            if not overlapping:
                merged.append({
                    "speaker": "UNKNOWN",
                    "start": w_start,
                    "end": w_end,
                    "text": w_text
                })
                continue

            for d in overlapping:
                start = max(w_start, d["start"])
                end = min(w_end, d["end"])
                merged.append({
                    "speaker": d["speaker"],
                    "start": start,
                    "end": end,
                    "text": w_text
                })
        return merged

    # --------------------------
    # Combined pipeline
    # --------------------------
    def diarized_transcript(self, audio_path: Union[str, Path]) -> List[Dict]:
        """
        Run full pipeline: transcription + diarization + merge
        """
        whisper_segments = self.transcribe(audio_path)
        diar_segments = self.diarize(audio_path)
        merged = self.merge_transcript_and_diarization(whisper_segments, diar_segments)
        return merged
