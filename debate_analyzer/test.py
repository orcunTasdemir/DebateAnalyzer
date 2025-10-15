from debate_analyzer.diarization.diarizer import Diarizer
import json
from pathlib import Path

def main():
    # Path to your fine-tuned Whisper model
    whisper_model_dir = Path(__file__).resolve().parents[1] / "audio_model" / "Model" / "whisper-debate-finetuned"
    audio_file = Path(__file__).resolve().parents[0] / "BWL_uwntpKI.wav"  # Change this to your test file

    diarizer = Diarizer(
        hf_auth_token="token",
        whisper_model_path=whisper_model_dir,
        min_speakers=2,
        max_speakers=5
    )

    print("\n🚀 Running diarized transcription...\n")
    result = diarizer.diarized_transcript(audio_file)

    # Save or print the result
    output_path = Path("diarized_transcript.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Diarized transcript saved to {output_path}")
    print("\nSample output:")
    for entry in result[:5]:
        print(f"[{entry['start']} - {entry['end']}] {entry['speaker']}: {entry['text']}")

if __name__ == "__main__":
    main()
