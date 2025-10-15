# Whisper Fine-Tuning for Debate Transcription

This module fine-tunes OpenAI's Whisper model on political debate audio for improved transcription accuracy.

## Hardware Constraints

**Important**: This module is designed to work with **PyTorch 2.2.0** (required for GTX 980 GPU compatibility).

### Why PyTorch 2.2.0?

- **GTX 980 GPU** only supports PyTorch up to version 2.2.0
- PyTorch 2.2.0 does **not** support `torchcodec` library
- This requires a **custom audio loading approach** using `librosa`

### Audio Loading Implementation

Instead of using HuggingFace's standard Audio column feature:
```python
# Standard approach (requires torchcodec)
batch["audio"]["array"]  # Won't work with PyTorch 2.2.0
```

We use librosa for manual audio loading:
```python
# Our approach (works with PyTorch 2.2.0)
audio_array, sr = librosa.load(audio_path, sr=16000)
```

**Important**: This approach is functionally equivalent and does **NOT** affect model training quality. The only difference is that audio is loaded on-the-fly during `dataset.map()` instead of being pre-loaded in the dataset.

## Installation

### 1. Create Conda Environment

```bash
conda create -n whisper-training python=3.10
conda activate whisper-training
```

### 2. Install PyTorch 2.2.0 (for GTX 980)

```bash
# For CUDA 11.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install Dependencies

```bash
pip install -e .
```

Required packages:
- `transformers` - HuggingFace transformers library
- `datasets` - HuggingFace datasets library
- `evaluate` - Evaluation metrics
- `librosa` - Audio loading (replaces torchcodec)
- `soundfile` - Audio file I/O
- `jiwer` - Word Error Rate calculation

## Usage

### Prerequisites

Run the `youtube-scraper` module first to create the debate dataset:
```bash
cd ../youtube-scraper
python main.py
```

This creates:
- Audio files: `Data/debate_dataset/audio/*.wav`
- Transcripts: `Data/debate_dataset/transcripts/*.json`
- Metadata: `Data/debate_dataset/metadata.json`

### Full Pipeline (Recommended)

Preprocess and train in one command:

```bash
conda run -n whisper-training python main.py
```

With custom options:
```bash
conda run -n whisper-training python main.py \
    --model-name openai/whisper-small \
    --num-epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-5
```

### Preprocessing Only

If you want to preprocess first and train later:

```bash
conda run -n whisper-training python main.py --preprocess-only
```

This creates:
- Segmented audio: `Data/debate_dataset_processed/segments/*.wav`
- Processed dataset: `Data/debate_dataset_processed/whisper_ready/`

### Training Only

If preprocessing is already done:

```bash
conda run -n whisper-training python main.py \
    --train-only \
    --dataset-path /path/to/whisper_ready
```

## Configuration

### Model Selection

Available Whisper models (in order of size):
- `openai/whisper-tiny` (default, fastest)
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v2`
- `openai/whisper-large-v3`

**Recommendation for GTX 980**: Use `whisper-tiny` or `whisper-base` due to limited GPU memory (4GB).

### Training Hyperparameters

Default configuration (in [config.py](whisper_train/config.py)):

```python
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WARMUP_STEPS = 200
DEFAULT_EVAL_STEPS = 100
DEFAULT_SAVE_STEPS = 100
```

**Effective batch size**: `batch_size * gradient_accumulation_steps = 1 * 16 = 16`

### Audio Processing

```python
DEFAULT_SAMPLING_RATE = 16000  # Hz (Whisper requirement)
DEFAULT_MAX_DURATION = 30      # seconds (Whisper maximum)
DEFAULT_MIN_DURATION = 1       # seconds
DEFAULT_SILENCE_THRESHOLD = 0.01
```

### Dataset Splits

```python
DEFAULT_TEST_SIZE = 0.2           # 20% for test+validation
DEFAULT_VALIDATION_SPLIT = 0.5    # 50% of test set becomes validation
DEFAULT_SEED = 42                 # Reproducibility
```

Result: 80% train / 10% validation / 10% test

## Command-Line Options

### General Options

```bash
--dataset-dir PATH          # Input debate dataset (default: Data/debate_dataset)
--output-dir PATH           # Processed data output (default: Data/debate_dataset_processed)
--model-name MODEL          # Whisper model to use (default: openai/whisper-tiny)
--model-output-dir PATH     # Fine-tuned model output (default: Model/whisper-debate-finetuned)
```

### Pipeline Control

```bash
--preprocess-only           # Only preprocess data, skip training
--train-only                # Only train, skip preprocessing
--dataset-path PATH         # Path to preprocessed dataset (for --train-only)
```

### Training Options

```bash
--num-epochs N              # Number of epochs (default: 3)
--batch-size N              # Batch size per device (default: 1)
--learning-rate LR          # Learning rate (default: 1e-5)
--freeze-encoder            # Freeze encoder, train decoder only (faster)
--max-debates N             # Limit debates for testing
```

## Examples

### Quick Test (1 Debate)

```bash
conda run -n whisper-training python main.py --max-debates 1
```

### Small Model, More Epochs

```bash
conda run -n whisper-training python main.py \
    --model-name openai/whisper-base \
    --num-epochs 10 \
    --batch-size 1
```

### Fast Training (Decoder Only)

```bash
conda run -n whisper-training python main.py \
    --model-name openai/whisper-tiny \
    --freeze-encoder
```

This trains only the decoder, which is faster but may have slightly lower accuracy.

## Programmatic Usage

```python
from whisper_train import DebatePreprocessor, WhisperTrainer

# Step 1: Preprocess
preprocessor = DebatePreprocessor(
    dataset_dir="Data/debate_dataset",
    output_dir="Data/debate_dataset_processed"
)
dataset = preprocessor.create_dataset(max_debates=5)
dataset_processed = preprocessor.prepare_for_whisper(dataset, model_name="openai/whisper-tiny")
save_path = preprocessor.save_dataset(dataset_processed)

# Step 2: Train
trainer = WhisperTrainer(
    model_name="openai/whisper-tiny",
    dataset_path=save_path,
    output_dir="Model/whisper-debate-finetuned",
    num_epochs=3,
    batch_size=1
)
trainer.setup()
trainer.train()
results = trainer.evaluate_all()

print(f"Validation WER: {results['validation']['eval_wer']:.2f}%")
print(f"Test WER: {results['test']['eval_wer']:.2f}%")
```

## Output Structure

After running the full pipeline:

```
DebateAnalyzer/
└── audio_model/
    ├── Data/
    │   ├── debate_dataset/                      # From youtube-scraper
    │   │   ├── audio/*.wav
    │   │   ├── transcripts/*.json
    │   │   └── metadata.json
    │   └── debate_dataset_processed/            # From preprocessor
    │       ├── segments/*.wav                   # Segmented audio
    │       └── whisper_ready/                   # HuggingFace dataset
    │           ├── dataset_dict.json
    │           ├── train/
    │           ├── validation/
    │           └── test/
    ├── Model/
    │   └── whisper-debate-finetuned/            # Fine-tuned model
    │       ├── config.json
    │       ├── preprocessor_config.json
    │       ├── tokenizer_config.json
    │       ├── pytorch_model.bin
    │       └── ...
    ├── youtube-scraper/                         # Scraping package
    ├── whisper-train/                           # Training package (this)
    └── project_config.py                        # Shared configuration
```

## Using the Fine-Tuned Model

### With Transformers Pipeline

```python
from transformers import pipeline

# Load fine-tuned model
pipe = pipeline(
    'automatic-speech-recognition',
    model='Model/whisper-debate-finetuned'
)

# Transcribe audio
result = pipe('path/to/debate_audio.wav')
print(result['text'])
```

### With Whisper Classes

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained('Model/whisper-debate-finetuned')
processor = WhisperProcessor.from_pretrained('Model/whisper-debate-finetuned')

# Load audio
audio, sr = librosa.load('path/to/audio.wav', sr=16000)

# Process
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)
```

## Monitoring Training

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir=Model/whisper-debate-finetuned
```

Metrics tracked:
- Training loss
- Validation loss
- Word Error Rate (WER)
- Learning rate

## Troubleshooting

### Out of Memory (OOM)

If you encounter CUDA out of memory errors:

1. **Use smaller model**: `--model-name openai/whisper-tiny`
2. **Reduce batch size**: Already at minimum (1)
3. **Freeze encoder**: `--freeze-encoder` (reduces memory usage)
4. **Reduce gradient accumulation**: Edit `DEFAULT_GRADIENT_ACCUMULATION_STEPS` in config.py

### Slow Training

GTX 980 has limited compute power. Expected training times:
- **whisper-tiny**: ~2-3 hours for 10 debates
- **whisper-base**: ~4-6 hours for 10 debates
- **whisper-small**: May not fit in 4GB VRAM

Speedup options:
- Use `--freeze-encoder` (trains decoder only)
- Use fewer debates: `--max-debates 5`
- Reduce epochs: `--num-epochs 1`

### librosa Import Error

If you get "No module named 'librosa'":

```bash
conda activate whisper-training
pip install librosa soundfile
```

### Dataset Not Found

Make sure to run `youtube-scraper` first:

```bash
cd ../youtube-scraper
conda run -n env-data-manipulation python main.py
```

## Technical Notes

### Why librosa Instead of Audio Column?

The standard HuggingFace approach uses:
```python
from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

This requires `torchcodec`, which is not available in PyTorch 2.2.0.

Our librosa approach:
```python
audio_array, sr = librosa.load(audio_path, sr=16000)
```

**Performance comparison**:
- Standard: Audio pre-loaded in dataset → faster iteration
- Librosa: Audio loaded on-the-fly → slower but works with PyTorch 2.2.0

**Quality comparison**: Identical - both produce the same 16kHz audio arrays

### Word Error Rate (WER)

WER measures transcription accuracy:
- **0%**: Perfect transcription
- **100%**: Completely wrong transcription

Typical WER for Whisper on debates:
- Base model: 15-25%
- Fine-tuned: 8-15% (improvement)

## Architecture

### Module Structure

```
whisper-train/
├── whisper_train/              # Package
│   ├── __init__.py
│   ├── config.py              # Configuration constants
│   ├── preprocessor.py        # DebatePreprocessor class
│   └── trainer.py             # WhisperTrainer class
├── main.py                     # CLI and pipeline functions
├── README.md                   # This file
└── pyproject.toml             # Package configuration
```

### Classes

- **DebatePreprocessor**: Segments audio, creates HuggingFace dataset
- **WhisperTrainer**: Fine-tunes Whisper model, handles training loop
- **DataCollatorSpeechSeq2SeqWithPadding**: Pads audio features and labels

### Pipeline Flow

```
1. Load debates from youtube-scraper
   ↓
2. Segment audio based on transcript timestamps
   ↓
3. Filter segments (duration, silence, text length)
   ↓
4. Create HuggingFace DatasetDict (train/val/test)
   ↓
5. Extract Whisper features using librosa
   ↓
6. Tokenize transcript text
   ↓
7. Save processed dataset
   ↓
8. Load Whisper model
   ↓
9. Train with Seq2SeqTrainer
   ↓
10. Evaluate on validation and test sets
   ↓
11. Save fine-tuned model
```

## License

MIT License - Same as parent DebateAnalyzer project

## Contributing

See main project [CONTRIBUTING.md](../CONTRIBUTING.md)

## References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [HuggingFace Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)
- [Fine-tuning Whisper Guide](https://huggingface.co/blog/fine-tune-whisper)
