# Training Subset Feature

## Overview

Added `--train-subset` flag to allow training on a percentage of the full dataset. This is useful for:
- Testing larger models (like whisper-medium) without risking GPU crashes
- Quick experimentation with different hyperparameters
- Debugging training issues on a smaller subset

## Usage

```bash
# Train on 20% of the data with whisper-medium
python main.py --only-train --model openai/whisper-medium --train-subset 20

# Train on 50% with custom settings
python main.py --only-train --train-subset 50 --epochs 5 --learning-rate 5e-6

# Test tiny model on 10% (for very quick iterations)
python main.py --only-train --train-subset 10 --epochs 1
```

## How It Works

1. The full dataset is loaded from the `whisper_ready` directory
2. If `--train-subset PERCENT` is specified:
   - The training set is shuffled with a fixed seed (42) for reproducibility
   - Only the first PERCENT% of samples are selected
   - Validation and test sets remain unchanged (always 100%)

3. Training proceeds normally on the subset

## Example Output

```
📊 Loading dataset from: /path/to/whisper_ready...

⚠️  Using 20% subset of training data
   Original: 10,000 samples
   Subset: 2,000 samples

✓ Train samples: 2,000
✓ Validation samples: 500
✓ Test samples: 500
```

## Recommended Use Cases

### Testing Whisper-Medium on GTX 980

Your GTX 980 has 4GB VRAM. Whisper-medium is significantly larger than tiny/base:
- **whisper-tiny**: ~39M parameters
- **whisper-base**: ~74M parameters
- **whisper-medium**: ~769M parameters

**Recommended approach:**
1. Start with 10% to ensure it doesn't crash:
   ```bash
   python main.py --only-train --model openai/whisper-medium --train-subset 10 --epochs 1
   ```

2. If successful, try 20%:
   ```bash
   python main.py --only-train --model openai/whisper-medium --train-subset 20 --epochs 3
   ```

3. Monitor GPU memory usage:
   ```bash
   # In another terminal, run:
   watch -n 1 nvidia-smi
   ```

4. Gradually increase subset size if memory allows

### Comparing Models Quickly

```bash
# Train all models on 20% for quick comparison
python main.py --only-train --model openai/whisper-tiny --train-subset 20
python main.py --only-train --model openai/whisper-base --train-subset 20
python main.py --only-train --model openai/whisper-small --train-subset 20
python main.py --only-train --model openai/whisper-medium --train-subset 20
```

## Important Notes

1. **Validation data unchanged**: The subset only affects training data. Validation and test sets always use 100% of data for accurate evaluation.

2. **Reproducibility**: Uses seed=42 for shuffling, so the same percentage will always select the same samples.

3. **Range validation**: The percentage must be between 1 and 100 (inclusive).

4. **Model accuracy**: Training on a subset will likely result in lower accuracy than training on the full dataset. Use this for testing, not final production models.

## GPU Memory Safety

If you get CUDA out of memory errors:
1. Reduce `--train-subset` percentage
2. Keep `--batch-size 1` (already the default)
3. Reduce `--gradient-accumulation` (default is 16, try 8 or 4)
4. Use a smaller model

Example safe command for whisper-medium:
```bash
python main.py --only-train \
    --model openai/whisper-medium \
    --train-subset 15 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --epochs 2
```
