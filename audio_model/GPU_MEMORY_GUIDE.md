# GPU Memory Guide for GTX 980 (4GB VRAM)

## Your GPU Constraint

**GTX 980 specs:**
- Total VRAM: 4GB (3.93 GB usable)
- Compute Capability: 5.2 (Maxwell)
- Max PyTorch: 2.2.0 with CUDA 11.8

## Whisper Model Sizes

| Model | Parameters | Approx GPU Memory (Training) | Fits GTX 980? |
|-------|-----------|------------------------------|---------------|
| whisper-tiny | 39M | ~500 MB | ✅ Yes, easily |
| whisper-base | 74M | ~800 MB | ✅ Yes, comfortably |
| whisper-small | 244M | ~2.0 GB | ✅ Yes, with care |
| whisper-medium | 769M | ~3.5-4.5 GB | ❌ **NO** - Too tight |
| whisper-large-v2/v3 | 1.5B+ | ~7+ GB | ❌ NO |

## What Happened with Whisper-Medium

Your error showed:
```
Of the allocated memory 3.10 GiB is allocated by PyTorch
GPU 0 has a total capacity of 3.93 GiB
```

The model alone takes 3.1 GB, leaving **only ~800 MB** for:
- Activations during forward pass
- Gradients during backward pass
- Optimizer states (Adam uses 2x model size)

**Result:** Out of memory even before processing the first batch.

## ✅ Recommended: Use Whisper-Small

Whisper-small is the **sweet spot** for your GPU:
- 6x larger than tiny (much better accuracy)
- Still fits comfortably in 4GB
- Good balance of quality and speed

```bash
python main.py --only-train \
    --model openai/whisper-small \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --epochs 3
```

### Why Small is Better Than Tiny

From OpenAI's benchmarks:
- **whisper-tiny**: ~10% WER (Word Error Rate)
- **whisper-small**: ~5-6% WER  ← **50% better accuracy!**
- **whisper-medium**: ~4-5% WER (only 20% better than small)

**The jump from tiny → small is MUCH bigger than small → medium.**

## If You MUST Try Medium (Not Recommended)

I've enabled gradient checkpointing which may help, but it's still risky. Try:

### Attempt 1: Extreme Memory Saving
```bash
# Reduce subset to 5%, increase gradient accumulation
python main.py --only-train \
    --model openai/whisper-medium \
    --train-subset 5 \
    --batch-size 1 \
    --gradient-accumulation 32 \
    --epochs 1
```

### Attempt 2: Use Mixed Precision (fp16)
This is already enabled by default in the trainer, but whisper-medium may still be too large.

### Attempt 3: Freeze Encoder (Not Implemented Yet)
Training only the decoder would save memory but reduce fine-tuning effectiveness.

## Memory Optimization Techniques Applied

Your trainer already uses:
1. ✅ **Gradient Accumulation** (simulate larger batches without memory)
2. ✅ **Mixed Precision (fp16)** (halves memory for activations)
3. ✅ **Gradient Checkpointing** (trades compute for memory - just added)
4. ✅ **Batch Size = 1** (minimum possible)

Even with all these, whisper-medium barely fits or doesn't fit at all.

## Comparison Table

### Tiny vs Small vs Medium on Your Data

Test this yourself with subsets:

```bash
# Test tiny (fast, but less accurate)
python main.py --only-train --model openai/whisper-tiny --train-subset 20

# Test small (recommended)
python main.py --only-train --model openai/whisper-small --train-subset 20

# Test medium (if you insist, but likely to crash)
python main.py --only-train --model openai/whisper-medium --train-subset 5
```

Compare the validation WER (Word Error Rate) at the end.

## Alternative: Use Google Colab for Medium

If you really want whisper-medium quality:

1. **Option A:** Use Google Colab (free T4 GPU with 16GB)
   - Upload your preprocessed dataset
   - Train whisper-medium there
   - Download the fine-tuned model

2. **Option B:** Rent GPU time
   - RunPod, Vast.ai, Lambda Labs
   - ~$0.20-0.50/hour for decent GPU

3. **Option C:** Stick with small locally**
   - Still get 50% better accuracy than tiny
   - Can train on full dataset
   - Faster iterations

## My Recommendation

```bash
# Use whisper-small with full dataset
conda activate whisper-training
cd /home/ot/Work/DebateAnalyzer/audio_model

python main.py --only-train \
    --model openai/whisper-small \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --epochs 3
```

This will:
- ✅ Fit comfortably in 4GB VRAM
- ✅ Train on 100% of your data
- ✅ Give you much better accuracy than tiny
- ✅ Complete without crashes
- ✅ Take ~1-2 hours (vs 4-6 hours for medium if it worked)

## Monitoring Memory Usage

While training, open another terminal and run:
```bash
watch -n 1 nvidia-smi
```

You'll see real-time GPU memory usage. Aim for 70-80% utilization for safety.
