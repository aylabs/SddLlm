# Quickstart: Automated Training Pipeline

**Feature**: 002-training-pipeline  
**Last Updated**: 2026-02-04  
**Time to Complete**: ~7 minutes (default configuration)

## Prerequisites

- Python 3.11 or higher
- 2GB free disk space
- Internet connection (for corpus download, first run only)
- Packages: `torch`, `sentencepiece`, `tqdm`, `numpy`

### Install Dependencies

```bash
# If not already installed from feature 001
pip install torch sentencepiece tqdm numpy pytest
```

## One-Command Training

Train a complete model from scratch:

```bash
./scripts/train_pipeline.sh
```

**Expected Output:**

```
🚀 TinyTransformer Training Pipeline
=====================================
[1/4] ✓ Checking prerequisites... PASS
[2/4] 📖 Downloading corpus...
      Downloaded pg1342.txt (Pride & Prejudice)
      Downloaded pg11.txt (Alice in Wonderland)
      Downloaded pg2000.txt (Don Quijote)
      ✓ Created corpus: 3.0 MB
[3/4] 🔤 Training tokenizer...
      ✓ Trained vocabulary: 8000 tokens
      ✓ Character coverage: 99.96%
[4/4] 🧠 Training model (20 epochs)...
      Epoch 1/20: train_loss=6.84, val_loss=6.17 [20s]
      Epoch 5/20: train_loss=5.26, val_loss=5.22 [20s]
      Epoch 10/20: train_loss=4.88, val_loss=4.94 [20s]
      Epoch 15/20: train_loss=4.74, val_loss=4.85 [20s]
      Epoch 20/20: train_loss=4.71, val_loss=4.84 [20s]
      
      ✅ Training complete!
      📊 Final metrics:
         - Train loss: 4.71
         - Validation loss: 4.84
         - Training time: 6m 40s
      
      💾 Artifacts saved:
         - data/best_model.pt (lowest val_loss)
         - data/final_model.pt (last epoch)
         - data/training_metrics.json
      
      ✓ Validation PASSED
         - Sample generation: coherent
         - Memory (RSS): 253 MB ≤ 400 MB ✓
         - Latency (p95): 1 ms ≤ 250 ms ✓
```

## Custom Configuration

### Fast Training (Quick Test)

Train for fewer epochs to validate setup:

```bash
./scripts/train_pipeline.sh --epochs 5 --batch-size 64
```

Expected time: ~90 seconds

### Resume Interrupted Training

If training was interrupted (Ctrl+C, system reboot):

```bash
./scripts/train_pipeline.sh --resume
```

Continues from last saved checkpoint (every 5 epochs).

### Experiment with Hyperparameters

```bash
# Lower learning rate for more stable training
./scripts/train_pipeline.sh --learning-rate 0.0001 --epochs 30

# Smaller batch size to reduce memory usage
./scripts/train_pipeline.sh --batch-size 16
```

### Retrain Model Only

If you already have corpus and tokenizer:

```bash
./scripts/train_pipeline.sh --skip-corpus --skip-tokenizer
```

Expected time: ~6 minutes (skips download and tokenizer phases)

## Verify Training Results

### Test Generated Text

```bash
# Generate English text
python -m src.cli.minimal_llm generate \
  --prompt "Hello, write a short story" \
  --max_tokens 50

# Generate Spanish text
python -m src.cli.minimal_llm generate \
  --prompt "Hola mundo" \
  --max_tokens 50
```

### Check Training Metrics

```bash
# View detailed training history
cat data/training_metrics.json | python -m json.tool

# Quick summary
python -c "
import json
with open('data/training_metrics.json') as f:
    metrics = json.load(f)
    history = metrics['epoch_history']
    print(f'Epochs: {len(history)}')
    print(f'Final train loss: {history[-1][\"train_loss\"]:.4f}')
    print(f'Final val loss: {history[-1][\"validation_loss\"]:.4f}')
    print(f'Training time: {metrics[\"final_metrics\"][\"total_training_time\"]}s')
"
```

### Validate Constitutional Compliance

```bash
# Test inference metrics
python -m src.cli.minimal_llm generate \
  --prompt "Test prompt" \
  --max_tokens 100 \
  --json > inference_test.json

# Check memory and latency
python -c "
import json
with open('inference_test.json') as f:
    result = json.load(f)
    metrics = result['metrics']
    print(f'RSS: {metrics[\"rss_mb\"]} MB (target: ≤ 400 MB)')
    print(f'Peak: {metrics[\"peak_mb\"]} MB (target: ≤ 512 MB)')
    print(f'p95 latency: {metrics[\"latency_p95_ms\"]} ms (target: ≤ 250 ms)')
    print(f'Tokens/sec: {metrics[\"tokens_per_sec\"]:.1f}')
"
```

## Troubleshooting

### Error: "Insufficient disk space"

**Cause**: Less than 2GB free disk space

**Solution**:
```bash
# Check disk space
df -h .

# Free up space or specify different output directory
./scripts/train_pipeline.sh --output-dir /path/with/space
```

### Error: "Out of memory" during training

**Cause**: Batch size too large for available RAM

**Solution**:
```bash
# Reduce batch size
./scripts/train_pipeline.sh --batch-size 16

# Or reduce sequence length
./scripts/train_pipeline.sh --max-seq-length 64
```

### Error: "Loss is NaN or Inf"

**Cause**: Training divergence (learning rate too high)

**Solution**:
```bash
# Use lower learning rate
./scripts/train_pipeline.sh --learning-rate 0.0001
```

### Training seems stuck (loss not decreasing)

**Symptoms**: Loss stays around 7.0-8.0 after several epochs

**Possible Causes**:
- Corrupted corpus
- Learning rate too low
- Model architecture issue

**Solutions**:
```bash
# Verify corpus is text (not binary)
file data/corpus_bilingual.txt  # Should say "UTF-8 Unicode text"

# Try higher learning rate
./scripts/train_pipeline.sh --learning-rate 0.001

# Check for errors in training log
./scripts/train_pipeline.sh 2>&1 | tee training.log
```

## Command Reference

### Full Option List

```bash
./scripts/train_pipeline.sh --help
```

**Available Options:**
- `--corpus-url <URL>`: Custom corpus source (default: Project Gutenberg)
- `--vocab-size <INT>`: Tokenizer vocabulary size (default: 8000)
- `--epochs <INT>`: Training epochs (default: 20)
- `--batch-size <INT>`: Batch size (default: 32)
- `--learning-rate <FLOAT>`: Learning rate (default: 0.0003)
- `--max-seq-length <INT>`: Max sequence length (default: 128)
- `--resume`: Resume from checkpoint without prompting
- `--skip-corpus`: Skip corpus download if exists
- `--skip-tokenizer`: Skip tokenizer training if exists
- `--output-dir <PATH>`: Output directory (default: ./data)

### Exit Codes

- `0`: Success
- `1`: Prerequisite check failed
- `2`: Corpus download failed
- `3`: Tokenizer training failed
- `4`: Model training failed
- `5`: Validation failed

## Next Steps

### Learn About Training Process

Read the comprehensive training guide:

```bash
# Open in VS Code
code docs/TRAINING_GUIDE.md

# Or view in terminal
less docs/TRAINING_GUIDE.md
```

**Guide Contents:**
- Tokenization explained (vocabulary building, BPE/unigram)
- Model initialization (architecture, parameters)
- Training loop (forward pass, attention, loss, backprop)
- Optimization (AdamW, learning rate scheduling)
- Troubleshooting common issues
- Hyperparameter tuning reference
- Constitutional compliance for 1GB devices

### Experiment with Hyperparameters

```bash
# Try different configurations
./scripts/train_pipeline.sh --epochs 10  # Faster iteration
./scripts/train_pipeline.sh --vocab-size 16000  # Larger vocabulary
./scripts/train_pipeline.sh --batch-size 64 --learning-rate 0.0005  # Different training dynamics
```

Compare results using `training_metrics.json` from each run.

### Deploy Trained Model

See feature 001 quickstart for deployment instructions:

```bash
# Bundle model with metadata
python -m src.lib.runtime create_bundle \
  --model data/best_model.pt \
  --tokenizer data/bilingual_8k.model

# Test in production-like environment
python -m src.cli.minimal_llm generate \
  --prompt "Production test" \
  --safety-mode strict \
  --json
```

## Files Created

After successful training:

```
data/
├── corpus_bilingual.txt       # Training corpus (3.0 MB)
├── bilingual_8k.model         # Tokenizer model (374 KB)
├── bilingual_8k.vocab         # Vocabulary file (147 KB)
├── best_model.pt              # Best checkpoint (~10 MB)
├── final_model.pt             # Final checkpoint (~10 MB)
├── checkpoint_epoch_5.pt      # Periodic checkpoints (~10 MB each)
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── checkpoint_epoch_20.pt
└── training_metrics.json      # Training history (~50 KB)
```

## Tips

1. **Start small**: Use `--epochs 5` to validate setup before full training
2. **Save checkpoints**: Don't delete checkpoint files until training complete
3. **Monitor progress**: Training logs show loss decreasing (6.8 → 4.7 is good)
4. **Validate quality**: Test generated text before deploying
5. **Check compliance**: Verify inference metrics meet constitutional budgets

## Support

- **Training issues**: See [TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md) troubleshooting section
- **Inference issues**: See [feature 001 documentation](../001-minimal-llm/quickstart.md)
- **Architecture questions**: See [DESIGN_LLM.md](../DESIGN_LLM.md)
- **Inference details**: See [INFERENCE_EXPLAINED.md](../INFERENCE_EXPLAINED.md)
