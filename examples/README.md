# Training Pipeline Configuration Examples

This directory contains sample configurations for common training scenarios.

## Quick Test (2-minute training)

For rapid iteration and debugging:

```bash
./scripts/train_pipeline.sh \
    --epochs 2 \
    --batch-size 8 \
    --vocab-size 1000 \
    --max-seq-length 64 \
    --output-dir ./examples/quick-test
```

**Use case:** Testing code changes, debugging issues  
**Training time:** ~2 minutes  
**Memory usage:** ~50 MB  
**Model quality:** Low (for testing only)

---

## High Quality (15-minute training)

For best model performance:

```bash
./scripts/train_pipeline.sh \
    --epochs 30 \
    --batch-size 64 \
    --vocab-size 16000 \
    --learning-rate 0.0001 \
    --max-seq-length 256 \
    --output-dir ./examples/high-quality
```

**Use case:** Production deployment, best generation quality  
**Training time:** ~15 minutes  
**Memory usage:** ~300 MB  
**Model quality:** High

---

## Low Memory (Raspberry Pi compatible)

For resource-constrained devices:

```bash
./scripts/train_pipeline.sh \
    --epochs 20 \
    --batch-size 4 \
    --vocab-size 4000 \
    --learning-rate 0.0005 \
    --max-seq-length 64 \
    --output-dir ./examples/low-memory
```

**Use case:** Raspberry Pi Zero 2W, embedded systems  
**Training time:** ~12 minutes  
**Memory usage:** ~40 MB  
**Model quality:** Medium

---

## Balanced Default (10-minute training)

Recommended starting point:

```bash
./scripts/train_pipeline.sh \
    --epochs 20 \
    --batch-size 32 \
    --vocab-size 8000 \
    --learning-rate 0.0003 \
    --max-seq-length 128 \
    --output-dir ./data
```

**Use case:** General-purpose, good balance  
**Training time:** ~10 minutes  
**Memory usage:** ~150 MB  
**Model quality:** Good

---

## Custom Corpus Example

Train on your own text data:

```bash
# 1. Prepare your corpus (plain text file)
cat > my_corpus.txt << 'EOF'
This is my custom training data.
It can be in any language or domain.
The model will learn patterns from this text.
EOF

# 2. Run pipeline with custom corpus
./scripts/train_pipeline.sh \
    --epochs 20 \
    --vocab-size 2000 \
    --output-dir ./examples/custom-corpus \
    --skip-corpus  # Skip default corpus download

# Note: You'll need to manually place my_corpus.txt in output-dir as corpus_bilingual.txt
```

**Use case:** Domain-specific training (legal, medical, code, etc.)

---

## Resume Training

Continue from a checkpoint:

```bash
# 1. Start training
./scripts/train_pipeline.sh --epochs 20 --output-dir ./examples/resume-test

# 2. Interrupt (Ctrl+C) after 5 epochs

# 3. Resume from checkpoint
./scripts/train_pipeline.sh --resume --epochs 20 --output-dir ./examples/resume-test
```

**Use case:** Recover from interruptions, extend training

---

## Hyperparameter Tuning Grid

Systematic search for best hyperparameters:

```bash
#!/bin/bash
# Save as: examples/grid_search.sh

for lr in 0.0001 0.0003 0.001; do
    for bs in 16 32 64; do
        OUTPUT_DIR="./examples/grid/lr${lr}_bs${bs}"
        echo "Training with LR=$lr, BS=$bs..."
        
        ./scripts/train_pipeline.sh \
            --epochs 10 \
            --learning-rate $lr \
            --batch-size $bs \
            --output-dir $OUTPUT_DIR
    done
done

# Compare results
echo "Results:"
for dir in ./examples/grid/*; do
    if [ -f "$dir/training_metrics.json" ]; then
        val_loss=$(jq '.final_metrics.final_val_loss' "$dir/training_metrics.json")
        echo "  $dir: val_loss=$val_loss"
    fi
done | sort -t= -k2 -n
```

**Use case:** Finding optimal hyperparameters for your use case

---

## Configuration Comparison

| Config | Epochs | Batch Size | Vocab Size | Time | Memory | Quality |
|--------|--------|-----------|-----------|------|--------|---------|
| **Quick Test** | 2 | 8 | 1000 | 2 min | 50 MB | Low |
| **Low Memory** | 20 | 4 | 4000 | 12 min | 40 MB | Medium |
| **Balanced** | 20 | 32 | 8000 | 10 min | 150 MB | Good |
| **High Quality** | 30 | 64 | 16000 | 15 min | 300 MB | High |

Choose based on your constraints and requirements.
