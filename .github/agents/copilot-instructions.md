# sddllm Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-02-02

## Active Technologies
- Python 3.11+ (consistent with feature 001) + PyTorch 2.x (CPU), SentencePiece 0.2.x, tqdm (progress bars), existing TinyTransformer model (002-training-pipeline)
- Local filesystem for corpus, tokenizer models, training checkpoints (requires ~2GB disk space) (002-training-pipeline)

- Python 3.11; PyTorch 2.x (CPU-only target) + PyTorch, sentencepiece (tokenizer), onnx export (optional), numpy (001-minimal-llm)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.11; PyTorch 2.x (CPU-only target): Follow standard conventions

## Recent Changes
- 002-training-pipeline: Added Python 3.11+ (consistent with feature 001) + PyTorch 2.x (CPU), SentencePiece 0.2.x, tqdm (progress bars), existing TinyTransformer model

- 001-minimal-llm: Added Python 3.11; PyTorch 2.x (CPU-only target) + PyTorch, sentencepiece (tokenizer), onnx export (optional), numpy

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
