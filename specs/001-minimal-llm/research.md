# Research: Minimal On-Device LLM (PyTorch)

**Date**: 2026-02-02
**Branch**: 001-minimal-llm

## Decisions

### Architecture
- **Decision**: Tiny Transformer (≈2 layers, d_model≈128, heads≈4, FF≈256)
- **Rationale**: Simplest modern architecture with acceptable quality; fits 1GB budgets when quantized; easier to maintain than LSTM for tokenizer interoperability.
- **Alternatives considered**: LSTM/RNN (lower compute but weaker text quality), Larger Transformer (better quality but exceeds budgets), Mamba (state-space models; less mature for minimal MVP).

### Tokenizer
- **Decision**: SentencePiece BPE, vocab≈8k, bilingual (EN+ES); special tokens [BOS, EOS, PAD]
- **Rationale**: Smaller vocab reduces memory footprint; bilingual meets spec; widely supported.
- **Alternatives considered**: WordPiece (similar), Byte-level BPE (larger vocab), Unigram LM (slightly different distribution).

### Quantization & Inference
- **Decision**: int8 quantized inference for weights; memory-mapped model file; modest KV-cache budget sized to 1k context.
- **Rationale**: Meets runtime RSS ≤ 400MB and peak ≤ 512MB; int8 widely supported in PyTorch.
- **Alternatives considered**: int4 (smaller but quality risk; consider later), FP16 (too heavy for 1GB), dynamic quantization only (less optimal).

### Safety
- **Decision**: On-device minimal classifier with rule-based fallback; categories: violence, sexual content, hate content, self-harm, illegal activity.
- **Rationale**: Meets constitution; auditable; lightweight to implement.
- **Alternatives considered**: Remote safety service (violates privacy/offline), heavy ML classifier (exceeds budgets).

### Performance Targets & Device Matrix
- **Decision**: Next-token p95 ≤ 250ms; tokens/sec documented per device; measure on representative 1GB devices.
- **Rationale**: Aligns with constitution; practical for MVP.
- **Alternatives considered**: Looser targets (worse UX), stricter targets (may sacrifice quality).

## Open Questions (to track)
- Export path (optional): ONNX vs TorchScript for mobile runtime embedding.
- Exact device list for lab (pick at least one 1GB RAM device).

