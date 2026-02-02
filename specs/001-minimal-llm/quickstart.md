# Quickstart: Minimal On-Device LLM (PyTorch)

## Prerequisites
- Python 3.11
- CPU-only environment (1GB device target)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch sentencepiece numpy pytest
```

## Run CLI (placeholder paths)

```bash
python -m src.cli.minimal_llm generate --prompt "Hola, escribe un resumen corto" --max_tokens 64
```

## Measure Metrics
- Record next-token p95, tokens/sec, RSS, peak memory
- Ensure: RSS ≤ 400MB; Peak ≤ 512MB; p95 ≤ 250ms

## Safety
- Safety checks run locally; unsafe prompts are refused or sanitized.

## Notes
- This quickstart uses CPU-only settings; optional acceleration may be added later if available.
