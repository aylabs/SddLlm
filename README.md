# Minimal On-Device LLM

A PyTorch-based minimal language model designed to run on 1GB RAM devices with strict performance and privacy guarantees.

## Features

- **On-device privacy**: All inference runs locally, no data leaves the device
- **Resource efficient**: Runtime RSS ≤ 400MB, peak ≤ 512MB
- **Fast inference**: Next-token p95 ≤ 250ms
- **Bilingual**: English + Spanish support
- **Safety first**: Local safety filtering with auditable policies

## Quick Start

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate text
python -m src.cli.minimal_llm generate --prompt "Hello world" --max_tokens 50

# Show bundle info
python -m src.cli.minimal_llm bundle-info --metadata bundle_metadata.json

# Run tests
pytest tests/
```

## Architecture

- **Model**: Tiny Transformer (2 layers, d_model=128, 4 heads)
- **Quantization**: int8 dynamic quantization
- **Context**: 1k tokens max
- **Vocab**: ~8k tokens (bilingual EN+ES)

## Constitution Compliance

This project adheres to the SDDLLM constitution v1.0.0:
- Privacy: On-device only
- Performance: Documented budgets and measurements
- Testing: Unit, integration, and profiling tests
- Safety: Local enforcement
- Compatibility: Semantic versioning

## License

MIT
