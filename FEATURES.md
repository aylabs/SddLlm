# Minimal On-Device LLM - Features & Architecture

## Overview

This is a **minimal Large Language Model** designed to run entirely on-device on smartphones with as little as 1GB of RAM. The implementation prioritizes privacy, resource efficiency, and responsible AI principles while maintaining offline functionality.

## Current Features

### 🔐 Privacy-First Architecture
- **100% Offline Operation**: No network calls, all processing happens locally
- **On-Device Inference**: Text generation runs entirely on the device CPU
- **Local Safety Enforcement**: Content filtering happens without sending data to external services
- **No Data Collection**: No telemetry, analytics, or usage tracking

### 🚀 Text Generation
- **Context Window**: Up to 1,000 tokens per conversation
- **Bilingual Support**: English and Spanish text generation
- **Temperature Control**: Adjustable randomness in generation (0.0 = deterministic, 1.0 = creative)
- **Streaming Output**: Token-by-token generation for responsive user experience
- **Special Tokens**: Support for beginning-of-sequence (BOS), end-of-sequence (EOS), and padding (PAD)

### 📦 Model Bundle Management
- **Integrity Verification**: SHA-256 checksums ensure model files haven't been tampered with
- **Semantic Versioning**: MAJOR.MINOR.PATCH versioning with compatibility checks
- **Memory-Mapped Loading**: Efficient weight loading that avoids loading entire model into RAM
- **Cold-Start Optimization**: Fast initialization from disk
- **Bundle Info Command**: Query model metadata, version, and size information

### 🛡️ Safety & Responsible AI
- **5-Category Safety Filter**: Violence, sexual content, hate speech, self-harm, illegal activities
- **Keyword-Based Detection**: Fast, deterministic safety checks
- **Auditable Logging**: All safety decisions logged with rationale for transparency
- **Configurable Modes**: Strict or normal safety enforcement
- **Refusal with Explanation**: Clear rationale provided when content is blocked

### ⚡ Performance Metrics
- **Real-Time Monitoring**: Track latency, throughput, and memory usage
- **P95 Latency Tracking**: 95th percentile next-token generation time
- **Tokens per Second**: Measure generation speed
- **Memory Profiling**: RSS (resident set size) and peak memory tracking
- **Constitutional Budgets**: Automatic validation against resource limits

### 🧪 Testing & Quality
- **13 Automated Tests**: Unit, integration, and contract validation
- **Budget Compliance Tests**: Verify memory and latency constraints
- **Safety Policy Tests**: Validate content filtering accuracy
- **API Contract Tests**: Ensure consistent response structure

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│           CLI Interface                     │
│  (src/cli/minimal_llm.py)                  │
└──────────────┬──────────────────────────────┘
               │
               ├──► generate command
               │    └──► Generation Service
               │
               └──► bundle-info command
                    └──► Runtime Utilities
                    
┌─────────────────────────────────────────────┐
│       Generation Pipeline                   │
├─────────────────────────────────────────────┤
│                                             │
│  1. Tokenization (SentencePiece)           │
│     ↓                                       │
│  2. Safety Check (keyword filter)          │
│     ↓                                       │
│  3. Model Inference (TinyTransformer)      │
│     ↓                                       │
│  4. Token Sampling (temperature)           │
│     ↓                                       │
│  5. Detokenization (text output)           │
│     ↓                                       │
│  6. Metrics Collection (latency, memory)   │
│                                             │
└─────────────────────────────────────────────┘
```

### Model Architecture: TinyTransformer

**Specifications**:
- **Parameters**: < 10 million (int8 quantized)
- **Layers**: 2 Transformer encoder layers
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Feed-Forward Hidden**: 256
- **Vocabulary Size**: 8,000 tokens
- **Maximum Sequence Length**: 1,000 tokens

**Components**:
1. **Token Embedding**: Maps token IDs to 128-dimensional vectors
2. **Positional Encoding**: Adds position information to embeddings
3. **Transformer Layers**: 2 layers of self-attention + feed-forward networks
4. **Language Model Head**: Linear projection to vocabulary logits
5. **int8 Quantization**: Reduces model size by ~75% with minimal quality loss

### Tokenization

- **Type**: SentencePiece-compatible subword tokenizer
- **Vocabulary**: 8,000 subword units
- **Languages**: English + Spanish (bilingual)
- **Special Tokens**: 
  - `<BOS>` (ID: 0) - Beginning of sequence
  - `<EOS>` (ID: 1) - End of sequence  
  - `<PAD>` (ID: 2) - Padding
- **Truncation**: Automatic truncation at 1,000 tokens

### Safety Filtering

**Categories**:
1. **Violence**: Graphic violence, weapons, physical harm
2. **Sexual**: Explicit sexual content, adult material
3. **Hate**: Hate speech, discrimination, slurs
4. **Self-Harm**: Suicide, self-injury content
5. **Illegal**: Criminal activities, drug manufacturing

**Detection Method**: Keyword-based pattern matching with category-specific banned word lists

**Output**: 
- Status: `allow` or `refuse`
- Category: Which safety category triggered (if refused)
- Rationale: Human-readable explanation

### Memory Optimization

**Techniques**:
- **int8 Quantization**: Compress weights from float32 to int8 (4× reduction)
- **Memory-Mapped Loading**: Stream weights from disk instead of loading all into RAM
- **Dynamic Quantization**: Apply quantization only to Linear layers (preserve accuracy)
- **Controlled Context**: Hard limit at 1,000 tokens prevents unbounded memory growth

**Memory Budgets**:
- Runtime RSS: ≤ 400 MB
- Peak Memory: ≤ 512 MB
- Model Size on Disk: ~2.5 MB (int8 quantized)

### Performance Characteristics

**Latency**:
- Next-token P95: ≤ 250 ms (target)
- Measured: ~1 ms on modern CPU (placeholder model)
- Cold-start: < 1 second

**Throughput**:
- Tokens/Second: ~1,600+ (measured on Apple Silicon M-series)
- Batch Size: 1 (no batching, single-user device)

**Energy**:
- Target: ≤ 5% battery drain per 30 minutes continuous inference
- CPU-only baseline (no GPU/NPU acceleration yet)

## Current Limitations

### ⚠️ Model Training

**Status**: **NOT TRAINED** - The model currently generates random tokens

**Why**: This implementation provides the complete inference infrastructure, but the model weights are randomly initialized. To generate meaningful text, the model needs:

1. **Training Data**: Large corpus of bilingual EN+ES text
2. **Training Process**: Run gradient descent to learn language patterns
3. **Fine-Tuning**: Optional task-specific or safety fine-tuning
4. **Validation**: Perplexity benchmarks and quality evaluation

**Next Steps**: See "Roadmap" section below for training plans.

### 🔤 Tokenizer Limitations

- **Vocabulary**: Currently using placeholder SentencePiece integration
- **Missing**: Trained bilingual EN+ES vocabulary
- **Impact**: Suboptimal text representation, potential out-of-vocabulary issues

### 🛡️ Safety System

- **Simple Keyword Matching**: Not as robust as ML-based classifiers
- **False Positives**: May flag benign content containing trigger words
- **False Negatives**: Sophisticated adversarial prompts may bypass filters
- **Language Coverage**: Best for English; Spanish coverage needs expansion
- **No Context**: Cannot distinguish harmful vs. educational content

### ⚡ Performance

- **CPU-Only**: No GPU/NPU acceleration yet
- **Single-Threaded**: No parallelization across cores
- **No Batching**: Processes one request at a time
- **Cold Cache**: First run includes model loading overhead

### 🌐 Language Support

- **Limited to EN+ES**: Only English and Spanish planned
- **No Multilingual**: Cannot handle mixed-language inputs well
- **No Code**: Limited programming/code generation support

### 📱 Device Testing

- **Not Tested on Real Devices**: Validation done on development machine (macOS)
- **Energy Metrics**: Battery drain not yet measured on actual smartphones
- **Thermal Behavior**: Heat generation under sustained load unknown
- **Different Chipsets**: Performance may vary on ARM vs x86, different vendors

## Resource Requirements

### Minimum Requirements

- **RAM**: 512 MB available (1 GB device recommended)
- **Storage**: 50 MB (model + dependencies)
- **CPU**: Any modern ARM or x86 processor
- **OS**: Linux, macOS, Windows (Python 3.11 compatible)

### Development Requirements

- Python 3.11
- PyTorch 2.x
- sentencepiece
- numpy
- pytest (for testing)

### Installed Size Breakdown

- PyTorch: ~80 MB
- Model weights: ~2.5 MB (int8 quantized)
- sentencepiece: ~1.3 MB
- Source code: < 1 MB
- **Total**: ~85 MB

## Usage Examples

### Basic Text Generation

```bash
python -m src.cli.minimal_llm generate \
  --prompt "Hello, how are you?" \
  --max_tokens 50 \
  --temperature 0.7
```

### Spanish Generation

```bash
python -m src.cli.minimal_llm generate \
  --prompt "Hola, escribe un resumen corto" \
  --max_tokens 64 \
  --temperature 0.8
```

### Strict Safety Mode

```bash
python -m src.cli.minimal_llm generate \
  --prompt "Tell me a story" \
  --max_tokens 100 \
  --safety-mode strict
```

### JSON Output (for integration)

```bash
python -m src.cli.minimal_llm generate \
  --prompt "What is AI?" \
  --max_tokens 50 \
  --json
```

Output:
```json
{
  "text": "tok1234 tok5678 ...",
  "metrics": {
    "latency_p95_ms": 1,
    "tokens_per_sec": 1664.76,
    "rss_mb": 212,
    "peak_mb": 212
  },
  "safety": {
    "status": "allow",
    "category": null,
    "rationale": "Prompt passed safety checks"
  }
}
```

### Query Bundle Information

```bash
python -m src.cli.minimal_llm bundle-info --metadata-path /path/to/bundle_metadata.json
```

## Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

**Test Coverage**:
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - End-to-end generation scenarios
- `tests/contract/` - API contract validation

**Current Status**: 13/13 tests passing ✅

## Roadmap

### Phase 1: Model Training (Next Priority)
- [ ] Acquire/curate bilingual EN+ES training dataset
- [ ] Train bilingual tokenizer with 8K vocabulary
- [ ] Pre-train TinyTransformer on language modeling task
- [ ] Evaluate perplexity and sample quality
- [ ] Create trained model bundle with metadata

### Phase 2: Quality Improvements
- [ ] Implement beam search for better generation
- [ ] Add top-k and nucleus (top-p) sampling options
- [ ] Fine-tune for instruction following
- [ ] Human evaluation of output quality
- [ ] A/B testing of different configurations

### Phase 3: Safety Enhancements
- [ ] Expand keyword lists based on real-world testing
- [ ] Consider lightweight ML-based safety classifier
- [ ] Add content sanitization (not just refusal)
- [ ] Multi-language safety coverage
- [ ] Red-team testing for adversarial prompts

### Phase 4: Performance Optimization
- [ ] Multi-threading for parallel token generation
- [ ] ONNX export for cross-platform optimization
- [ ] Investigate NPU/GPU acceleration on mobile
- [ ] Quantization experiments (int4, mixed precision)
- [ ] KV-cache for faster autoregressive generation

### Phase 5: Device Validation
- [ ] Test on real Android devices (various chipsets)
- [ ] Test on iOS devices (A-series chips)
- [ ] Measure actual battery drain over time
- [ ] Profile thermal behavior under load
- [ ] Optimize for specific device classes

### Phase 6: Developer Experience
- [ ] Python SDK for easy integration
- [ ] Mobile bindings (Swift for iOS, Kotlin for Android)
- [ ] Model compression tools
- [ ] Fine-tuning scripts
- [ ] Deployment guides

## Contributing

This project follows the **SpecKit** development methodology:

1. **Constitution**: Core principles in `.specify/memory/constitution.md`
2. **Specification**: Feature specs in `specs/001-minimal-llm/spec.md`
3. **Planning**: Implementation plan in `specs/001-minimal-llm/plan.md`
4. **Tasks**: Breakdown in `specs/001-minimal-llm/tasks.md`
5. **Implementation**: Source in `src/`, tests in `tests/`

## License

[To be determined]

## Acknowledgments

Built with PyTorch, SentencePiece, and following constitutional principles for responsible on-device AI.

---

**Version**: 1.0.0  
**Last Updated**: February 2, 2026  
**Status**: ✅ Infrastructure Complete | ⏳ Model Training Pending
