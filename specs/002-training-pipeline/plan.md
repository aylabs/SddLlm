# Implementation Plan: Training Pipeline Automation & Documentation

**Branch**: `002-training-pipeline` | **Date**: 2026-02-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-training-pipeline/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature provides automated training pipeline orchestration and comprehensive documentation for LLM training. The primary requirement is to enable developers to train a complete model from scratch with a single command, while understanding the training process through detailed documentation. The technical approach leverages existing PyTorch training infrastructure (from feature 001) with shell-based orchestration for the unified pipeline and Markdown documentation with diagrams for the educational content.

## Technical Context

**Language/Version**: Python 3.11+ (consistent with feature 001)  
**Primary Dependencies**: PyTorch 2.x (CPU), SentencePiece 0.2.x, tqdm (progress bars), existing TinyTransformer model  
**Storage**: Local filesystem for corpus, tokenizer models, training checkpoints (requires ~2GB disk space)  
**Testing**: pytest for pipeline integration tests; manual validation of documentation completeness  
**Target Platform**: macOS/Linux CPU-only environments (1GB device training workflow)  
**Project Type**: Single project with scripts/ directory for automation and docs/ for documentation  
**Performance Goals**: Complete training in <10 minutes for 20 epochs on 3MB corpus; clear progress indicators  
**Constraints**: Training must respect 1GB device budgets (even though training may occur on dev machine, trained artifacts must fit budget); resume capability for long-running processes  
**Scale/Scope**: 3 shell scripts (download, train-tokenizer, train-model unified), 1 comprehensive training guide document (~2000 lines Markdown), ~5 diagram illustrations

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Privacy Compliance**: ✅ PASS - Training pipeline operates entirely locally on developer machine. Corpus data (public domain texts) never leaves device. No telemetry or cloud dependencies.

**Memory/Energy Compliance**: ✅ PASS - Training process itself may use >400MB on dev machine (acceptable for development), but trained artifacts (model checkpoints, tokenizer files) remain within 1GB device budgets:
- Trained model: best_model.pt ~10MB (2.45M params × 4 bytes)
- Tokenizer: bilingual_8k.model ~374KB
- Runtime RSS during inference: 246-253MB (verified in feature 001)
- Peak memory during inference: 251-253MB (verified in feature 001)

Training documentation will emphasize that training is a development-time activity; deployment uses trained artifacts within budget.

**Performance Compliance**: ✅ PASS - Documentation will reinforce device-class targets from constitution:
- Next-token p95 ≤ 250ms (achieved: 0-1ms in feature 001)
- Tokens/sec: ~1300-1400 (measured in feature 001)
- Test matrix: CPU-only on 1GB-class devices (smartphones)

Training pipeline includes validation step to generate sample outputs and measure inference metrics post-training.

**Testing Compliance**: ✅ PASS - Will include:
- Integration test: Run full pipeline on minimal corpus, validate artifacts created
- Unit tests: Individual script validation (prerequisite checks, file parsing, error handling)
- Reproducible benchmarks: Training metrics logged to JSON for comparison across runs
- Performance validation: Trained model tested for inference latency/memory (reuse feature 001 tests)

**Compatibility Compliance**: ✅ PASS - Training pipeline generates versioned artifacts:
- Model checkpoints include config metadata for backward compatibility checks
- Tokenizer vocabulary frozen at 8000 tokens for consistency
- Training documentation explains semantic versioning for model releases
- Breaking changes: Retraining with different vocab size would be MAJOR version change

**Safety Compliance**: ✅ PASS - Training documentation includes section on:
- Corpus selection criteria (avoiding harmful content in training data)
- Safety filter validation post-training (test with unsafe prompts)
- Model behavior monitoring during training (sample generations for quality)
- Reference to feature 001 safety policies

**Assessment**: All constitutional requirements met. Training is development-time activity; runtime artifacts comply with 1GB budgets. Training documentation reinforces constitutional principles.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Single project structure (existing from feature 001)
scripts/                          # Training automation scripts
├── download_simple_corpus.py     # [EXISTS] Downloads bilingual corpus from Project Gutenberg
├── train_tokenizer.py            # [EXISTS] Trains SentencePiece tokenizer
├── train_model.py                # [EXISTS] Trains TinyTransformer model
└── train_pipeline.sh             # [NEW] Unified orchestration script

docs/                             # Comprehensive documentation
└── TRAINING_GUIDE.md             # [NEW] ~2000 line training explanation with diagrams

src/                              # [EXISTS from feature 001]
├── models/                       
│   ├── tiny_transformer.py       # [EXISTS] Model architecture
│   ├── tokenizer.py              # [EXISTS] Tokenizer wrapper
│   └── quantization.py           # [EXISTS] Quantization utilities
├── services/
│   ├── generate.py               # [EXISTS] Generation service
│   └── safety.py                 # [EXISTS] Safety filtering
├── cli/
│   └── minimal_llm.py            # [EXISTS] CLI interface
└── lib/
    └── runtime.py                # [EXISTS] Runtime utilities

tests/
├── integration/
│   ├── test_offline_generation.py  # [EXISTS]
│   └── test_training_pipeline.py   # [NEW] End-to-end pipeline test
└── unit/
    ├── test_model_shapes.py         # [EXISTS]
    ├── test_tokenizer.py            # [EXISTS]
    ├── test_safety.py               # [EXISTS]
    └── test_training_scripts.py     # [NEW] Script validation tests

data/                             # Training artifacts
├── corpus_bilingual.txt          # [EXISTS] Training corpus
├── bilingual_8k.model            # [EXISTS] Trained tokenizer
├── bilingual_8k.vocab            # [EXISTS] Vocabulary file
├── best_model.pt                 # [EXISTS] Best checkpoint
├── final_model.pt                # [EXISTS] Final model
└── checkpoint_epoch_*.pt         # [EXISTS] Periodic checkpoints
```

**Structure Decision**: Single project structure maintained from feature 001. New additions are minimal: one shell script for orchestration (scripts/train_pipeline.sh), one documentation file (docs/TRAINING_GUIDE.md), and two test files for pipeline validation.

## Complexity Tracking

No constitutional violations. All design decisions align with simplicity and 1GB constraints.
