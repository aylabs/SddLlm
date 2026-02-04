# Research: Training Pipeline Automation & Documentation

**Feature**: 002-training-pipeline  
**Date**: 2026-02-04  
**Status**: Complete - No research needed

## Overview

This document records research and technical decision-making for the training pipeline automation feature. Since all core technical components were implemented and validated in feature 001 (minimal-llm), no new research was required. This document summarizes the proven approaches that will be reused.

## Technical Decisions from Feature 001

### 1. Shell vs Python for Orchestration

**Decision**: Use bash shell script for `train_pipeline.sh`

**Rationale**:
- Simple sequential orchestration (corpus → tokenizer → model)
- No additional dependencies beyond Python scripts
- Standard Unix tools for prerequisite checks (df, command -v)
- Portable across macOS/Linux
- Easy to read and modify

**Alternatives Considered**:
- Python orchestration script: Rejected due to unnecessary complexity for simple sequential execution
- Makefile: Rejected due to poor error message customization and less intuitive for non-build workflows

**Validation**: Existing scripts (download, train_tokenizer, train_model) successfully execute independently with clear outputs

### 2. Documentation Format

**Decision**: Markdown with ASCII/Unicode diagrams

**Rationale**:
- Renders in GitHub, VS Code, terminal (via cat/less)
- Portable - no special tools needed
- Version-controllable as plain text
- ASCII art diagrams work everywhere (no image hosting needed)
- Code blocks with syntax highlighting

**Alternatives Considered**:
- Jupyter Notebooks: Rejected due to version control challenges and requiring Jupyter to read
- Sphinx/ReadTheDocs: Rejected as overkill for single guide document
- LaTeX/PDF: Rejected due to poor inline code editing and non-plain-text format

**Validation**: Existing DESIGN_LLM.md (1943 lines) and INFERENCE_EXPLAINED.md successfully explain complex concepts with Markdown

### 3. Progress Indication

**Decision**: tqdm library for progress bars in Python scripts

**Rationale**:
- Already dependency in train_model.py
- Clear visual feedback during long training runs
- Shows iteration speed, ETA, percentage complete
- Works in terminal and Jupyter
- Minimal code to integrate

**Alternatives Considered**:
- Simple print statements: Rejected due to poor UX for long-running processes
- Custom progress implementation: Rejected as reinventing the wheel
- No progress indication: Rejected due to poor developer experience

**Validation**: train_model.py successfully uses tqdm for 20-epoch training with clear progress display

### 4. Error Handling Strategy

**Decision**: Bash `set -e` for fail-fast + Python try/except with actionable messages

**Rationale**:
- `set -e` stops pipeline immediately on any command failure
- Python scripts print actionable error messages (not just stack traces)
- Early failure prevents wasted time on subsequent phases
- Clear error codes for different failure types

**Implementation Pattern**:
```bash
#!/usr/bin/env bash
set -e  # Exit on first error
set -u  # Exit on undefined variable
set -o pipefail  # Catch errors in pipes

# Run scripts
python scripts/download_simple_corpus.py || exit 2
python scripts/train_tokenizer.py || exit 3
python scripts/train_model.py || exit 4
```

**Alternatives Considered**:
- Continue on error: Rejected as wastes time (why train model if tokenizer failed?)
- Try/catch all errors: Rejected as hides root causes
- Silent failures: Rejected due to debugging difficulty

**Validation**: Existing scripts fail gracefully with clear messages (e.g., "SentencePiece not found")

### 5. Configuration Approach

**Decision**: CLI arguments (no config files)

**Rationale**:
- Matches feature 001 pattern (src/cli/minimal_llm.py uses argparse)
- Simple to use: `./train_pipeline.sh --epochs 10`
- No file parsing logic needed
- Default values in script (visible, documented)
- Easy to override for experimentation

**Implementation Pattern**:
```bash
# Default configuration
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.0003

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
```

**Alternatives Considered**:
- YAML/JSON config files: Rejected as overkill for 5-6 parameters
- Environment variables: Rejected as less discoverable than --help
- Interactive prompts: Rejected as breaks automation (CI/CD)

**Validation**: train_model.py successfully uses argparse for hyperparameter configuration

### 6. Resume Strategy

**Decision**: Check for existing checkpoints; prompt user interactively

**Rationale**:
- Training can be interrupted (Ctrl+C, system reboot, OOM)
- Checkpoints saved every 5 epochs (from feature 001)
- Interactive prompt prevents accidental overwrite
- Non-interactive mode via --force-restart flag

**Implementation Pattern**:
```bash
if [ -f data/checkpoint_epoch_*.pt ]; then
  echo "Found existing checkpoints. Resume training? [y/N]"
  read -r response
  if [[ "$response" =~ ^[Yy]$ ]]; then
    # Pass --resume to train_model.py
  fi
fi
```

**Alternatives Considered**:
- Always resume: Rejected as prevents fresh start experiments
- Always restart: Rejected as wastes completed epochs
- Automatic based on timestamp: Rejected as less explicit/predictable

**Validation**: Feature 001 checkpoints successfully save epoch, model state, optimizer state for resumption

## Research Areas: None Required

All technical approaches proven in feature 001:
- ✅ PyTorch training loop functional
- ✅ SentencePiece tokenizer training successful
- ✅ Corpus download from Project Gutenberg working
- ✅ Progress indication with tqdm clear and helpful
- ✅ Error handling with try/except provides actionable guidance
- ✅ Checkpoint saving/loading implemented and tested

## Risks & Mitigations

### Risk 1: Pipeline Fails Midway with Unclear Error

**Mitigation**: Each phase logs start/completion with timestamps; shell script prints phase headers; exit codes distinguish failure types (2=corpus, 3=tokenizer, 4=model, 5=validation)

### Risk 2: Documentation Too Technical for New Developers

**Mitigation**: Structure doc with increasing detail levels; start with high-level overview and diagrams; include "Prerequisites" section defining assumed knowledge; use analogies (e.g., "vocabulary is like a dictionary")

### Risk 3: Training Takes Too Long (Poor UX)

**Mitigation**: Document expected timing (~7 minutes for 20 epochs); provide --epochs option for quick testing (e.g., --epochs 5 for 90-second validation); show progress bars with ETA

### Risk 4: Disk Space Exhaustion During Training

**Mitigation**: Prerequisite check at pipeline start (df -h, require 2GB free); option to --skip-corpus if already downloaded; checkpoints saved incrementally (not all at once)

## Implementation Notes

### Bash Script Best Practices
- Use `set -euo pipefail` for strict error handling
- Quote all variables: `"$VAR"` not `$VAR`
- Provide --help with usage examples
- Log to stderr for errors, stdout for progress
- Return meaningful exit codes (0=success, 1-5=specific failures)

### Documentation Structure
- Start with TL;DR / Quick Reference
- Use consistent heading hierarchy (## for major sections, ### for subsections)
- Include Table of Contents for navigation
- Code examples: Complete, runnable, with comments
- Diagrams: Simple ASCII art, max 80 characters wide for terminal viewing

### Testing Strategy
- Integration test: Full pipeline on tiny corpus (1000 lines, 1 epoch, 2 minutes)
- Unit tests: Prerequisite validation, argument parsing, file existence checks
- Validation: Trained model generates non-garbage text
- Metrics: Log to JSON for automated comparison (CI/CD)

## Conclusion

No new research required. All technical approaches validated in feature 001. Implementation can proceed directly to Phase 1 (design) using proven patterns.

**Next Step**: Create data-model.md, contracts/, and quickstart.md per SpecKit Phase 1 workflow.
