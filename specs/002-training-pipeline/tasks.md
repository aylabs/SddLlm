# Implementation Tasks: Training Pipeline Automation & Documentation

**Feature**: 002-training-pipeline  
**Branch**: `002-training-pipeline`  
**Created**: 2026-02-04  
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Task Summary

**Total Tasks**: 23  
**Parallelizable**: 8 tasks marked with [P]  
**User Stories**: 3 (P1: Automation, P2: Documentation, P3: Configuration)

## Implementation Strategy

This feature builds upon existing training scripts from feature 001. The strategy is:
1. **MVP First**: User Story 1 (P1) - Automated pipeline delivers immediate value
2. **Educational Layer**: User Story 2 (P2) - Documentation enables team learning
3. **Power User Features**: User Story 3 (P3) - Advanced configuration for experimentation

All tasks organized by user story for independent implementation and testing.

---

## Phase 1: Setup & Prerequisites

**Goal**: Establish project structure and validation framework

- [ ] T001 Verify existing training scripts functional (download, tokenizer, model)
- [ ] T002 Create tests/integration/ and tests/unit/ directories if not exist
- [ ] T003 Create docs/ directory for training documentation

---

## Phase 2: Foundational Tasks (Complete Before User Stories)

**Goal**: Shared infrastructure needed by all user stories

- [X] T004 [P] Update scripts/train_model.py to export training_metrics.json with epoch history
- [X] T005 [P] Update scripts/train_model.py to accept --output-dir parameter
- [X] T006 [P] Update scripts/train_tokenizer.py to accept --output-dir parameter

---

## Phase 3: User Story 1 - Automated End-to-End Training (P1)

**Goal**: Single-command training from corpus to validated model

**Story Label**: [US1]

**Independent Test**: Run `./scripts/train_pipeline.sh` and verify all artifacts created

### Orchestration Script

- [X] T007 [US1] Create scripts/train_pipeline.sh with shebang and basic structure
- [X] T008 [US1] Implement prerequisite check function in train_pipeline.sh (disk space, Python version, packages)
- [X] T009 [US1] Implement CLI argument parsing in train_pipeline.sh (--epochs, --batch-size, --learning-rate, --vocab-size, --skip-corpus, --skip-tokenizer, --resume, --output-dir)
- [X] T010 [US1] Implement corpus download phase in train_pipeline.sh (call download_simple_corpus.py with error handling)
- [X] T011 [US1] Implement tokenizer training phase in train_pipeline.sh (call train_tokenizer.py with vocab-size parameter)
- [X] T012 [US1] Implement model training phase in train_pipeline.sh (call train_model.py with hyperparameters)
- [X] T013 [US1] Implement validation phase in train_pipeline.sh (test sample generation, check inference metrics)
- [X] T014 [US1] Implement resume logic in train_pipeline.sh (detect checkpoints, prompt user or auto-resume with --resume flag)
- [X] T015 [US1] Add progress indicators in train_pipeline.sh (phase headers, timing, artifact summaries)
- [X] T016 [US1] Add error handling with actionable messages in train_pipeline.sh (exit codes 1-5 for different failure types)

### Testing

- [X] T017 [US1] Create tests/integration/test_training_pipeline.py with minimal corpus test (50 lines, 2 epochs, validate artifacts created)
- [X] T018 [US1] Create tests/unit/test_training_scripts.py to validate prerequisite checks, argument parsing, file handling

### Validation

- [X] T019 [US1] Run full pipeline end-to-end and verify SC-001 (< 10 minutes for 20 epochs)
- [X] T020 [US1] Test resume functionality by interrupting training and resuming
- [X] T021 [US1] Validate constitutional compliance metrics (RSS ≤ 400MB, p95 ≤ 250ms) in validation phase

---

## Phase 4: User Story 2 - Understanding Training Process (P2)

**Goal**: Comprehensive documentation for learning and troubleshooting

**Story Label**: [US2]

**Independent Test**: New team member reads doc and successfully explains training process

### Documentation Content

- [X] T022 [P] [US2] Create docs/TRAINING_GUIDE.md with structure (10 section headers, TOC)
- [X] T023 [P] [US2] Write Section 1: Training Overview with workflow diagram in docs/TRAINING_GUIDE.md
- [X] T024 [P] [US2] Write Section 2: Tokenization with 2 diagrams and 2 code examples in docs/TRAINING_GUIDE.md
- [X] T025 [P] [US2] Write Section 3: Model Initialization with architecture diagram and code example in docs/TRAINING_GUIDE.md
- [X] T026 [P] [US2] Write Section 4: Data Preparation with batching diagram and dataset code in docs/TRAINING_GUIDE.md
- [X] T027 [US2] Write Section 5: Training Loop with 3 diagrams (forward, attention, backprop) and 2 code examples in docs/TRAINING_GUIDE.md
- [X] T028 [US2] Write Section 6: Optimization with LR schedule diagram and optimizer code in docs/TRAINING_GUIDE.md
- [X] T029 [US2] Write Section 7: Evaluation with checkpointing diagram and save/load examples in docs/TRAINING_GUIDE.md
- [X] T030 [US2] Write Section 8: Troubleshooting with problem-solution matrix table covering 10+ issues in docs/TRAINING_GUIDE.md
- [X] T031 [US2] Write Section 9: Hyperparameter Reference with tuning guide table in docs/TRAINING_GUIDE.md
- [X] T032 [US2] Write Section 10: Constitutional Compliance with memory profiling and deployment checklist in docs/TRAINING_GUIDE.md

### Documentation Validation

- [ ] T033 [US2] Validate all code examples in TRAINING_GUIDE.md are runnable without modification
- [ ] T034 [US2] Verify TRAINING_GUIDE.md meets quality criteria (1500-2500 lines, 10+ diagrams, 10+ code examples, 3+ tables)
- [ ] T035 [US2] Cross-reference TRAINING_GUIDE.md sections and ensure glossary/troubleshooting completeness

---

## Phase 5: User Story 3 - Training Configuration Flexibility (P3)

**Goal**: Enable hyperparameter experimentation via CLI arguments

**Story Label**: [US3]

**Independent Test**: Run pipeline with different configs and verify behavior changes

### Configuration Enhancement

- [X] T036 [US3] Add --help flag to train_pipeline.sh with usage examples and parameter descriptions
- [X] T037 [US3] Document default vs custom configuration patterns in quickstart.md
- [X] T038 [US3] Add configuration validation in train_pipeline.sh (check parameter ranges, warn about memory implications)
- [X] T039 [US3] Update scripts/train_model.py to log hyperparameters to training_metrics.json for reproducibility
- [X] T040 [US3] Create examples/ directory with sample configurations for common use cases (quick test, high quality, low memory)

### Testing

- [X] T041 [US3] Test pipeline with --learning-rate 0.0001 --epochs 10 and verify parameters applied
- [X] T042 [US3] Test pipeline with --batch-size 64 and verify faster completion with memory impact
- [X] T043 [US3] Test pipeline with custom --corpus-path and verify training on provided data

---

## Phase 6: Polish & Cross-Cutting Concerns

**Goal**: Final quality, integration, and documentation polish

- [X] T044 Make scripts/train_pipeline.sh executable (chmod +x)
- [X] T045 Update main README.md to reference training pipeline and documentation
- [X] T046 Add training pipeline example to quickstart.md if not already present
- [X] T047 Run full test suite (pytest tests/) and verify all tests pass
- [ ] T048 Generate training run with default config and commit training_metrics.json as reference
- [ ] T049 Perform constitutional compliance validation on trained artifacts (verify inference metrics)
- [X] T050 Update .gitignore to exclude large checkpoint files but keep best_model.pt and final_model.pt

---

## Dependencies & Parallel Execution

### Dependency Graph (User Story Completion Order)

```
Setup (Phase 1) 
    ↓
Foundational (Phase 2)
    ↓
US1 (Phase 3) ────────→ US2 (Phase 4) ────→ US3 (Phase 5)
    ↓                        ↓                    ↓
    └────────────────────────┴────────────────────┘
                             ↓
                    Polish (Phase 6)
```

**Key Dependencies**:
- Phase 2 (Foundational) must complete before any user story
- US1 (Pipeline Script) must complete before US3 (Configuration), as US3 enhances US1
- US2 (Documentation) is independent and can run parallel to US1/US3
- Phase 6 (Polish) requires all user stories complete

### Parallel Execution Opportunities

**Phase 2 (Foundational)**: T004, T005, T006 can run in parallel (different files)

**Phase 4 (US2 Documentation)**: T022-T026 can run in parallel (different sections of same file, but independent content)

**Within Each User Story**:
- US1: Tasks are sequential (orchestration script builds incrementally)
- US2: Documentation sections can be written in parallel by different team members
- US3: Configuration tasks depend on US1 completion

**Suggested Parallel Batches**:

Batch 1 (after Phase 1):
- T004 (train_model.py metrics export)
- T005 (train_model.py output-dir)
- T006 (train_tokenizer.py output-dir)

Batch 2 (after US1 core complete, T007-T013):
- T022 (TRAINING_GUIDE.md structure)
- T023 (Section 1: Overview)
- T024 (Section 2: Tokenization)
- T025 (Section 3: Initialization)
- T026 (Section 4: Data Prep)

Batch 3 (documentation continuation):
- T027 (Section 5: Training Loop)
- T028 (Section 6: Optimization)
- T029 (Section 7: Evaluation)

Batch 4 (documentation finish):
- T030 (Section 8: Troubleshooting)
- T031 (Section 9: Hyperparameters)
- T032 (Section 10: Compliance)

---

## Implementation Notes

### MVP Delivery (Minimum Viable Product)

To deliver value early, implement **User Story 1 only**:
- Tasks T001-T021 (Setup + Foundational + US1)
- Deliverable: Working end-to-end training pipeline
- Test: `./scripts/train_pipeline.sh` produces trained model
- Timeline: ~1 day of focused work

### Incremental Delivery

**Sprint 1**: US1 (Automation) - T001-T021  
**Sprint 2**: US2 (Documentation) - T022-T035  
**Sprint 3**: US3 (Configuration) + Polish - T036-T050

### Task Execution Tips

1. **Start with tests**: Create test files (T017, T018) early to validate development
2. **Incremental orchestration**: Build train_pipeline.sh phase by phase (T007-T016)
3. **Documentation in parallel**: Multiple team members can write different sections simultaneously
4. **Validate continuously**: Run pipeline after each major task to catch issues early
5. **Constitutional checks**: Validate inference metrics throughout (not just at end)

### File Change Summary

**New Files Created**:
- scripts/train_pipeline.sh (~200 lines)
- docs/TRAINING_GUIDE.md (~2000 lines)
- tests/integration/test_training_pipeline.py (~150 lines)
- tests/unit/test_training_scripts.py (~100 lines)
- examples/*.txt (optional config examples)

**Files Modified**:
- scripts/train_model.py (add --output-dir, export metrics JSON)
- scripts/train_tokenizer.py (add --output-dir parameter)
- quickstart.md (add pipeline examples)
- README.md (reference training docs)
- .gitignore (exclude large checkpoints)

**Total Lines**: ~2500 new lines (mostly documentation)

---

## Success Validation Checklist

After completing all tasks, verify:

- [ ] SC-001: Training completes in < 10 minutes (20 epochs, 3MB corpus)
- [ ] SC-002: Pipeline runs successfully on fresh environment (no manual steps)
- [ ] SC-003: Final validation loss < 5.0 after 20 epochs
- [ ] SC-004: Documentation enables process explanation (team review)
- [ ] SC-005: Consistent results across runs (loss variance < 5%)
- [ ] SC-006: Troubleshooting guide addresses common issues
- [ ] SC-007: Resume functionality works after interruption
- [ ] SC-008: Training metrics logged to JSON
- [ ] SC-009: Generated text is coherent (no repetition)
- [ ] SC-010: Bilingual capability demonstrated
- [ ] SC-011: Training loss decreases monotonically
- [ ] SC-012: No significant overfitting (val_loss - train_loss < 0.5)
- [ ] SC-013: Clear progress indicators throughout pipeline
- [ ] SC-014: Error messages provide actionable guidance
- [ ] SC-015: All documentation code examples runnable

---

## Task Completion Tracking

**Phase 1 (Setup)**: 0/3 tasks complete  
**Phase 2 (Foundational)**: 0/3 tasks complete  
**Phase 3 (US1 - Automation)**: 0/15 tasks complete  
**Phase 4 (US2 - Documentation)**: 0/14 tasks complete  
**Phase 5 (US3 - Configuration)**: 0/8 tasks complete  
**Phase 6 (Polish)**: 0/7 tasks complete

**Overall Progress**: 0/50 tasks complete (0%)

---

## Notes

- All existing training scripts (download, tokenizer, model) are functional from feature 001
- Focus is on orchestration, documentation, and usability
- Constitutional compliance already validated in feature 001; revalidate with pipeline
- Documentation is substantial (~2000 lines) but high value for team learning
- Pipeline script is simple bash orchestration (~200 lines) wrapping existing Python scripts
