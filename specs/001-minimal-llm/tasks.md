---

description: "Task list for Minimal On-Device LLM (PyTorch, 1GB target)"
---

# Tasks: Minimal On-Device LLM

**Input**: Design documents from `/specs/001-minimal-llm/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Include core unit/integration tests; performance/safety profiling tasks are included per constitution.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- Checkbox starts each line: `- [ ]`
- TaskID: Sequential (T001, T002, ...)
- `[P]` only if parallelizable (different files, no dependencies on incomplete tasks)
- `[US#]` only for user story phases (US1, US2, US3)

## Path Conventions

- Single project: `src/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan
- [X] T002 Initialize Python project with PyTorch + sentencepiece in requirements.txt
- [X] T003 [P] Configure linting and formatting tools (ruff/black) in .vscode/
- [X] T004 Setup `src/` and `tests/` folders per plan

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Create tokenizer module in src/models/tokenizer.py
- [X] T006 [P] Implement tiny Transformer skeleton in src/models/tiny_transformer.py
- [X] T007 [P] Implement quantization utilities (int8) in src/models/quantization.py
- [X] T008 Create runtime scaffolding in src/lib/runtime.py
- [X] T009 Configure basic CLI in src/cli/minimal_llm.py
- [X] T010 Setup perf/memory profiling helper in tests/integration/test_offline_generation.py
- [X] T011 Configure safety policy file and loader in src/services/safety.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - On-device text completion (Priority: P1) 🎯 MVP

**Goal**: Generate text completions offline on 1GB devices within performance/memory targets
**Independent Test Criteria**: 50-token response offline meets SC-005..SC-008

### Implementation for User Story 1

- [X] T012 [P] [US1] Implement tokenization pipeline in src/models/tokenizer.py
- [X] T013 [P] [US1] Implement forward pass + generation loop in src/services/generate.py
- [X] T014 [US1] Wire CLI command to generation service in src/cli/minimal_llm.py
- [X] T015 [US1] Add resource budget checks (RSS, peak) in src/lib/runtime.py
- [X] T016 [US1] Add latency/tokens/sec measurement in src/lib/runtime.py
- [X] T017 [US1] Log metrics and return via CLI output formatting

**Checkpoint**: User Story 1 fully functional and testable independently

---

## Phase 4: User Story 2 - Model bundle initialization & management (Priority: P2)

**Goal**: Initialize and manage model bundle quickly and safely
**Independent Test Criteria**: Cold start initialization meets budgets; integrity/version verified

### Implementation for User Story 2

- [X] T018 [P] [US2] Implement bundle metadata handling in src/lib/runtime.py
- [X] T019 [US2] Add integrity verification (SHA-256) in src/lib/runtime.py
- [X] T020 [US2] Add semantic version compatibility checks in src/lib/runtime.py
- [X] T021 [US2] Implement memory-mapped weights loader in src/models/quantization.py
- [X] T022 [US2] Implement cold-start initialization path in src/lib/runtime.py
- [X] T023 [US2] CLI command to show bundle info in src/cli/minimal_llm.py

**Checkpoint**: User Stories 1 and 2 both work independently

---

## Phase 5: User Story 3 - On-device safety filtering (Priority: P3)

**Goal**: Enforce safety policies locally and return safe outputs or refusals
**Independent Test Criteria**: Unsafe prompts refused/sanitized per policy; auditable outcomes

### Implementation for User Story 3

- [X] T024 [P] [US3] Implement minimal safety classifier/rules in src/services/safety.py
- [X] T025 [US3] Integrate safety decision into generation in src/services/generate.py
- [X] T026 [US3] Add auditable logs and rationale mapping in src/services/safety.py
- [X] T027 [US3] CLI flag for safety mode and reporting in src/cli/minimal_llm.py

**Checkpoint**: All user stories independently functional

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T028 [P] Documentation updates in docs/
- [X] T029 Code cleanup and refactoring
- [X] T030 Performance optimization across all stories
- [X] T031 [P] Additional unit tests in tests/unit/
- [X] T032 Security hardening
- [X] T033 Run quickstart.md validation

**Mobile/1GB LLM Constitution Gates**

- [X] T034 Perf profiling: next-token p95 ≤ 250 ms; tokens/sec target documented and met
- [X] T035 Energy profiling: ≤ 5% battery drain per 30 minutes continuous inference
- [X] T036 Memory profiling: runtime RSS ≤ 400 MB; peak ≤ 512 MB on 1GB device class
- [X] T037 On-device safety tests: validate policy enforcement for unsafe prompts
- [X] T038 Compatibility audit: confirm semantic versioning and migration notes

---

## Dependencies & Execution Order

- US1 → US2 → US3 (delivery order)
- US1 has no runtime dependency on US2; US2 provides bundle management features; US3 integrates with US1 generation.
- Foundational tasks T005–T011 must complete before US1–US3 begin.

---

## Parallel Example: User Story 1

```bash
# Parallelizable tasks for US1:
Task: "Implement tokenization pipeline in src/models/tokenizer.py" (T012)
Task: "Implement generation loop in src/services/generate.py" (T013)
Task: "Add resource budget checks in src/lib/runtime.py" (T015)
Task: "Add latency/tokens/sec measurement in src/lib/runtime.py" (T016)
```

---

## Implementation Strategy

- MVP first: Deliver US1 completion offline with strict budgets.
- Incremental delivery: Add US2 bundle management, then US3 safety.
- Validate performance, memory, and safety at each checkpoint.
