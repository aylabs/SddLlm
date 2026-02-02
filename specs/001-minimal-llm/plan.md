# Implementation Plan: Minimal On-Device LLM

**Branch**: `001-minimal-llm` | **Date**: 2026-02-02 | **Spec**: ../spec.md
**Input**: Feature specification from `/specs/001-minimal-llm/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build the simplest on-device LLM that runs fully offline on 1GB devices using PyTorch. MVP delivers text completion with a 1k-token context window, bilingual (EN+ES) tokenizer, and local safety filtering. Implementation focuses on a tiny Transformer architecture with int8 quantized inference, strict memory budgets (runtime RSS ≤ 400MB; peak ≤ 512MB), and next-token p95 ≤ 250ms.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11; PyTorch 2.x (CPU-only target)  
**Primary Dependencies**: PyTorch, sentencepiece (tokenizer), onnx export (optional), numpy  
**Storage**: N/A (model bundle files, memory-mapped weights preferred)  
**Testing**: pytest; perf/energy/memory profiling scripts; safety test suite  
**Target Platform**: 1GB RAM devices, offline; CPU-only baseline; optional NPU/GPU when available  
**Project Type**: single (library + CLI)  
**Performance Goals**: next-token p95 ≤ 250ms; tokens/sec target documented (device-class matrix)  
**Constraints**: runtime RSS ≤ 400MB; peak memory ≤ 512MB; offline-first; safety local  
**Scale/Scope**: MVP: tiny Transformer (≈2 layers, d_model≈128, heads≈4), vocab≈8k; EN+ES only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The plan MUST demonstrate compliance with the SDDLLM constitution for 1GB systems:
- Privacy: All inference and personalization on-device; no raw data leaves device without explicit opt-in.
- Memory/Energy: On 1GB devices, target runtime RSS ≤ 400 MB, peak memory ≤ 512 MB; document and meet battery impact targets.
- Performance: Define device-class targets (tokens/sec, next-token p95 ≤ 250 ms) and measurement methodology.
- Testing: Include unit/integration tests for target platforms with reproducible perf/energy/memory benchmarks.
- Compatibility: Document semantic version impacts (models, tokenizer, APIs); include migration plan for breaks.
- Safety: On-device guardrails for unsafe content; versioned, testable policies.

Note: Final numeric targets may vary per device class (see constitution TODO(PERFORMANCE_TARGETS)); include your assumed test matrix here.

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
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
├── models/
│   ├── tiny_transformer.py
│   ├── tokenizer.py
│   └── quantization.py
├── services/
│   ├── generate.py
│   └── safety.py
├── cli/
│   └── minimal_llm.py
└── lib/
  └── runtime.py

tests/
├── contract/
│   └── test_generate_api.py
├── integration/
│   └── test_offline_generation.py
└── unit/
  ├── test_tokenizer.py
  ├── test_model_shapes.py
  └── test_safety.py

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: Single project with `src/` and `tests/` at repo root. Library exposes a CLI for offline generation. Contracts define local API semantics only (for possible future embedding).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
