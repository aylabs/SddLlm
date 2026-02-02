# Feature Specification: Minimal On-Device LLM

**Feature Branch**: `001-minimal-llm`  
**Created**: 2026-02-02  
**Status**: Draft  
**Input**: User description: "create the simplest LLM"

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - On-device text completion (Priority: P1)

User opens the app, enters a short prompt (e.g., a question or instruction), and receives a coherent text completion fully offline, within the defined latency and memory budgets for 1GB devices.

**Why this priority**: This is the core value of the minimal LLM: reliable, fast, offline text generation on constrained hardware.

**Independent Test**: Generate a 50-token response offline on a 1GB device; measure next-token p95 latency and memory usage. The feature passes if SC-005..SC-008 are met.

**Acceptance Scenarios**:

1. **Given** a 1GB device and the minimal model installed, **When** the user inputs a simple prompt, **Then** the app produces a coherent completion offline while meeting memory and latency targets.
2. **Given** a 1GB device under low battery mode, **When** the user requests a completion, **Then** the app completes within energy constraints or provides a graceful notification to reduce generation length.

---

### User Story 2 - Model bundle initialization & management (Priority: P2)

User installs or updates the minimal model bundle. On app launch or first use, the model initializes quickly and stays within the 1GB resource budgets.

**Why this priority**: Fast, predictable initialization and robust packaging are critical for user trust on constrained systems.

**Independent Test**: Cold start initialization on a 1GB device; verify bundle integrity, version compatibility, and memory budgets.

**Acceptance Scenarios**:

1. **Given** a fresh install, **When** the app initializes the model, **Then** cold start time and memory budgets are within Success Criteria and the bundle integrity check passes.
2. **Given** an updated bundle with a MINOR version change, **When** the app starts, **Then** initialization succeeds without migration steps; with a MAJOR change, **Then** migration notes are presented and compatibility shims are applied.

---

### User Story 3 - On-device safety filtering (Priority: P3)

User submits a prompt that may violate safety policies. The app filters or refuses generation locally, returning a safe, helpful alternative or refusal message.

**Why this priority**: Safety and platform compliance are essential, even for minimal functionality.

**Independent Test**: Submit a suite of unsafe prompts; verify local policy enforcement and auditable outcomes.

**Acceptance Scenarios**:

1. **Given** a policy-defined unsafe prompt, **When** the user submits it, **Then** the app refuses or provides a safe alternative without contacting any remote service.
2. **Given** borderline prompts, **When** tested, **Then** outcomes are consistent with the defined safety policy and documented rationale.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- Extremely long prompts exceeding the context window → app truncates or guides the user to shorten input.
- Rapid successive prompts causing potential thermal throttling → app rate-limits or reduces generation length.
- Low battery state → app reduces generation length or warns before proceeding.
- Storage nearly full → app prevents bundle install/update and informs user.
- Corrupted or tampered model bundle → app fails integrity checks and refuses load.
- Out-of-memory during generation → app aborts safely and suggests shorter output or lower settings.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST generate coherent text completions fully offline on 1GB devices, meeting defined latency and memory budgets.
- **FR-002**: System MUST initialize the model bundle quickly and verify integrity and semantic version compatibility before use.
- **FR-003**: System MUST enforce resource budgets: runtime RSS ≤ 400 MB and peak memory ≤ 512 MB on 1GB devices during typical inference.
- **FR-004**: System MUST enforce on-device safety policies for unsafe prompts and produce safe refusals or alternatives.
- **FR-005**: System MUST provide clear, user-friendly feedback for error conditions (e.g., OOM, corrupted bundle, insufficient storage).

*Critical clarifications impacting scope and testability:*

- **FR-006**: Context window MUST be limited to 1k tokens to meet memory budgets on 1GB devices.
- **FR-007**: Supported language set MUST be bilingual (English + Spanish) to define test coverage and acceptance.
- **FR-008**: Safety categories MUST follow a custom minimal set (e.g., violence, sexual content, hate content, self-harm, illegal activity) for consistent on-device enforcement.

### Key Entities *(include if feature involves data)*

- **Model Bundle**: Versioned package containing minimal weights, tokenizer, and metadata (id, version, size, hash, integrity).
- **Tokenizer Vocabulary**: Versioned mapping with vocabulary size and special tokens; impacts memory and compatibility.
- **Prompt Session**: Input prompt, generated tokens, latency/throughput metrics, and safety decision outcome.
- **Runtime Config**: Adjustable generation parameters (e.g., max tokens, temperature) and resource budgets; persisted locally.
- **Safety Policy**: Versioned rule set defining unsafe categories and handling; auditable offline.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Users receive a coherent text completion for a short prompt fully offline within an MVP response time acceptable to end users (qualitative satisfaction ≥ 80%).
- **SC-002**: Cold start initialization completes within a user-acceptable timeframe and without violating memory budgets.
- **SC-003**: 90% of users successfully complete the primary generation task on first attempt without errors.
- **SC-004**: Support requests related to initialization failures or OOM conditions reduced after release by ≥ 50%.

### Mobile/1GB Runtime Metrics *(if applicable)*

- **SC-005**: Runtime RSS ≤ 400 MB and peak memory ≤ 512 MB on 1GB devices
- **SC-006**: Next-token p95 latency ≤ 250 ms; tokens/sec target documented and met
- **SC-007**: Battery impact ≤ 5% drain per 30 minutes of continuous inference
- **SC-008**: All core inference functions operate offline with no network dependency
