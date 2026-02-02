<!--
Sync Impact Report
Version change: none → 1.0.0
Modified principles:
- [PRINCIPLE_1_NAME] → On‑Device Privacy First (NON‑NEGOTIABLE)
- [PRINCIPLE_2_NAME] → Resource Discipline for 1GB Systems
- [PRINCIPLE_3_NAME] → Test‑First with Benchmarks & Profiling
- [PRINCIPLE_4_NAME] → Backward Compatibility & Semantic Versioning
- [PRINCIPLE_5_NAME] → Safety & Responsible AI
Added sections:
- Additional Constraints & 1GB Standards
- Development Workflow & Quality Gates
Removed sections:
- None
Templates requiring updates:
- ⚠ .specify/templates/plan-template.md (update Constitution Check for 1GB gates)
- ⚠ .specify/templates/tasks-template.md (add perf/energy/memory/safety checks)
- ⚠ .specify/templates/spec-template.md (require mobile/1GB success metrics)
Follow-up TODOs:
- TODO(PERFORMANCE_TARGETS): finalize device‑class metrics, tokens/sec, and test devices
- TODO(RATIFICATION_APPROVALS): record approvers and ratification notes
-->

# SDDLLM Constitution

## Core Principles

### I. On‑Device Privacy First (NON‑NEGOTIABLE)
All inference, caching, and personalization MUST occur on‑device by default. No raw
user data (inputs, prompts, context windows, embeddings) may leave the device unless
explicitly opt‑in by the user with clear disclosures. Telemetry, if enabled, MUST be
aggregated, privacy‑preserving, and redact sensitive payloads. Storage MUST use OS‑level
encryption at rest; sensitive operations SHOULD leverage secure enclave/TEE when available.
Rationale: Trust, compliance, and offline usability require strict privacy guarantees.

### II. Resource Discipline for 1GB Systems
The model and runtime MUST operate within 1GB RAM systems and MUST define enforceable budgets:
- Quantization‑first (int4/int8) with hardware‑aware kernels; avoid FP32 paths.
- Total runtime RSS on a 1GB device during typical inference: ≤ 400 MB.
- Peak memory (including KV cache, activations): ≤ 512 MB.
- Model package size (installed): ≤ 500 MB; memory‑mapped weights preferred.
- Latency/throughput targets: next‑token p95 ≤ 250 ms; tokens/sec target MUST be documented per device class.
- Offline‑first: NO network dependency for core inference.
Rationale: 1GB systems have tight thermal and memory envelopes; budgets prevent instability.
Note: TODO(PERFORMANCE_TARGETS) to finalize device‑class numeric targets and test matrix.

### III. Test‑First with Benchmarks & Profiling
TDD is mandatory. Each feature MUST ship with unit tests, integration tests for target platforms,
and reproducible performance/energy/memory benchmarks on a 1GB device class. CI/CD MUST include
battery, memory, and latency profiles. Red‑Green‑Refactor cycles and regression suites MUST gate merges.
Rationale: Reliability and performance are first‑class features on constrained systems.

### IV. Backward Compatibility & Semantic Versioning
We use semantic versioning for models, tokenizer/runtime, and on‑device APIs.
- MINOR for additive capabilities or improvements without breaking existing contracts.
- MAJOR for any backward‑incompatible change (tokenizer vocabulary, file formats, on‑device APIs).
- PATCH for non‑semantic fixes and clarifications.
Any MAJOR change MUST include a migration plan and compatibility shims when feasible.
Rationale: Deployments on low‑resource systems are long‑lived; breakage is costly.

### V. Safety & Responsible AI
On‑device guardrails MUST prevent generation that violates safety policies. Safety classifiers,
prompt sanitization, and output filters MUST run locally. The policy set MUST be testable and versioned,
and MUST be auditable offline. Rationale: User safety and platform compliance depend on robust safeguards.

## Additional Constraints & 1GB Standards

- Target hardware: 1GB RAM class devices; assume CPU‑only; optional GPU/NPU acceleration when available.
- Packaging: Signed model bundles with integrity checks; incremental updates supported.
- Quantization standards: Prefer int4/int8 with per‑channel scaling; verify quality via test suites.
- Data retention: On‑device only; explicit opt‑in needed for any remote sync.
- Observability: Structured logs limited to non‑PII; developer mode exposes text I/O diagnostics.

## Development Workflow & Quality Gates

- Code review MUST verify privacy, performance, safety, compatibility, and 1GB memory budgets.
- Release gates MUST pass: unit/integration tests, latency/throughput/battery profiles, memory ceiling.
- Device lab matrix MUST include at least one 1GB RAM device.
- Rollout: staged updates with rollback support; model/hash pinned in app config.
- Documentation: each release MUST include metrics, device list, and migration notes.

## Governance

This constitution supersedes other practices for 1GB‑target LLM development. Amendments require a documented
proposal, impact assessment (privacy/perf/safety/compatibility), and approval by maintainers. Versioning policy:
- MAJOR: Backward‑incompatible governance or principle redefinitions.
- MINOR: New principle/section added or materially expanded guidance.
- PATCH: Clarifications, wording, and non‑semantic refinements.
Compliance reviews MUST be performed per release and quarterly across active branches.

**Version**: 1.0.0 | **Ratified**: 2026-02-02 | **Last Amended**: 2026-02-02
