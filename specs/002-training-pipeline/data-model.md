# Data Model: Training Pipeline Automation & Documentation

**Feature**: 002-training-pipeline  
**Date**: 2026-02-04  
**Purpose**: Define data structures and entities for training pipeline orchestration

## Overview

This feature involves three primary data domains:
1. **Pipeline Configuration** - Parameters controlling the training workflow
2. **Pipeline State** - Runtime execution tracking and progress
3. **Training Documentation** - Structured educational content

All data structures are implementation-agnostic (no Python/bash specifics), focusing on conceptual relationships and constraints.

---

## Entity Definitions

### 1. PipelineConfiguration

**Purpose**: Encapsulates all parameters needed to execute the complete training workflow from corpus download through model validation.

**Attributes**:
- `corpus_source` (String): URL or file path to training text corpus
  - Constraint: Must be accessible (HTTP 200 or file exists)
  - Default: Project Gutenberg bilingual collection
- `corpus_size_mb` (Integer): Expected corpus size in MB for disk space validation
  - Constraint: > 0, typically 1-100 MB
- `tokenizer_vocab_size` (Integer): Number of tokens in vocabulary
  - Constraint: Power of 2 recommended, range [1000, 32000]
  - Default: 8000
- `tokenizer_model_type` (Enum): Algorithm for tokenization
  - Values: unigram | bpe | char | word
  - Default: unigram
- `model_architecture` (Object): Neural network structure parameters
  - `num_layers` (Integer): Transformer layers (1-12)
  - `d_model` (Integer): Embedding dimension (64-512)
  - `num_heads` (Integer): Attention heads (1-16)
  - `dim_feedforward` (Integer): FFN hidden size (128-2048)
  - Default: {num_layers: 2, d_model: 128, num_heads: 4, dim_feedforward: 256}
- `training_hyperparameters` (Object): Optimization settings
  - `batch_size` (Integer): Samples per training step (1-128)
  - `learning_rate` (Float): Initial LR (1e-5 to 1e-2)
  - `num_epochs` (Integer): Complete passes through dataset (1-100)
  - `max_sequence_length` (Integer): Token context window (32-2048)
  - Default: {batch_size: 32, learning_rate: 0.0003, num_epochs: 20, max_sequence_length: 128}
- `output_directory` (String): Where to save artifacts
  - Constraint: Must be writable, have sufficient space
  - Default: ./data/
- `resume_from_checkpoint` (Boolean): Whether to continue from existing checkpoint
  - Default: false (prompt user interactively if checkpoints found)
- `skip_phases` (Set[String]): Phases to skip if artifacts already exist
  - Values: {corpus_download, tokenizer_training}
  - Default: empty set

**Relationships**:
- Produces → PipelineStatus (after execution begins)
- Validates against → PrerequisiteChecks (before execution)

**Validation Rules**:
- `batch_size * max_sequence_length` should not exceed memory budget
- `num_heads` must divide evenly into `d_model`
- `learning_rate` typically decreases with larger `batch_size`

---

### 2. PipelineStatus

**Purpose**: Tracks execution state of the training pipeline, enabling progress monitoring, error recovery, and result validation.

**Attributes**:
- `pipeline_id` (UUID): Unique identifier for this training run
- `start_time` (Timestamp): When pipeline execution began
- `current_phase` (Enum): Active stage of workflow
  - Values: prerequisite_check | corpus_download | tokenizer_training | model_training | validation | complete | failed
  - Transitions: Must follow sequential order (no skipping unless skip_phases configured)
- `completed_phases` (List[String]): Ordered list of finished phases
- `phase_timings` (Map[String, Duration]): Time spent in each phase (for performance analysis)
- `artifacts_created` (List[FilePath]): Generated files tracked for cleanup/validation
  - Examples: [corpus_bilingual.txt, bilingual_8k.model, best_model.pt]
- `error_details` (Object | null): If failed, diagnostic information
  - `failed_phase` (String): Which phase encountered error
  - `error_type` (String): Category (disk_space | network | oom | validation_failure)
  - `error_message` (String): Human-readable explanation
  - `stderr_output` (String): Last 50 lines of error output
- `validation_results` (Object | null): Post-training quality checks
  - `sample_generations` (List[String]): Example outputs from trained model
  - `final_train_loss` (Float): Training set loss at completion
  - `final_val_loss` (Float): Validation set loss at completion
  - `inference_metrics` (Object):
    - `latency_p95_ms` (Integer): 95th percentile next-token time
    - `tokens_per_sec` (Float): Generation throughput
    - `rss_mb` (Integer): Resident set size during inference
    - `peak_mb` (Integer): Peak memory usage
  - `constitutional_compliance` (Boolean): Whether inference metrics meet budgets

**Relationships**:
- Created by → PipelineConfiguration (initiated with)
- References → TrainingMetrics (detailed training history)

**State Transitions**:
```
prerequisite_check → [PASS] → corpus_download | [FAIL] → failed
corpus_download → [SUCCESS] → tokenizer_training | [ERROR] → failed
tokenizer_training → [SUCCESS] → model_training | [ERROR] → failed
model_training → [SUCCESS] → validation | [ERROR] → failed
validation → [PASS] → complete | [FAIL] → failed
```

---

### 3. TrainingMetrics

**Purpose**: Detailed per-epoch statistics collected during model training for analysis, comparison, and troubleshooting.

**Attributes**:
- `run_id` (UUID): Links to PipelineStatus.pipeline_id
- `epoch_history` (List[EpochMetrics]): Chronological training progress
  - Each `EpochMetrics` contains:
    - `epoch_number` (Integer): 1-indexed epoch count
    - `train_loss` (Float): Average loss on training set
    - `validation_loss` (Float): Average loss on held-out validation set
    - `learning_rate` (Float): Optimizer LR at this epoch
    - `elapsed_seconds` (Integer): Time to complete epoch
    - `sample_outputs` (List[String]): Generated text examples for quality assessment
    - `gradient_norm` (Float): For monitoring training stability
- `final_metrics` (Object): Aggregated results
  - `total_training_time` (Duration): Sum of all epochs
  - `best_epoch` (Integer): Epoch with lowest validation loss
  - `convergence_achieved` (Boolean): Whether loss plateaued (variance < threshold)
  - `overfitting_detected` (Boolean): Whether val_loss >> train_loss (gap > 0.5)

**Relationships**:
- Belongs to → PipelineStatus (one-to-one)
- Used by → TrainingCheckpoint (for resume capability)

**Persistence**:
- Saved to: `{output_directory}/training_metrics.json`
- Format: JSON for easy parsing, plotting, CI/CD integration

---

### 4. TrainingCheckpoint

**Purpose**: Snapshot of model and optimizer state enabling resume from interruptions.

**Attributes**:
- `epoch` (Integer): Checkpoint taken after this epoch
- `model_state` (Tensor Dict): Learned weights and biases
  - Keys: Layer names (e.g., "embedding.weight", "attention.query.weight")
  - Values: Multi-dimensional numerical arrays
- `optimizer_state` (Dict): AdamW state for momentum/variance
- `training_config` (PipelineConfiguration): Hyperparameters used (for validation on resume)
- `loss_history` (Object):
  - `train_losses` (List[Float]): Per-epoch training loss
  - `val_losses` (List[Float]): Per-epoch validation loss
- `checkpoint_path` (FilePath): Where this checkpoint is saved
  - Format: `checkpoint_epoch_{epoch}.pt`

**Relationships**:
- Created during → PipelineStatus (when current_phase = model_training)
- References → PipelineConfiguration (to validate compatibility on resume)

**Validation on Resume**:
- Vocabulary size must match (else incompatible tokenizer)
- Model architecture must match (else weight shapes wrong)
- Training config can differ (allows hyperparameter adjustments between runs)

---

### 5. TrainingDocumentation

**Purpose**: Structured educational content explaining the training process from theory through troubleshooting.

**Attributes**:
- `sections` (List[DocumentSection]): Ordered instructional content
  - Each `DocumentSection`:
    - `section_id` (String): Unique identifier (e.g., "tokenization", "training_loop")
    - `title` (String): Human-readable heading
    - `content` (Markdown): Explanatory text with examples
    - `diagrams` (List[ASCIIDiagram]): Visual representations
    - `code_examples` (List[CodeSnippet]): Runnable demonstrations
    - `prerequisites` (List[String]): Concepts assumed known
- `cross_references` (Map[String, String]): Links between related sections
- `glossary` (Map[String, String]): Term definitions for newcomers
- `troubleshooting_guide` (List[TroubleshootingEntry]):
  - Each entry:
    - `symptom` (String): Observed problem (e.g., "loss not decreasing")
    - `possible_causes` (List[String]): What might be wrong
    - `solutions` (List[String]): Actions to take
    - `related_sections` (List[String]): Where to read more

**Relationships**:
- References → PipelineConfiguration (explains parameter choices)
- References → TrainingMetrics (uses real examples from training runs)
- Supports → PipelineStatus.error_details (troubleshooting guide indexed by error types)

**Quality Criteria**:
- Length: 1500-2500 lines of Markdown
- Readability: Accessible to developers with basic ML knowledge
- Completeness: Covers all phases from corpus selection to deployment
- Practicality: Includes runnable code examples without modification

---

## Data Flow

### Training Pipeline Execution

```
[User Input]
    |
    v
[PipelineConfiguration] ──── validates ───> [PrerequisiteChecks]
    |                                             |
    | creates                                    PASS/FAIL
    v                                             |
[PipelineStatus: prerequisite_check] <───────────┘
    |
    | phase transition
    v
[PipelineStatus: corpus_download]
    |
    | creates artifact
    v
[corpus_bilingual.txt]
    |
    | phase transition
    v
[PipelineStatus: tokenizer_training]
    |
    | creates artifacts
    v
[bilingual_8k.model, bilingual_8k.vocab]
    |
    | phase transition
    v
[PipelineStatus: model_training]
    |
    | generates (per epoch)
    v
[TrainingMetrics.epoch_history] & [TrainingCheckpoint]
    |
    | creates artifacts
    v
[best_model.pt, final_model.pt]
    |
    | phase transition
    v
[PipelineStatus: validation]
    |
    | tests against
    v
[Constitutional Budgets] ──> [validation_results]
    |
    | phase transition
    v
[PipelineStatus: complete]
```

### Documentation Usage

```
[Developer (new to LLM training)]
    |
    | reads
    v
[TrainingDocumentation: Overview Section]
    |
    | follows learning path
    v
[TrainingDocumentation: Tokenization → Model Init → Training Loop]
    |
    | encounters issue
    v
[TrainingDocumentation: Troubleshooting Guide]
    |
    | cross-references
    v
[TrainingDocumentation: Hyperparameter Reference]
    |
    | applies solution
    v
[Modified PipelineConfiguration]
    |
    | re-runs
    v
[Successful Training]
```

---

## Constraints & Invariants

### Disk Space
- Required: `corpus_size_mb + (model_params * 4 bytes * 3 checkpoints) + 100MB buffer`
- For TinyTransformer (2.45M params): `3 + (10 * 3) + 100 = ~133 MB minimum`

### Memory (Training)
- Development machine: No hard limit (not 1GB device)
- Trained artifacts must fit 1GB device: `RSS ≤ 400MB, peak ≤ 512MB` during inference

### Training Time
- Target: < 10 minutes for 20 epochs on 3MB corpus (CPU-only)
- Scaling: Approximately linear with corpus size and epochs

### Model Quality
- Minimum: `validation_loss < 5.0` after 20 epochs
- Target: Coherent text generation (no token repetition, grammatically plausible)
- Bilingual capability: Generates both English and Spanish

### Documentation Completeness
- All 10 sections must cover their designated topics
- Minimum 10 diagrams illustrating data transformations
- Minimum 10 code examples (all runnable without edits)
- Troubleshooting guide addresses 80% of common issues

---

## Versioning & Compatibility

### Model Checkpoints
- **MAJOR** version change: Vocabulary size change (incompatible tokenizer)
- **MINOR** version change: Architecture change (num_layers, d_model)
- **PATCH** version change: Hyperparameter tuning (same architecture, new weights)

### Configuration Evolution
- New optional parameters: Backward compatible (use defaults)
- Renamed parameters: Breaking change (requires migration script)
- Removed parameters: Breaking change (document in changelog)

---

## Testing Validation

### Data Integrity
- All artifacts have non-zero size
- Checkpoint files loadable without corruption
- JSON metrics parseable and valid

### Functional Correctness
- Pipeline completes all phases in order
- Trained model generates text (not random noise)
- Inference metrics within constitutional budgets

### Error Handling
- Disk full → Fails at prerequisite check (before downloading)
- Network timeout → Retries corpus download 3 times
- OOM during training → Suggests reducing batch_size in error message

This data model provides the foundation for implementation in Phase 2. All entities map cleanly to bash variables (PipelineConfiguration), JSON files (TrainingMetrics, PipelineStatus), PyTorch checkpoint files (TrainingCheckpoint), and Markdown documents (TrainingDocumentation).
