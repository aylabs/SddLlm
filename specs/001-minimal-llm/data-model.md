# Data Model: Minimal On-Device LLM

## Entities

### ModelBundle
- id: string
- version: semver
- size_bytes: int
- hash_sha256: string
- path: string
- quantization: {type: "int8"}
- tokenizer_version: semver
- integrity_verified: boolean

### TokenizerVocabulary
- vocab_size: int (≈8000)
- specials: ["<bos>", "<eos>", "<pad>"]
- language_scope: ["en", "es"]

### RuntimeConfig
- max_context_tokens: int (1000)
- max_output_tokens: int (e.g., 128)
- temperature: float
- top_p: float
- budgets: {rss_mb_max: 400, peak_mb_max: 512, p95_ms_max: 250}

### PromptSession
- prompt_text: string
- generated_text: string
- metrics: {latency_p95_ms: int, tokens_per_sec: float, rss_mb: int, peak_mb: int}
- safety_outcome: {status: "allow"|"refuse"|"sanitize", category?: string}
- timestamp: datetime

### SafetyPolicy
- categories: ["violence", "sexual", "hate", "self-harm", "illegal"]
- version: semver
- rules: array (human-readable)

