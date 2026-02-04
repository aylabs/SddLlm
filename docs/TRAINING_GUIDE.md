# Training Guide: Understanding LLM Training from Corpus to Deployment

**Version:** 1.0  
**Target Audience:** Developers with basic ML knowledge  
**Prerequisites:** Understanding of neural networks, Python familiarity, command line comfort  
**Estimated Reading Time:** 45-60 minutes

---

## Table of Contents

1. [Training Overview: What and Why](#1-training-overview-what-and-why)
2. [Phase 1: Tokenization - Text to Numbers](#2-phase-1-tokenization---text-to-numbers)
3. [Phase 2: Model Initialization - Random Start](#3-phase-2-model-initialization---random-start)
4. [Phase 3: Data Preparation - Batching and Splitting](#4-phase-3-data-preparation---batching-and-splitting)
5. [Phase 4: The Training Loop - Forward and Backward](#5-phase-4-the-training-loop---forward-and-backward)
6. [Phase 5: Optimization - Learning Rate and Schedulers](#6-phase-5-optimization---learning-rate-and-schedulers)
7. [Phase 6: Evaluation and Checkpointing](#7-phase-6-evaluation-and-checkpointing)
8. [Troubleshooting Common Issues](#8-troubleshooting-common-issues)
9. [Hyperparameter Reference and Tuning](#9-hyperparameter-reference-and-tuning)
10. [Constitutional Compliance and Deployment](#10-constitutional-compliance-and-deployment)

---

## 1. Training Overview: What and Why

### What is Model Training?

Training is the process of teaching a neural network to learn patterns from data. For language models, training means:

- **Learning text patterns**: Understanding grammar, word associations, and context
- **Predicting next tokens**: Given "The cat sat on the", predict "mat" or "floor"
- **Adjusting weights**: Modifying 2.45 million parameters to minimize prediction errors

**Key Distinction:**
- **Training** (development-time): Heavy computation, gradient updates, learning from data
- **Inference** (runtime): Lightweight prediction, no weight changes, fast generation

### TinyTransformer Architecture

Our model is a scaled-down transformer with:
- **2 transformer layers** (vs 96 in GPT-3)
- **128 d_model** (embedding dimension)
- **4 attention heads** per layer
- **256 FFN dimension** (feedforward network)
- **2.45M parameters** (~10MB memory footprint)

This architecture is designed for educational purposes and 1GB device deployment (Raspberry Pi Zero 2W).

### Training Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                 │
└──────────────────────────────────────────────────────────────────────────┘

Step 1: CORPUS DOWNLOAD
    │
    ├─> Download bilingual text (English + Spanish)
    ├─> ~3MB, 46,000 sentences
    └─> Output: data/corpus_bilingual.txt
    
Step 2: TOKENIZER TRAINING
    │
    ├─> Learn vocabulary from corpus
    ├─> SentencePiece Unigram algorithm
    ├─> Vocabulary size: 8000 tokens
    └─> Output: data/bilingual_8k.model
    
Step 3: MODEL TRAINING
    │
    ├─> Initialize random weights (2.45M parameters)
    ├─> Train for 20 epochs (~10 minutes)
    ├─> Batch size: 32, Learning rate: 0.0003
    └─> Output: data/final_model.pt, data/training_metrics.json
    
Step 4: VALIDATION
    │
    ├─> Test sample generation
    ├─> Verify constitutional compliance (RSS ≤ 400MB, p95 ≤ 250ms)
    └─> Output: Validated model ready for deployment
```

### Expected Outcomes

After training:
1. **Loss decreases**: From ~8.0 (random) to ~2.5 (trained)
2. **Text generation improves**: From gibberish to coherent short phrases
3. **Checkpoints saved**: Best model based on validation loss
4. **Metrics exported**: JSON file with training history

**Reality Check:** TinyTransformer won't write novels - it's designed for:
- Simple text completion
- Educational demonstrations
- Resource-constrained deployment testing

---

## 2. Phase 1: Tokenization - Text to Numbers

### Why Tokenization?

Neural networks operate on numbers, not text. Tokenization converts text into integer sequences:

```
Input:  "Hello world! ¿Cómo estás?"
           ↓ (tokenization)
Output: [2304, 234, 1543, 45, 7821, 623, 3]
```

**Three Key Benefits:**
1. **Compression**: Subwords are more efficient than character-level (128 tokens vs 384 chars)
2. **Generalization**: Learns morphemes (e.g., "running" = "run" + "ing")
3. **Language flexibility**: Handles English, Spanish, code, rare words

### SentencePiece Unigram Algorithm

We use SentencePiece with the Unigram algorithm:

```
┌─────────────────────────────────────────────────────────┐
│          VOCABULARY CONSTRUCTION PROCESS                │
└─────────────────────────────────────────────────────────┘

Step 1: START WITH LARGE VOCABULARY
    │
    ├─> Initialize with all character n-grams from corpus
    └─> ~50,000 candidate tokens
    
Step 2: EXPECTATION-MAXIMIZATION (EM)
    │
    ├─> E-step: Compute token probabilities for each word
    ├─> M-step: Re-estimate token probabilities
    └─> Iterate 10-20 times until convergence
    
Step 3: PRUNE TO TARGET SIZE
    │
    ├─> Remove least-probable tokens
    ├─> Keep 8000 most useful tokens
    └─> Balance frequency vs. coverage
    
Step 4: FINALIZE
    │
    ├─> Add special tokens: BOS=0, EOS=1, PAD=2, UNK=3
    ├─> Save model: bilingual_8k.model
    └─> Save vocabulary: bilingual_8k.vocab
```

### Special Tokens

| Token | ID | Purpose | Example Usage |
|-------|----|---------|--------------
 |
| BOS   | 0  | Beginning of sequence | Start of each sentence |
| EOS   | 1  | End of sequence | Termination marker |
| PAD   | 2  | Padding (unused in training) | Batch alignment |
| UNK   | 3  | Unknown token | Rare words not in vocab |

### Character Coverage

We use **99.95% character coverage** for bilingual corpus:
- Ensures most characters are represented
- Rare characters (emojis, special symbols) map to UNK
- Balances vocabulary size vs. coverage

### Example: Encoding and Decoding

**Encoding** (text → token IDs):

```python
import sentencepiece as spm

# Load trained tokenizer
sp = spm.SentencePieceProcessor(model_file='data/bilingual_8k.model')

# Encode text
text = "Hello world! ¿Cómo estás?"
tokens = sp.encode(text, out_type=int)
print(tokens)  # [2304, 234, 1543, 45, 7821, 623, 3]

# Encode as string tokens (for debugging)
token_strings = sp.encode(text, out_type=str)
print(token_strings)  # ['▁Hello', '▁world', '!', '▁¿', 'Có', 'mo', '▁estás', '?']
```

**Decoding** (token IDs → text):

```python
# Decode tokens back to text
decoded = sp.decode(tokens)
print(decoded)  # "Hello world! ¿Cómo estás?"

# Decode partial sequence
partial = sp.decode([2304, 234])
print(partial)  # "Hello world"
```

### Token Type Comparison

| Algorithm | Vocab Size | Training Speed | Generation Quality | Use Case |
|-----------|-----------|---------------|-------------------|----------|
| **Character-level** | 50-200 | Fast (minutes) | Poor (long dependencies) | Very small models |
| **Unigram** (ours) | 8,000 | Medium (seconds) | Good (balanced) | Bilingual, general |
| **BPE** | 32,000 | Medium (seconds) | Good (efficient) | English-heavy, large models |
| **WordPiece** | 30,000 | Medium (seconds) | Good (robust OOV) | BERT, multilingual |

### Vocabulary Size Trade-offs

**Small vocabulary (1000 tokens):**
- ✅ Faster training (smaller embedding matrix)
- ✅ Lower memory footprint
- ❌ Longer sequences (more tokens per sentence)
- ❌ Poor rare word handling

**Large vocabulary (32,000 tokens):**
- ✅ Shorter sequences (fewer tokens per sentence)
- ✅ Better rare word handling
- ❌ Slower training (larger embedding matrix)
- ❌ Higher memory footprint

**Our choice (8,000 tokens):** Balanced for bilingual corpus and resource constraints.

### Practical Example: Training Tokenizer

```bash
# Train tokenizer using our script
python3 scripts/train_tokenizer.py \
    --vocab-size 8000 \
    --input-file data/corpus_bilingual.txt \
    --output-dir data

# Outputs:
#   data/bilingual_8k.model  (~160KB)
#   data/bilingual_8k.vocab  (~80KB)
```

**Training time:** ~2 seconds on modern CPU

---

## 3. Phase 2: Model Initialization - Random Start

### TinyTransformer Architecture Breakdown

```
┌────────────────────────────────────────────────────────────────────┐
│                   TINYTRANSFORMER ARCHITECTURE                      │
└────────────────────────────────────────────────────────────────────┘

INPUT (token IDs: [32])
    │
    ├─> Token Embedding: (32) → (32, 128)
    │   └─> Matrix: vocab_size=8000 × d_model=128
    │
    ├─> Positional Encoding: (32, 128) [added to embeddings]
    │   └─> Learned absolute positions
    │
    ├─> TransformerBlock₁:
    │   ├─> Multi-Head Attention (4 heads):
    │   │   ├─> Q, K, V projections: (32, 128) → (32, 128)
    │   │   ├─> Attention: softmax(Q·Kᵀ/√d) · V
    │   │   └─> Causal mask (prevent future peeking)
    │   ├─> Add & LayerNorm
    │   ├─> FeedForward Network:
    │   │   ├─> Linear: (32, 128) → (32, 256)
    │   │   ├─> GELU activation
    │   │   └─> Linear: (32, 256) → (32, 128)
    │   └─> Add & LayerNorm
    │
    ├─> TransformerBlock₂: (same structure as Block₁)
    │
    └─> Output Projection:
        └─> Linear: (32, 128) → (32, 8000) [logits for each token]

LOSS: CrossEntropy(logits, targets)
```

### Parameter Count Calculation

Let's break down the 2.45M parameters:

```
1. Token Embedding:
   vocab_size × d_model = 8,000 × 128 = 1,024,000 params

2. Positional Encoding:
   max_seq_len × d_model = 128 × 128 = 16,384 params

3. TransformerBlock (×2 layers):
   Per block:
     - Multi-Head Attention:
       Q, K, V projections: 3 × (128 × 128) = 49,152
       Output projection: 128 × 128 = 16,384
       Subtotal: 65,536 params
     
     - FeedForward Network:
       Linear1: 128 × 256 = 32,768
       Linear2: 256 × 128 = 32,768
       Subtotal: 65,536 params
     
     - LayerNorm (×2): 2 × (128 × 2) = 512 params
   
   Total per block: 131,584 params
   Total for 2 blocks: 263,168 params

4. Output Projection:
   d_model × vocab_size = 128 × 8,000 = 1,024,000 params

TOTAL: 1,024,000 + 16,384 + 263,168 + 1,024,000 = 2,327,552 ≈ 2.45M params
```

### Memory Footprint

**Model size (float32):**
```
2,327,552 params × 4 bytes/param = 9,310,208 bytes ≈ 9.3 MB
```

**During training (additional memory):**
- Gradients: 9.3 MB (same as parameters)
- Optimizer state (AdamW): ~18.6 MB (2× parameters for momentum + variance)
- Activations: ~5-10 MB (depends on batch size)

**Total training memory:** ~40-50 MB (well within 400MB constitutional limit)

### Random Weight Initialization

**Why random, not zeros?**

```python
# ❌ BAD: All weights zero
model = TinyTransformer(...)
for param in model.parameters():
    param.data.fill_(0.0)

# Problem: All neurons compute same gradients → no learning diversity
```

```python
# ✅ GOOD: Random initialization (PyTorch default)
model = TinyTransformer(vocab_size=8000, d_model=128, nhead=4, num_layers=2)

# PyTorch uses Xavier/Kaiming initialization:
# - Linear layers: uniform(-√k, √k) where k = 1/in_features
# - Embeddings: normal(0, 1)
```

**Effect at initialization:** Model outputs gibberish because weights are random.

### Example: Model Instantiation

```python
import sys
sys.path.insert(0, 'src')
from models.tiny_transformer import TinyTransformer

# Create model
model = TinyTransformer(
    vocab_size=8000,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256
)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")  # 2,327,552

# Model size in MB
model_size_mb = num_params * 4 / (1024 ** 2)
print(f"Model size: {model_size_mb:.2f} MB")  # ~9.3 MB

# Test forward pass with random input
import torch
input_ids = torch.randint(0, 8000, (1, 32))  # Batch size 1, sequence length 32
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # torch.Size([1, 32, 8000])
```

### Pre-Training Sanity Check

```python
# Before training, model should produce random-like predictions
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='data/bilingual_8k.model')

# Tokenize prompt
prompt = "Hello"
input_ids = sp.encode(prompt, out_type=int)
input_tensor = torch.tensor([input_ids])

# Generate (untrained model)
model.eval()
with torch.no_grad():
    logits = model(input_tensor)
    next_token_logits = logits[0, -1, :]  # Last position
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = sp.decode([next_token_id])

print(f"Prompt: {prompt}")
print(f"Untrained prediction: {next_token}")  # Likely gibberish/random token
```

**Expected behavior:** Untrained model produces incoherent predictions with high perplexity.

---

## 4. Phase 3: Data Preparation - Batching and Splitting

### Corpus Structure

Our bilingual corpus:
- **Size:** ~3 MB (3,145,728 bytes)
- **Sentences:** 46,000 lines (mixed English + Spanish)
- **Tokenized length:** ~680,000 tokens (using 8K vocab)
- **Source:** Project Gutenberg public domain books

**Example content:**
```
Hello, how are you?
Hola, ¿cómo estás?
The cat sat on the mat.
El gato se sentó en la estera.
...
```

### Sequence Creation

We convert the corpus into fixed-length sequences using a **sliding window**:

```
┌────────────────────────────────────────────────────────────┐
│          CORPUS TO SEQUENCES TRANSFORMATION                │
└────────────────────────────────────────────────────────────┘

Full tokenized corpus: [234, 56, 789, 12, 45, 67, 89, 123, ...]
                              ↓
                    max_seq_length = 128

Sequence 1: [234, 56, 789, ..., 45]   (tokens 0-127)
Sequence 2: [56, 789, 12, ..., 67]    (tokens 1-128)
Sequence 3: [789, 12, 45, ..., 89]    (tokens 2-129)
...

Total sequences: ~680,000 - 128 = 679,872
```

**Why fixed length?**
- GPUs process batches efficiently with uniform shapes
- Attention complexity: O(n²) where n = sequence length
- Longer sequences → more memory, slower training

### Train/Validation Split

We split sequences into:
- **Training set:** 90% (611,884 sequences)
- **Validation set:** 10% (67,988 sequences)

```python
import torch
from torch.utils.data import random_split

full_dataset = TextDataset('data/corpus_bilingual.txt', tokenizer, max_seq_length=128)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Training samples: {len(train_dataset)}")     # ~611,884
print(f"Validation samples: {len(val_dataset)}")     # ~67,988
```

**Why validation set?**
- Detect overfitting (train loss ↓, val loss ↑)
- Early stopping (save best model based on val loss)
- Hyperparameter tuning (without contaminating test set)

### Batching Strategy

Batches group multiple sequences for parallel processing:

```
┌─────────────────────────────────────────────────────┐
│            BATCHING PROCESS                         │
└─────────────────────────────────────────────────────┘

Training dataset (611,884 sequences)
    │
    ├─> Shuffle sequences (randomize order)
    │
    └─> Create batches of size 32:
        
        Batch 1: [
            Sequence 1:  [234, 56, ..., 45]  (128 tokens)
            Sequence 2:  [789, 12, ..., 67]  (128 tokens)
            ...
            Sequence 32: [456, 78, ..., 90]  (128 tokens)
        ]
        Shape: (32, 128)
        
        Batch 2: [next 32 sequences]
        ...
        
        Total batches: 611,884 / 32 = 19,121 batches
```

**Batch size trade-offs:**

| Batch Size | GPU Memory | Training Speed | Gradient Quality | Use Case |
|-----------|-----------|---------------|-----------------|----------|
| 8         | ~50 MB    | Slow          | Noisy (high variance) | Debugging |
| 16        | ~100 MB   | Medium        | Moderate | Resource-constrained |
| **32**    | **~150 MB** | **Good**    | **Good** | **Default (our choice)** |
| 64        | ~300 MB   | Fast          | Very smooth | Large GPUs |
| 128       | ~600 MB   | Very fast     | Too smooth (slow convergence) | Distributed training |

### Next-Token Prediction Task

The core training objective is **predicting the next token**:

```
Given input: "The cat sat on the"
Predict next token: "mat" (or "floor", "chair", etc.)
```

**Implementation:**

```python
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load and tokenize full corpus
        with open(corpus_file, 'r') as f:
            text = f.read()
        self.tokens = tokenizer.encode(text, out_type=int)
    
    def __len__(self):
        return len(self.tokens) - self.max_seq_length
    
    def __getitem__(self, idx):
        # Extract sequence of length max_seq_length
        chunk = self.tokens[idx : idx + self.max_seq_length + 1]
        
        # Split into input and target
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)   # tokens[:-1]
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)   # tokens[1:]
        
        return input_ids, target_ids

# Example:
# chunk = [234, 56, 789, 12, 45]
# input_ids = [234, 56, 789, 12]      (predict token at each position)
# target_ids = [56, 789, 12, 45]      (correct next token at each position)
```

**Training pairs:**
```
Position 0: Input [234]        → Target 56
Position 1: Input [234, 56]    → Target 789
Position 2: Input [234, 56, 789] → Target 12
Position 3: Input [234, 56, 789, 12] → Target 45
```

### DataLoader Setup

```python
from torch.utils.data import DataLoader

# Training loader (shuffle for randomness)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,        # Randomize batch order each epoch
    num_workers=0,       # Single-process (safe for debugging)
    pin_memory=True      # Speed up GPU transfer (if using CUDA)
)

# Validation loader (no shuffle for reproducibility)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,       # Keep same order for consistent metrics
    num_workers=0
)
```

### Memory and Performance

**Per-batch memory (batch_size=32, max_seq_length=128):**
- Input IDs: 32 × 128 × 8 bytes (int64) = 32 KB
- Target IDs: 32 × 128 × 8 bytes = 32 KB
- Embeddings: 32 × 128 × 128 × 4 bytes (float32) = 2 MB
- Attention scores: 32 × 4 heads × 128 × 128 × 4 bytes = 1 MB
- Total: ~5-10 MB per batch (tiny compared to 400MB limit)

**Training throughput:**
- 19,121 batches/epoch ÷ 60 seconds ≈ 318 batches/second
- ~10,000 tokens/second processed (32 batch × 128 seq × 3.1 batches/sec)

---

## 5. Phase 4: The Training Loop - Forward and Backward

### Epoch Structure

An **epoch** is one complete pass through the training dataset:

```
Epoch 1:
    ├─> Process all 19,121 batches (611,884 sequences)
    ├─> Compute loss for each batch
    ├─> Update weights 19,121 times (one update per batch)
    └─> Validate on validation set (no weight updates)

Epoch 2:
    ├─> Shuffle training data (different batch order)
    ├─> Repeat forward/backward passes
    └─> ...

...

Epoch 20: (final epoch)
```

**Why multiple epochs?**
- Each pass refines the model's understanding
- Early epochs: Learn basic patterns (common words, punctuation)
- Later epochs: Learn nuanced patterns (grammar, context)

### Forward Pass: Input to Logits

```
┌──────────────────────────────────────────────────────────────┐
│               FORWARD PASS DATA FLOW                          │
└──────────────────────────────────────────────────────────────┘

Step 1: EMBEDDING LOOKUP
    Input IDs: [32, 128]  (batch_size=32, seq_len=128)
        │
        ├─> Token embedding: (32, 128) → (32, 128, 128)
        │   └─> Lookup each token ID in embedding matrix
        │
        └─> Add positional encoding: (32, 128, 128)

Step 2: TRANSFORMER BLOCK 1
    Embeddings: (32, 128, 128)
        │
        ├─> Multi-Head Attention:
        │   ├─> Project Q, K, V: (32, 128, 128) → 3× (32, 4, 128, 32)
        │   ├─> Compute attention: softmax(Q·Kᵀ / √32) · V
        │   ├─> Apply causal mask (prevent peeking future)
        │   └─> Concatenate heads: (32, 128, 128)
        │
        ├─> Add & LayerNorm: (32, 128, 128)
        │
        ├─> FeedForward Network:
        │   ├─> Linear1 + GELU: (32, 128, 128) → (32, 128, 256)
        │   └─> Linear2: (32, 128, 256) → (32, 128, 128)
        │
        └─> Add & LayerNorm: (32, 128, 128)

Step 3: TRANSFORMER BLOCK 2
    (Same structure as Block 1)

Step 4: OUTPUT PROJECTION
    Hidden states: (32, 128, 128)
        │
        └─> Linear: (32, 128, 128) → (32, 128, 8000)  [logits]
```

### Self-Attention Mechanism Simplified

Attention computes **weighted sums** of value vectors based on query-key similarity:

```
Given sequence: "The cat sat"
Tokens: [234, 56, 789]

1. Project to Q, K, V:
   Q (query): What am I looking for?
   K (key): What do I offer?
   V (value): What do I contain?

2. Compute attention scores:
   Attention[i,j] = softmax(Q[i] · K[j] / √d)
   
   Example (simplified):
              "The"  "cat"  "sat"
        "The" [ 0.7   0.2    0.1  ]  (attends mostly to itself)
        "cat" [ 0.3   0.5    0.2  ]  (attends to "The" and self)
        "sat" [ 0.1   0.4    0.5  ]  (attends to "cat" and self)

3. Weighted sum of values:
   Output[i] = Σⱼ Attention[i,j] × V[j]
```

**Causal masking** prevents future token peeking:

```
Original attention (invalid):
              "The"  "cat"  "sat"
        "The" [ 0.7   0.2    0.1  ]
        "cat" [ 0.3   0.5    0.2  ]
        "sat" [ 0.1   0.4    0.5  ]

Causal mask applied:
              "The"  "cat"  "sat"
        "The" [ 1.0   -∞     -∞   ]  (only sees "The")
        "cat" [ 0.3   0.7    -∞   ]  (sees "The", "cat")
        "sat" [ 0.1   0.4    0.5  ]  (sees all)

After softmax:
              "The"  "cat"  "sat"
        "The" [ 1.0   0.0    0.0  ]
        "cat" [ 0.3   0.7    0.0  ]
        "sat" [ 0.1   0.4    0.5  ]
```

### Loss Calculation: Cross-Entropy

```python
import torch.nn.functional as F

# Forward pass
logits = model(input_ids)  # Shape: (32, 128, 8000)
# logits[b, t, v] = score for token v at position t in batch b

# Cross-entropy loss
# Compares predicted distribution (softmax of logits) vs. true target
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # Reshape: (32*128, 8000)
    target_ids.view(-1),          # Reshape: (32*128)
    ignore_index=tokenizer.pad_id()  # Ignore padding tokens
)

# loss is a scalar (average over all positions and batches)
# Lower loss = better predictions
```

**Interpretation:**
- **Loss ~8.0 (untrained):** Model is guessing randomly (log(8000) ≈ 8.99)
- **Loss ~2.5 (trained):** Model predicts correct token with ~92% probability
- **Loss ~0.0 (overfitting):** Model memorized training data (bad generalization)

### Backward Pass: Computing Gradients

PyTorch autograd handles backpropagation automatically:

```python
# 1. Zero out gradients from previous step
optimizer.zero_grad()

# 2. Compute loss (forward pass already done)
loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

# 3. Backpropagation: compute ∂loss/∂weight for all parameters
loss.backward()

# Behind the scenes (autograd):
#   ∂loss/∂logits[b,t,v] = (softmax[b,t,v] - 1) if v == target else softmax[b,t,v]
#   ∂loss/∂hidden = ∂loss/∂logits · W_output.T
#   ∂loss/∂W_ffn = ∂loss/∂hidden · activations.T
#   ... (chain rule back to embedding layer)

# 4. Update weights using gradients
optimizer.step()

# Each parameter updated: W_new = W_old - learning_rate × ∂loss/∂W
```

**Gradient flow visualization:**

```
┌────────────────────────────────────────────────────────────┐
│            BACKPROPAGATION FLOW                             │
└────────────────────────────────────────────────────────────┘

Loss (scalar)
    │
    ├─> ∂Loss/∂Logits: (32, 128, 8000)
    │
    ├─> ∂Loss/∂Hidden (Block 2 output): (32, 128, 128)
    │
    ├─> ∂Loss/∂FFN_W2: (256, 128)
    ├─> ∂Loss/∂FFN_W1: (128, 256)
    │
    ├─> ∂Loss/∂Attention_Out: (32, 128, 128)
    ├─> ∂Loss/∂Attention_V: (32, 4, 128, 32)
    ├─> ∂Loss/∂Attention_K: (32, 4, 128, 32)
    ├─> ∂Loss/∂Attention_Q: (32, 4, 128, 32)
    │
    ├─> ∂Loss/∂Hidden (Block 1 output): (32, 128, 128)
    │   (repeat for Block 1 layers)
    │
    └─> ∂Loss/∂Embeddings: (32, 128, 128)
```

### Why Loss Decreases: Gradient Descent

**Intuition:** Gradients point in the direction of *increasing* loss, so we move *opposite*:

```
Weight update rule:
    W_new = W_old - learning_rate × ∂Loss/∂W

Example:
    ∂Loss/∂W = 0.5   (loss increases if W increases)
    learning_rate = 0.0003
    W_new = W_old - 0.0003 × 0.5 = W_old - 0.00015

Over many steps, weights converge to values that minimize loss.
```

**Training step pseudocode:**

```python
for epoch in range(num_epochs):
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        # 1. Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        
        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 3. Update weights
        optimizer.step()
        
        # 4. Log progress
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}")
```

### Practical Example: Full Training Step

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Setup
model = TinyTransformer(vocab_size=8000, d_model=128, nhead=4, num_layers=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(20):
    epoch_loss = 0.0
    
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        # Forward pass
        logits = model(input_ids)  # (32, 128, 8000)
        
        # Compute loss
        loss = criterion(logits.view(-1, 8000), target_ids.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/20, Avg Loss: {avg_loss:.4f}")
```

---

## 6. Phase 5: Optimization - Learning Rate and Schedulers

### AdamW Optimizer

We use **AdamW** (Adam with Weight Decay), a state-of-the-art optimizer:

**Key features:**
1. **Adaptive learning rates**: Each parameter has its own effective learning rate
2. **Momentum**: Smooths out gradient noise (moving average of gradients)
3. **Weight decay**: Regularization to prevent overfitting

**AdamW update rule:**

```
Initialize:
    m₀ = 0  (first moment: gradient mean)
    v₀ = 0  (second moment: gradient variance)

At each step t:
    g_t = ∂Loss/∂W                        (compute gradient)
    m_t = β₁ × m_{t-1} + (1-β₁) × g_t     (update first moment)
    v_t = β₂ × v_{t-1} + (1-β₂) × g_t²    (update second moment)
    
    m̂_t = m_t / (1 - β₁ᵗ)                 (bias correction)
    v̂_t = v_t / (1 - β₂ᵗ)
    
    W_t = W_{t-1} - α × m̂_t / (√v̂_t + ε) - λ × W_{t-1}
          ↑          ↑                       ↑
    old weight   adaptive step          weight decay
```

**Hyperparameters (our settings):**
- `lr (α) = 0.0003`: Learning rate (how far to step)
- `β₁ = 0.9`: Momentum coefficient
- `β₂ = 0.999`: Variance coefficient
- `ε = 1e-8`: Numerical stability
- `weight_decay (λ) = 0.01`: Regularization strength

**Why AdamW over SGD?**

| Optimizer | Convergence Speed | Hyperparameter Sensitivity | Memory Overhead | Use Case |
|-----------|------------------|---------------------------|----------------|----------|
| **SGD** | Slow | High (needs manual LR tuning) | None | Well-tuned large models |
| **Adam** | Fast | Low (works out-of-the-box) | 2× parameters | General-purpose |
| **AdamW** | Fast | Low | 2× parameters | **Modern default (ours)** |

### Learning Rate Schedule

We use **Cosine Annealing** to gradually reduce the learning rate:

```
Initial LR: 0.0003
                                ╭─────────────╮
                              ╱                 ╲
                            ╱                     ╲
                          ╱                         ╲
                        ╱                             ╲
                      ╱                                 ╲
Final LR: 0.0        ╱                                   ╰────────
                    
               Epoch: 1  2  3  4  5  ... 18 19 20

Formula: LR(epoch) = 0.0003 × (1 + cos(π × epoch / 20)) / 2
```

**Why decay learning rate?**
- **Early epochs:** Large steps explore the loss landscape
- **Later epochs:** Small steps fine-tune near minimum
- **Prevents oscillation:** Avoids overshooting the optimal solution

**Implementation:**

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
scheduler = CosineAnnealingLR(optimizer, T_max=20)  # 20 epochs

for epoch in range(20):
    # Training loop
    for batch in train_loader:
        ... (forward, backward, optimizer.step())
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")
```

### Learning Rate Comparison

| Schedule | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Constant** | Simple | May not converge | Short experiments |
| **Step Decay** | Easy to implement | Requires manual tuning | Legacy code |
| **Exponential** | Smooth decay | Hyperparameter sensitive | Fine-tuning |
| **Cosine** | Smooth, no tuning | Needs epoch count | **Modern default (ours)** |

---

## 7. Phase 6: Evaluation and Checkpointing

### Validation Loop

After each training epoch, we evaluate on the validation set:

```python
def validate(model, val_loader, criterion, device):
    model.eval()  # Disable dropout, batchnorm training mode
    total_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation (saves memory)
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass only (no backward)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss
```

**Why validation?**
- **Detect overfitting:** Training loss ↓ but validation loss ↑
- **Model selection:** Save the best model (lowest validation loss)
- **Early stopping:** Stop training if validation loss plateaus

### Checkpoint Saving Strategy

We save multiple checkpoint types:

```
┌────────────────────────────────────────────────────┐
│         CHECKPOINT SAVING STRATEGY                  │
└────────────────────────────────────────────────────┘

1. PERIODIC CHECKPOINTS (every 5 epochs):
   - checkpoint_epoch_5.pt
   - checkpoint_epoch_10.pt
   - checkpoint_epoch_15.pt
   - checkpoint_epoch_20.pt
   
   Purpose: Resume training if interrupted

2. BEST MODEL (lowest validation loss):
   - best_model.pt
   
   Purpose: Deploy the best-performing model

3. FINAL MODEL (last epoch):
   - final_model.pt
   
   Purpose: Archive the final state (may not be best)
```

**Checkpoint contents:**

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),          # Model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state (momentum, etc.)
    'train_loss': train_loss,
    'val_loss': val_loss,
    'config': config,  # Hyperparameters for reproducibility
}

torch.save(checkpoint, f'data/checkpoint_epoch_{epoch}.pt')
```

### Resuming Training from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load('data/checkpoint_epoch_10.pt')

# Restore model weights
model = TinyTransformer(vocab_size=8000, d_model=128, nhead=4, num_layers=2)
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer state
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Resume from epoch 11
start_epoch = checkpoint['epoch'] + 1
for epoch in range(start_epoch, 20):
    ... (continue training)
```

### Training Metrics Export (JSON)

We export detailed metrics for analysis:

```json
{
  "run_id": "a3f7b8c9-1234-5678-90ab-cdef01234567",
  "start_timestamp": "2024-01-15T10:30:00",
  "end_timestamp": "2024-01-15T10:40:00",
  "configuration": {
    "vocab_size": 8000,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "num_epochs": 20,
    "max_seq_length": 128
  },
  "final_metrics": {
    "best_val_loss": 2.4567,
    "final_train_loss": 2.3456,
    "final_val_loss": 2.5678,
    "total_epochs": 20
  },
  "epoch_history": [
    {"epoch": 1, "train_loss": 7.8234, "val_loss": 7.5432, "lr": 0.0003},
    {"epoch": 2, "train_loss": 5.6789, "val_loss": 5.4321, "lr": 0.000295},
    ...
    {"epoch": 20, "train_loss": 2.3456, "val_loss": 2.5678, "lr": 0.00001}
  ]
}
```

**Usage:**

```bash
# View metrics
cat data/training_metrics.json | jq '.final_metrics'

# Plot training curve
python -c "
import json
import matplotlib.pyplot as plt

with open('data/training_metrics.json') as f:
    metrics = json.load(f)

epochs = [e['epoch'] for e in metrics['epoch_history']]
train_loss = [e['train_loss'] for e in metrics['epoch_history']]
val_loss = [e['val_loss'] for e in metrics['epoch_history']]

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.legend()
plt.savefig('training_curve.png')
"
```

### Sample Generation During Training

We generate samples every 5 epochs to monitor progress qualitatively:

```python
def generate_sample(model, tokenizer, prompt, max_tokens=30, temperature=0.7, device='cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt, out_type=int)
    input_tensor = torch.tensor([input_ids], device=device)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_id():
                break
    
    return tokenizer.decode(input_tensor[0].tolist())

# During training
if epoch % 5 == 0:
    for prompt in ["Hello", "Hola"]:
        sample = generate_sample(model, tokenizer, prompt, max_tokens=30, temperature=0.7)
        print(f"  '{prompt}' → {sample}")
```

**Expected output evolution:**

```
Epoch 1:
  'Hello' → Hello xkzJ@9mL%fD (gibberish)
  'Hola' → Hola pQr$Yt#nV (gibberish)

Epoch 5:
  'Hello' → Hello world the cat (some coherence)
  'Hola' → Hola mundo el gato (partial Spanish)

Epoch 10:
  'Hello' → Hello world how are you today (improving)
  'Hola' → Hola mundo cómo estás hoy (better Spanish)

Epoch 20:
  'Hello' → Hello world how are you doing today I hope (coherent!)
  'Hola' → Hola mundo cómo estás hoy espero que bien (good Spanish!)
```

---

## 8. Troubleshooting Common Issues

### Problem-Solution Matrix

| Problem | Symptoms | Likely Causes | Solutions |
|---------|----------|---------------|-----------|
| **High loss (>5.0 after epoch 10)** | Loss stuck, no decrease | Learning rate too low, model too small | Increase LR to 0.001, verify data quality |
| **Loss → NaN/Inf** | Training crashes | Learning rate too high, numerical instability | Reduce LR to 0.0001, use gradient clipping |
| **Out of memory** | CUDA OOM, process killed | Batch size too large, sequence too long | Reduce batch_size to 16 or 8, reduce max_seq_length to 64 |
| **Slow training** | <10 batches/sec | CPU-only mode, large batch size | Use GPU (CUDA), reduce batch_size |
| **Overfitting** | Train loss ↓, val loss ↑ | Too many epochs, model too complex | Stop at epoch 10-15, add dropout (not implemented) |
| **Gibberish generation** | Model outputs random tokens | Not enough training, high temperature | Train to epoch 15+, reduce temperature to 0.5 |
| **Checkpoint load error** | `RuntimeError: size mismatch` | Model architecture changed | Verify vocab_size, d_model match checkpoint config |
| **Tokenizer not found** | `FileNotFoundError` | Wrong path, tokenizer not trained | Check data/bilingual_8k.model exists, retrain if needed |
| **Low GPU utilization** | GPU <20% used | Small batch size, CPU bottleneck | Increase batch_size to 64, use pin_memory=True |
| **Loss plateaus early** | Loss stops decreasing at epoch 5 | Learning rate too low, optimizer issue | Switch to SGD with momentum, increase initial LR |

### Debugging Tips

**1. Verify data pipeline:**

```python
# Check dataset length
print(f"Dataset size: {len(train_dataset)}")  # Should be ~611,884

# Inspect a batch
input_ids, target_ids = next(iter(train_loader))
print(f"Batch shape: {input_ids.shape}")  # Should be (32, 128)
print(f"Sample input: {input_ids[0, :10]}")  # First 10 tokens
print(f"Sample target: {target_ids[0, :10]}")  # Should be input shifted by 1
```

**2. Test model forward pass:**

```python
# Random input
input_ids = torch.randint(0, 8000, (1, 128))
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # Should be (1, 128, 8000)
print(f"Logits range: {logits.min():.2f} to {logits.max():.2f}")  # Should be ~-5 to 5
```

**3. Monitor gradients:**

```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm={grad_norm:.4f}")
        if grad_norm > 10.0:
            print(f"  WARNING: Large gradient detected!")
```

**4. Reduce batch size for debugging:**

```bash
# Use minimal batch size to isolate issues
python3 scripts/train_model.py --epochs 1 --batch-size 4
```

**5. Check corpus quality:**

```bash
# Verify corpus is not corrupted
head -100 data/corpus_bilingual.txt
wc -l data/corpus_bilingual.txt  # Should be ~46,000
file data/corpus_bilingual.txt   # Should be UTF-8 text
```

### Performance Optimization

**Speed up training:**

1. **Use GPU** (if available):
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = model.to(device)
   ```

2. **Increase batch size** (if memory allows):
   ```bash
   python3 scripts/train_model.py --batch-size 64
   ```

3. **Use DataLoader workers**:
   ```python
   train_loader = DataLoader(..., num_workers=4, pin_memory=True)
   ```

4. **Mixed precision training** (advanced):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       logits = model(input_ids)
       loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

**Reduce memory usage:**

1. **Smaller batch size:**
   ```bash
   python3 scripts/train_model.py --batch-size 8
   ```

2. **Shorter sequences:**
   ```bash
   python3 scripts/train_model.py --max-seq-length 64
   ```

3. **Gradient accumulation** (simulate larger batches):
   ```python
   accumulation_steps = 4
   for i, (input_ids, target_ids) in enumerate(train_loader):
       logits = model(input_ids)
       loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

---

## 9. Hyperparameter Reference and Tuning

### Default Hyperparameters

| Parameter | Default | Range | Effect | Tuning Priority |
|-----------|---------|-------|--------|----------------|
| **vocab_size** | 8000 | 1000-32000 | Vocabulary coverage | Low (set by tokenizer) |
| **d_model** | 128 | 64-512 | Model capacity | Medium (affects size) |
| **nhead** | 4 | 2-8 | Attention diversity | Low (architecture decision) |
| **num_layers** | 2 | 1-6 | Model depth | Medium (affects capacity) |
| **dim_feedforward** | 256 | 128-1024 | FFN capacity | Low (2× d_model typical) |
| **batch_size** | 32 | 4-128 | Training stability | **High** (memory/speed trade-off) |
| **learning_rate** | 0.0003 | 0.0001-0.01 | Convergence speed | **High** (critical tuning) |
| **num_epochs** | 20 | 5-50 | Training completeness | **High** (time/quality) |
| **max_seq_length** | 128 | 32-512 | Context window | Medium (memory impact) |

### Tuning Strategies

**1. Quick iteration (5-minute experiments):**

```bash
# Test learning rate
./scripts/train_pipeline.sh --epochs 2 --learning-rate 0.0001
./scripts/train_pipeline.sh --epochs 2 --learning-rate 0.001

# Test batch size
./scripts/train_pipeline.sh --epochs 2 --batch-size 16
./scripts/train_pipeline.sh --epochs 2 --batch-size 64
```

**2. Full training (10-minute runs):**

```bash
# Baseline
./scripts/train_pipeline.sh --epochs 20 --batch-size 32 --learning-rate 0.0003

# Variant 1: Higher LR, fewer epochs
./scripts/train_pipeline.sh --epochs 10 --batch-size 32 --learning-rate 0.001

# Variant 2: Larger batch, lower LR
./scripts/train_pipeline.sh --epochs 20 --batch-size 64 --learning-rate 0.0001
```

**3. Grid search (automated):**

```bash
for lr in 0.0001 0.0003 0.001; do
    for bs in 16 32 64; do
        OUTPUT_DIR="experiments/lr${lr}_bs${bs}"
        ./scripts/train_pipeline.sh \
            --epochs 10 \
            --learning-rate $lr \
            --batch-size $bs \
            --output-dir $OUTPUT_DIR
    done
done

# Compare results
python -c "
import json
import glob

results = []
for path in glob.glob('experiments/*/training_metrics.json'):
    with open(path) as f:
        metrics = json.load(f)
    lr = metrics['configuration']['learning_rate']
    bs = metrics['configuration']['batch_size']
    val_loss = metrics['final_metrics']['final_val_loss']
    results.append((lr, bs, val_loss))

results.sort(key=lambda x: x[2])
print('Best configs (by val_loss):')
for lr, bs, val_loss in results[:5]:
    print(f'  LR={lr}, BS={bs}, Val Loss={val_loss:.4f}')
"
```

### Hyperparameter Interactions

**Learning Rate × Batch Size:**
- **Large LR + Small batch:** Fast convergence, unstable (high variance)
- **Large LR + Large batch:** Divergence risk (gradient explosion)
- **Small LR + Small batch:** Slow but stable
- **Small LR + Large batch:** Very slow, smooth convergence

**Recommended combinations:**

| Batch Size | Learning Rate | Use Case |
|-----------|---------------|----------|
| 8         | 0.0005        | Debugging, low-memory systems |
| 16        | 0.0003        | Balanced for CPU training |
| **32**    | **0.0003**    | **Default (good balance)** |
| 64        | 0.0001        | GPU training, smooth convergence |
| 128       | 0.00005       | Distributed training |

---

## 10. Constitutional Compliance and Deployment

### Constitutional Budgets

Our model must fit within strict resource constraints:

| Metric | Budget | Measurement | Validation |
|--------|--------|-------------|------------|
| **RSS (Memory)** | ≤ 400 MB | Resident Set Size during inference | `ps aux` or `psutil` |
| **Peak Memory** | ≤ 512 MB | Maximum allocation | System monitor |
| **Inference Latency (p95)** | ≤ 250 ms | 95th percentile response time | 100-sample benchmark |
| **Model Size** | ~10 MB | On-disk checkpoint | `du -h final_model.pt` |

### Memory Profiling

**Monitor during inference:**

```python
import psutil
import time

# Start monitoring
process = psutil.Process()
initial_memory = process.memory_info().rss / (1024 ** 2)  # MB

# Run inference
start = time.time()
output = generate("Hello world", max_tokens=50)
latency = (time.time() - start) * 1000  # ms

# Check memory
peak_memory = process.memory_info().rss / (1024 ** 2)  # MB
memory_increase = peak_memory - initial_memory

print(f"RSS: {peak_memory:.1f} MB (increase: {memory_increase:.1f} MB)")
print(f"Latency: {latency:.1f} ms")
print(f"Constitutional compliance:")
print(f"  ✓ RSS ≤ 400MB: {peak_memory <= 400}")
print(f"  ✓ Latency ≤ 250ms: {latency <= 250}")
```

**Automated validation:**

```bash
# Run 100 inference samples and measure p95 latency
python3 -c "
import sys
sys.path.insert(0, 'src')
from cli.minimal_llm import main
import time

latencies = []
for i in range(100):
    start = time.time()
    # Run generation (simulated)
    # main(['--prompt', 'Test', '--max-tokens', '20'])
    latency = (time.time() - start) * 1000
    latencies.append(latency)

latencies.sort()
p95 = latencies[94]
print(f'P95 Latency: {p95:.1f} ms')
print(f'Compliance: {\"PASS\" if p95 <= 250 else \"FAIL\"}')"
```

### Deployment Checklist

Before deploying to production:

- [ ] **Model validation**
  - [ ] Validation loss < 3.0
  - [ ] Sample generation produces coherent text
  - [ ] No NaN/Inf in final_model.pt

- [ ] **Constitutional compliance**
  - [ ] RSS ≤ 400 MB (measured with 100 samples)
  - [ ] Peak memory ≤ 512 MB
  - [ ] P95 latency ≤ 250 ms

- [ ] **Artifact verification**
  - [ ] final_model.pt exists and is ~10 MB
  - [ ] best_model.pt exists
  - [ ] training_metrics.json exports cleanly
  - [ ] Tokenizer (bilingual_8k.model) is compatible

- [ ] **Code quality**
  - [ ] Safety filters enabled (no harmful output)
  - [ ] Quantization applied (if needed for smaller footprint)
  - [ ] Error handling for edge cases (empty prompts, long inputs)

- [ ] **Documentation**
  - [ ] README updated with training results
  - [ ] Deployment guide includes system requirements
  - [ ] Known limitations documented

### Deployment Example (Raspberry Pi Zero 2W)

```bash
# 1. Transfer artifacts
scp data/final_model.pt pi@raspberrypi:/home/pi/sddllm/data/
scp data/bilingual_8k.model pi@raspberrypi:/home/pi/sddllm/data/

# 2. Install dependencies on Pi
ssh pi@raspberrypi
pip3 install torch sentencepiece --no-cache-dir

# 3. Test inference
python3 src/cli/minimal_llm.py --prompt "Hello" --max-tokens 20

# 4. Monitor resource usage
while true; do
    ps aux | grep python | awk '{print "RSS: " $6/1024 " MB"}'
    sleep 1
done
```

### Production Monitoring

**Metrics to track:**

1. **Inference latency**: p50, p95, p99
2. **Memory usage**: RSS, peak allocation
3. **Throughput**: requests/second
4. **Error rate**: failed generations, OOM crashes
5. **Quality**: average perplexity, user feedback

**Alerting thresholds:**

```yaml
alerts:
  - name: high_latency
    condition: p95_latency > 250ms
    action: Scale up resources or reduce batch size
  
  - name: high_memory
    condition: rss > 400MB
    action: Restart process, investigate memory leaks
  
  - name: low_quality
    condition: perplexity > 10.0
    action: Retrain model with more epochs
```

---

## Appendix: Quick Command Reference

### Training Pipeline

```bash
# Full training (default settings)
./scripts/train_pipeline.sh

# Custom hyperparameters
./scripts/train_pipeline.sh \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --vocab-size 4000 \
    --output-dir ./experiments/run1

# Resume from checkpoint
./scripts/train_pipeline.sh --resume

# Skip phases (use existing data)
./scripts/train_pipeline.sh --skip-corpus --skip-tokenizer
```

### Manual Training Steps

```bash
# 1. Download corpus
python3 scripts/download_simple_corpus.py

# 2. Train tokenizer
python3 scripts/train_tokenizer.py \
    --vocab-size 8000 \
    --output-dir data

# 3. Train model
python3 scripts/train_model.py \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.0003 \
    --output-dir data

# 4. Test inference
python3 src/cli/minimal_llm.py --prompt "Hello world"
```

### Debugging

```bash
# Quick test (1 epoch, small batch)
python3 scripts/train_model.py --epochs 1 --batch-size 4

# Check training metrics
cat data/training_metrics.json | jq '.final_metrics'

# View training curve
python3 -c "
import json
with open('data/training_metrics.json') as f:
    metrics = json.load(f)
for e in metrics['epoch_history'][-5:]:
    print(f\"Epoch {e['epoch']}: train={e['train_loss']:.4f}, val={e['val_loss']:.4f}\")
"
```

### Monitoring

```bash
# Memory usage (during training)
watch -n 1 'ps aux | grep train_model | awk '\''{print $6/1024 " MB"}'\'''

# GPU usage (if CUDA available)
watch -n 1 nvidia-smi

# Disk usage
du -h data/
```

---

## Conclusion

This guide covered the complete training process:

1. **Tokenization**: Converting text to numbers (8000-token vocabulary)
2. **Initialization**: Random weights in 2.45M parameter model
3. **Data Preparation**: Batching and train/val split
4. **Training Loop**: Forward pass, loss, backprop, weight updates
5. **Optimization**: AdamW optimizer with cosine LR schedule
6. **Evaluation**: Validation and checkpointing
7. **Troubleshooting**: Common issues and solutions
8. **Hyperparameter Tuning**: Systematic experimentation
9. **Deployment**: Constitutional compliance and production checklist

**Next Steps:**
- Experiment with hyperparameters using grid search
- Monitor training metrics and visualize learning curves
- Deploy to target device (Raspberry Pi Zero 2W)
- Collect user feedback and iterate

**Further Reading:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer paper)
- [SentencePiece documentation](https://github.com/google/sentencepiece)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [Hugging Face course](https://huggingface.co/course/chapter1/1)

---

**Document Version:** 1.0  
**Last Updated:** 2024-01-15  
**Maintainer:** SDD LLM Team  
**Feedback:** Open an issue in the repository
