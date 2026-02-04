import torch
import time
import logging
import sentencepiece as spm
from pathlib import Path
from src.models.tiny_transformer import TinyTransformer
from src.models.tokenizer import Tokenizer
from src.lib.runtime import GenerationMetrics, current_rss_mb
from src.services.safety import check_safety, get_safety_rationale

# Configure logging for auditable safety outcomes
logging.basicConfig(level=logging.INFO)

# Global model cache to avoid reloading
_model_cache = None
_tokenizer_cache = None


def load_trained_model(model_path: str = "data/best_model.pt", vocab_size: int = 8000):
    """Load trained model from checkpoint."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    
    model = TinyTransformer(vocab_size=vocab_size)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded trained model from {model_path} (epoch {checkpoint.get('epoch', '?')})")
    else:
        logging.warning(f"Model checkpoint not found at {model_path}, using random weights")
    
    model.eval()
    _model_cache = model
    return model


def load_trained_tokenizer(tokenizer_path: str = "data/bilingual_8k.model"):
    """Load trained SentencePiece tokenizer."""
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache
    
    if Path(tokenizer_path).exists():
        tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        logging.info(f"Loaded trained tokenizer from {tokenizer_path}")
        _tokenizer_cache = tokenizer
        return tokenizer
    else:
        logging.warning(f"Tokenizer not found at {tokenizer_path}, using fallback")
        return Tokenizer(vocab_size=8000)


def generate_text_core(
    prompt: str, max_tokens: int = 64, vocab_size: int = 8000, temperature: float = 0.7
) -> tuple[str, GenerationMetrics]:
    """Core generation loop with metrics tracking."""
    tokenizer = load_trained_tokenizer()
    model = load_trained_model()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, out_type=int)
    if len(input_ids) > 1000:
        input_ids = input_ids[-1000:]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)


    rss_start = current_rss_mb()
    start_time = time.time()
    token_times = []

    generated_ids = input_ids.copy()
    with torch.no_grad():
        for _ in range(max_tokens):
            token_start = time.time()
            logits = model(input_tensor)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            token_times.append((time.time() - token_start) * 1000)
            
            generated_ids.append(next_id)
            # Check for EOS token (SentencePiece uses ID 1 for EOS)
            if next_id == 1:
                break
            input_tensor = torch.tensor([generated_ids], dtype=torch.long)

    elapsed_ms = (time.time() - start_time) * 1000
    p95_latency = int(sorted(token_times)[int(len(token_times) * 0.95)]) if token_times else 0
    tokens_per_sec = len(generated_ids) / max(elapsed_ms / 1000.0, 1e-3)
    
    rss_end = current_rss_mb()
    peak_mb = max(rss_start, rss_end)

    output_text = tokenizer.decode(generated_ids)
    metrics = GenerationMetrics(
        latency_p95_ms=p95_latency,
        tokens_per_sec=tokens_per_sec,
        rss_mb=rss_end,
        peak_mb=peak_mb,
    )
    return output_text, metrics


def generate_with_safety(prompt: str, max_tokens: int = 64, temperature: float = 0.7):
    """Generate text with safety checks and auditable logging."""
    status, category = check_safety(prompt)
    rationale = get_safety_rationale(category)
    
    if status == "refuse":
        logging.warning(f"Refused prompt due to safety: {rationale}")
        return {
            "text": "Request refused due to safety policy.",
            "metrics": {},
            "safety": {"status": status, "category": category, "rationale": rationale},
        }
    
    text, metrics = generate_text_core(prompt, max_tokens=max_tokens, temperature=temperature)
    return {
        "text": text,
        "metrics": metrics.__dict__,
        "safety": {"status": status, "category": category, "rationale": rationale},
    }
