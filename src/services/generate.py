import torch
import time
import logging
from src.models.tiny_transformer import TinyTransformer
from src.models.tokenizer import Tokenizer
from src.lib.runtime import GenerationMetrics, current_rss_mb
from src.services.safety import check_safety, get_safety_rationale

# Configure logging for auditable safety outcomes
logging.basicConfig(level=logging.INFO)


def generate_text_core(
    prompt: str, max_tokens: int = 64, vocab_size: int = 8000, temperature: float = 0.7
) -> tuple[str, GenerationMetrics]:
    """Core generation loop with metrics tracking."""
    tok = Tokenizer(vocab_size=vocab_size)
    model = TinyTransformer(vocab_size=vocab_size)
    model.eval()

    input_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    input_ids = tok.truncate(input_ids, max_length=1000)
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
            if next_id == tok.eos_id:
                break
            input_tensor = torch.tensor([generated_ids], dtype=torch.long)

    elapsed_ms = (time.time() - start_time) * 1000
    p95_latency = int(sorted(token_times)[int(len(token_times) * 0.95)]) if token_times else 0
    tokens_per_sec = len(generated_ids) / max(elapsed_ms / 1000.0, 1e-3)
    
    rss_end = current_rss_mb()
    peak_mb = max(rss_start, rss_end)

    output_text = tok.decode(generated_ids)
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
