from typing import List

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover
    spm = None

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"


class Tokenizer:
    """Bilingual (EN+ES) tokenizer with fallback to simple hash-based encoding."""

    def __init__(self, vocab_size: int = 8000, spm_model_path: str | None = None):
        self.vocab_size = vocab_size
        self._sp = None
        if spm is not None and spm_model_path:
            self._sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        # Special token IDs (reserved)
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        if self._sp is not None:
            ids = self._sp.encode(text, out_type=int)
        else:
            # Fallback: simple whitespace hash-based tokens (reserve 0-2 for special)
            tokens = text.strip().split()
            ids = [3 + (abs(hash(t)) % (self.vocab_size - 3)) for t in tokens]
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Filter out special tokens
        filtered = [i for i in ids if i not in [self.bos_id, self.eos_id, self.pad_id]]
        if self._sp is not None:
            return self._sp.decode(filtered)
        # Fallback decode: join placeholder tokens
        return " ".join(f"tok{i}" for i in filtered)

    def truncate(self, ids: List[int], max_length: int = 1000) -> List[int]:
        """Truncate to max context window (1k tokens per spec)."""
        return ids[:max_length]
