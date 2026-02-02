import torch
import torch.nn as nn
import mmap
from pathlib import Path
from typing import Optional


def quantize_int8_dynamic(model: nn.Module) -> nn.Module:
    """Applies dynamic quantization to Linear layers."""
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def load_weights_mmap(weights_path: Path) -> Optional[mmap.mmap]:
    """Load model weights using memory-mapped file for efficient loading."""
    if not weights_path.exists():
        return None
    with open(weights_path, "r+b") as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def load_quantized_state_dict(weights_path: Path) -> Optional[dict]:
    """Load quantized model state dict from file."""
    if not weights_path.exists():
        return None
    return torch.load(weights_path, map_location="cpu")
