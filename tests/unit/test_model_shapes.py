import torch
from src.models.tiny_transformer import TinyTransformer


def test_forward_shapes():
    model = TinyTransformer(vocab_size=128, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    input_ids = torch.randint(0, 128, (1, 8))
    logits = model(input_ids)
    assert logits.shape == (1, 8, 128)


def test_model_parameters():
    model = TinyTransformer(vocab_size=8000, d_model=128, nhead=4, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    # Ensure model is small enough for 1GB devices
    assert total_params < 10_000_000  # Less than 10M parameters
