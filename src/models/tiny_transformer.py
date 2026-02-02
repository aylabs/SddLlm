import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    """Minimal Transformer for 1GB devices."""
    
    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(1024, d_model)  # max 1k context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Generate causal mask to prevent attending to future positions."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Generate causal mask to prevent attending to future tokens
        causal_mask = self._generate_causal_mask(seq_len, input_ids.device)
        
        x = self.embedding(input_ids) + self.pos_encoding(positions)
        x = self.encoder(x, mask=causal_mask)
        logits = self.lm_head(x)
        return logits
