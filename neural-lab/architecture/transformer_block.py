
import torch
import torch.nn as nn

from architecture.config import ModelConfig
from architecture.multi_head_attention import MultiHeadCausalSelfAttention
from architecture.feed_forward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(config)

        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForwardNetwork(config)

    def forward(
        self,
        x: torch.Tensor,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:

        attention_output = self.attention(
            self.norm1(x),
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
        )

        x = x + attention_output
        x = x + self.feed_forward(self.norm2(x))

        return x
