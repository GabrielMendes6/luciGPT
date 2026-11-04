import torch
import torch.nn as nn

from architecture.config import ModelConfig


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        hidden_dim = config.ffn_dim

        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)