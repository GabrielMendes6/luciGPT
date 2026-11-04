import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(128, 128)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(self.embedding_dim)

        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        attended = attention_weights @ v

        return self.output_projection(attended)
