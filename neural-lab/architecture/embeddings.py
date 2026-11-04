
import torch
import torch.nn as nn

from architecture.config import ModelConfig
from architecture.positional import PositionalEmbedding


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class InputEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionalEmbedding(config)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:

        token_vectors = self.token_embedding(token_ids)

        position_vectors = self.position_embedding(
            token_ids,
            position_offset=position_offset,
        )

        return token_vectors + position_vectors
