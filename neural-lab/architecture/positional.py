
import torch
import torch.nn as nn

from architecture.config import ModelConfig


class PositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.max_seq_len = config.max_seq_len

        self.embedding = nn.Embedding(
            num_embeddings=config.max_seq_len,
            embedding_dim=config.embedding_dim,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:

        sequence_length = token_ids.size(1)

        if position_offset < 0:
            raise ValueError("position_offset não pode ser negativo.")

        if position_offset + sequence_length > self.max_seq_len:
            raise ValueError(
                "Posições excedem o contexto máximo do modelo."
            )

        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=token_ids.device,
        )

        return self.embedding(positions)
