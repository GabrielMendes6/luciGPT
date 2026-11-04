
import torch

from architecture.config import ModelConfig
from architecture.positional import PositionalEmbedding


torch.manual_seed(42)

config = ModelConfig(
    vocab_size=848,
    embedding_dim=128,
    max_seq_len=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
    ffn_dim=512,
)

embedding = PositionalEmbedding(config)

token_ids = torch.zeros((1, 12), dtype=torch.long)

# Posições da sequência completa.
full_positions = embedding(token_ids, position_offset=0)

# Simula o processamento dos primeiros 10 tokens.
cached_positions = embedding(
    token_ids[:, 10:11],
    position_offset=10,
)

# Compara a posição 10 da sequência completa
# com a posição atribuída ao token incremental.
difference = (
    full_positions[10:11, :] - cached_positions
).abs()

print("Sequência completa:", full_positions.shape)
print("Token incremental:", cached_positions.shape)
print("Diferença máxima:", difference.max().item())

assert torch.equal(
    full_positions[10:11, :],
    cached_positions,
)

print("\nPOSICIONAMENTO ABSOLUTO VALIDADO")
