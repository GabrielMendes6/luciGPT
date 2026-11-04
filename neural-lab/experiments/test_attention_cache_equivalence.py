
import torch

from architecture.config import ModelConfig
from architecture.multi_head_attention import MultiHeadCausalSelfAttention
from architecture.kv_cache import KVCache


torch.manual_seed(42)
torch.set_num_threads(2)

config = ModelConfig(
    vocab_size=848,
    embedding_dim=128,
    max_seq_len=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
    ffn_dim=512,
)

attention = MultiHeadCausalSelfAttention(config).eval()

batch_size = 2
sequence_length = 12

x = torch.randn(
    batch_size,
    sequence_length,
    config.embedding_dim,
)

# Referência: processamento completo.
with torch.inference_mode():
    reference = attention(x)

# Teste: processamento incremental com cache.
cache = KVCache.create(
    config=config,
    batch_size=batch_size,
    device=x.device,
    dtype=x.dtype,
)

with torch.inference_mode():
    first_output = attention(
        x[:, :8, :],
        kv_cache=cache,
        layer_index=0,
        cache_position=0,
    )

    cache.advance(8)

    second_output = attention(
        x[:, 8:, :],
        kv_cache=cache,
        layer_index=0,
        cache_position=8,
    )

incremental = torch.cat(
    [first_output, second_output],
    dim=1,
)

difference = (reference - incremental).abs()

print("===== ATTENTION CACHE EQUIVALENCE =====")
print("Reference shape:", reference.shape)
print("Incremental shape:", incremental.shape)
print("Maximum difference:", difference.max().item())
print("Mean difference:", difference.mean().item())
print(
    "Equivalent:",
    torch.allclose(
        reference,
        incremental,
        atol=1e-5,
        rtol=1e-4,
    ),
)
