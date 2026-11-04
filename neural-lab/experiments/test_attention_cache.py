import torch

from architecture.config import ModelConfig
from architecture.multi_head_attention import (
    MultiHeadCausalSelfAttention,
)


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

attention = MultiHeadCausalSelfAttention(config)
attention.eval()

batch_size = 2
sequence_length = 12

x = torch.randn(batch_size, sequence_length, 128)

with torch.inference_mode():

    # Referência: sequência inteira.
    full_output = attention(x)

    # Processamento incremental com KV Cache.
    cache = None
    incremental_outputs = []

    for position in range(sequence_length):
        token = x[:, position:position + 1, :]

        output, cache = attention(
            token,
            past_key_value=cache,
            use_cache=True,
        )

        incremental_outputs.append(output)

    incremental_output = torch.cat(
        incremental_outputs,
        dim=1,
    )

    difference = (full_output - incremental_output).abs()

    print("===== KV CACHE =====")
    print("Referência:", full_output.shape)
    print("Incremental:", incremental_output.shape)
    print("Cache K:", cache[0].shape)
    print("Cache V:", cache[1].shape)
    print("Diferença máxima:", difference.max().item())
    print("Diferença média:", difference.mean().item())

    assert torch.allclose(
        full_output,
        incremental_output,
        atol=1e-5,
        rtol=1e-4,
    )

print("\nKV CACHE VALIDADO")
