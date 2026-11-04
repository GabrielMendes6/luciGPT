
import torch

from architecture.config import ModelConfig
from architecture.transformer_block import TransformerBlock


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

block = TransformerBlock(config)
block.eval()

x = torch.randn(2, 12, 128)

with torch.inference_mode():

    full_output = block(x)

    cache = None
    outputs = []

    for position in range(12):
        token = x[:, position:position + 1, :]

        output, cache = block(
            token,
            past_key_value=cache,
            use_cache=True,
        )

        outputs.append(output)

    incremental_output = torch.cat(outputs, dim=1)

    difference = (full_output - incremental_output).abs()

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

print("\nTRANSFORMER BLOCK KV CACHE VALIDADO")
