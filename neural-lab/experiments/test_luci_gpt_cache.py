
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig(
    vocab_size=848,
    embedding_dim=128,
    max_seq_len=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers=12,
    ffn_dim=512,
)

model = LuciGPT(config)
model.eval()

batch_size = 2
sequence_length = 12

token_ids = torch.randint(
    0,
    config.vocab_size,
    (batch_size, sequence_length),
)

with torch.inference_mode():

    # A: processamento convencional.
    full_logits = model(token_ids)

    # B: processamento incremental.
    cache = None
    incremental_logits = []

    for position in range(sequence_length):

        current_tokens = token_ids[:, position:position + 1]

        logits, cache = model(
            current_tokens,
            past_key_values=cache,
            use_cache=True,
        )

        incremental_logits.append(logits)

    incremental_logits = torch.cat(
        incremental_logits,
        dim=1,
    )

    difference = (full_logits - incremental_logits).abs()

    print("===== LUCI GPT KV CACHE =====")
    print("Referência:", full_logits.shape)
    print("Incremental:", incremental_logits.shape)
    print("Quantidade de caches:", len(cache))
    print("Camadas esperadas:", config.num_layers)
    print("Cache K primeira camada:", cache[0][0].shape)
    print("Cache V primeira camada:", cache[0][1].shape)
    print("Cache K última camada:", cache[-1][0].shape)
    print("Cache V última camada:", cache[-1][1].shape)
    print("Diferença máxima:", difference.max().item())
    print("Diferença média:", difference.mean().item())

    assert len(cache) == config.num_layers

    assert torch.allclose(
        full_logits,
        incremental_logits,
        atol=1e-5,
        rtol=1e-4,
    )

print("\nLUCI GPT KV CACHE VALIDADO")
