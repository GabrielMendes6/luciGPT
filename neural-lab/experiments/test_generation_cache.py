
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT
from architecture.generation import generate


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

token_ids = torch.randint(
    0,
    config.vocab_size,
    (2, 12),
)


@torch.inference_mode()
def reference_generate(model, token_ids, max_new_tokens):
    """Geração de referência sem KV Cache."""

    generated = token_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(generated)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            (generated, next_token),
            dim=1,
        )

    return generated


print("===== TESTE DE GERAÇÃO COM KV CACHE =====")

with torch.inference_mode():

    reference = reference_generate(
        model,
        token_ids,
        max_new_tokens=10,
    )

    cached = generate(
        model,
        token_ids,
        max_new_tokens=10,
    )

    print("Referência:", reference.shape)
    print("KV Cache:", cached.shape)

    print(
        "Tokens iguais:",
        torch.equal(reference, cached),
    )

    assert torch.equal(reference, cached)

print("\n===== TESTE DE ESTADO =====")

model.train()

generate(
    model,
    token_ids,
    max_new_tokens=3,
)

print("Estado preservado:", model.training)

assert model.training is True

print("\n===== TESTE DE LIMITE =====")

try:
    generate(
        model,
        token_ids,
        max_new_tokens=120,
    )

except ValueError as error:
    print("Limite validado:", error)

else:
    raise AssertionError("O limite de contexto não foi aplicado.")

print("\n===== TESTE DE ZERO TOKENS =====")

result = generate(
    model,
    token_ids,
    max_new_tokens=0,
)
    
assert torch.equal(result, token_ids)

print("Prompt preservado:", torch.equal(result, token_ids))

print("\nGERAÇÃO COM KV CACHE VALIDADA")
