import torch

from architecture.attention import CausalSelfAttention
from architecture.config import ModelConfig
from architecture.embeddings import InputEmbedding


config = ModelConfig()

embedding = InputEmbedding(config)
attention = CausalSelfAttention(config.embedding_dim)

tokens = torch.tensor([
    [12, 45, 103, 8],
    [7, 21, 90, 14],
])

x = embedding(tokens)
output = attention(x)

print("===== LUCI SELF-ATTENTION =====")
print("Entrada:", x.shape)
print("Saída:", output.shape)

parameters = sum(
    p.numel()
    for p in attention.parameters()
)

print("Parâmetros da atenção:", parameters)

loss = output.mean()
loss.backward()

print("Gradientes Q:", attention.query.weight.grad is not None)
print("Gradientes K:", attention.key.weight.grad is not None)
print("Gradientes V:", attention.value.weight.grad is not None)

print("\n===== TESTE DE MÁSCARA CAUSAL =====")

with torch.no_grad():
    original_output = attention(x)

    modified_x = x.clone()

    # Alteramos drasticamente o último token.
    modified_x[:, 3, :] += 100

    modified_output = attention(modified_x)

    # As três primeiras posições não podem ser afetadas.
    causal_valid = torch.allclose(
        original_output[:, :3, :],
        modified_output[:, :3, :],
        atol=1e-6,
    )

print("Tokens anteriores preservados:", causal_valid)