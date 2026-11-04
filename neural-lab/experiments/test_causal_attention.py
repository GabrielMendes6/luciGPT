import torch

from architecture.config import ModelConfig
from architecture.multi_head_attention import (
    MultiHeadCausalSelfAttention,
)


torch.manual_seed(42)

config = ModelConfig()
attention = MultiHeadCausalSelfAttention(config)

attention.eval()

x1 = torch.randn(1, 8, config.embedding_dim)
x2 = x1.clone()

# Alteramos somente os tokens futuros.
x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10

with torch.no_grad():
    output1 = attention(x1)
    output2 = attention(x2)

# Os primeiros cinco tokens não podem sofrer influência
# das alterações realizadas nos tokens futuros.
difference = (output1[:, :5, :] - output2[:, :5, :]).abs()

print("Diferença máxima:", difference.max().item())
print("Diferença média:", difference.mean().item())

assert torch.allclose(
    output1[:, :5, :],
    output2[:, :5, :],
    atol=1e-6,
    rtol=1e-5,
), "Falha: tokens futuros influenciaram posições anteriores!"

print("TESTE DE CAUSALIDADE APROVADO")
