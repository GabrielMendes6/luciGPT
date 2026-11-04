import torch

from architecture.config import ModelConfig
from architecture.feed_forward import FeedForwardNetwork


config = ModelConfig()

ffn = FeedForwardNetwork(config)

x = torch.randn(2, 4, config.embedding_dim)

output = ffn(x)

print("===== LUCI FEED FORWARD NETWORK =====")
print("Entrada:", x.shape)
print("Saída:", output.shape)

print(
    "Dimensão intermediária:",
    config.embedding_dim * 4
)

parameters = sum(
    p.numel()
    for p in ffn.parameters()
)

print("Parâmetros treináveis:", parameters)

loss = output.mean()
loss.backward()

print("\n===== BACKPROPAGATION =====")

for name, parameter in ffn.named_parameters():
    print(
        f"{name}:",
        parameter.grad is not None
    )