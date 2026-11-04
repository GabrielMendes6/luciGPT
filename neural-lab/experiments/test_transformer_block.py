import torch

from architecture.config import ModelConfig
from architecture.embeddings import InputEmbedding
from architecture.transformer_block import TransformerBlock


config = ModelConfig()

embedding = InputEmbedding(config)
block = TransformerBlock(config)

tokens = torch.tensor([
    [12, 45, 103, 8],
    [7, 21, 90, 14],
])

x = embedding(tokens)
output = block(x)

print("===== LUCI TRANSFORMER BLOCK =====")
print("Entrada:", x.shape)
print("Saída:", output.shape)

parameters = sum(
    p.numel()
    for p in block.parameters()
)

print("Parâmetros do bloco:", parameters)

loss = output.mean()
loss.backward()

print("\n===== BACKPROPAGATION =====")

for name, parameter in block.named_parameters():
    print(f"{name}: {parameter.grad is not None}")

print(
    "\nTodos os gradientes:",
    all(
        parameter.grad is not None
        for parameter in block.parameters()
    )
)