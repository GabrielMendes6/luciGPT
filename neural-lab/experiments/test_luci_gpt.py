import torch
import torch.nn.functional as F

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()

model = LuciGPT(config)

tokens = torch.tensor([
    [12, 45, 103, 8],
    [7, 21, 90, 14],
])

# Próximo token esperado em cada posição.
targets = torch.tensor([
    [45, 103, 8, 20],
    [21, 90, 14, 32],
])

logits = model(tokens)

print("===== LUCI GPT =====")
print("Entrada:", tokens.shape)
print("Logits:", logits.shape)

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Parâmetros treináveis:", parameters)

# Calcula o erro de previsão.
loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    targets.reshape(-1),
)

print("\n===== TREINAMENTO =====")
print("Loss:", loss.item())

loss.backward()

print("Backpropagation:", all(
    p.grad is not None
    for p in model.parameters()
    if p.requires_grad
))

print("\n===== WEIGHT TYING =====")

tied = (
    model.lm_head.weight.data_ptr()
    == model.input_embedding.token_embedding.embedding.weight.data_ptr()
)

print("Pesos compartilhados:", tied)