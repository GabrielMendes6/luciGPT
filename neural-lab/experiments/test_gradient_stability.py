import torch
import torch.nn.functional as F

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()
model = LuciGPT(config)

model.train()

tokens = torch.randint(
    0,
    config.vocab_size,
    (2, 32),
)

targets = torch.randint(
    0,
    config.vocab_size,
    (2, 32),
)

logits = model(tokens)

loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    targets.reshape(-1),
)

loss.backward()

print("========== GRADIENT STABILITY ==========")
print("Loss:", loss.item())

print("\n========== TRANSFORMER BLOCKS ==========")

for index, block in enumerate(model.transformer_blocks):

    attention_grad = block.attention.query.weight.grad.norm().item()
    ffn_grad = block.feed_forward.network[0].weight.grad.norm().item()

    print(
        f"Block {index + 1:02d} | "
        f"Attention: {attention_grad:.8f} | "
        f"FFN: {ffn_grad:.8f}"
    )

print("\n========== GLOBAL GRADIENTS ==========")

total_norm = 0.0
missing_gradients = []
non_finite_gradients = []

for name, parameter in model.named_parameters():

    if parameter.grad is None:
        missing_gradients.append(name)
        continue

    if not torch.isfinite(parameter.grad).all():
        non_finite_gradients.append(name)

    total_norm += parameter.grad.detach().norm(2).item() ** 2

total_norm = total_norm ** 0.5

print("Global gradient norm:", total_norm)
print("Missing gradients:", missing_gradients)
print("Non-finite gradients:", non_finite_gradients)

assert not missing_gradients
assert not non_finite_gradients
assert torch.isfinite(torch.tensor(total_norm))

print("\nGRADIENT STABILITY TEST COMPLETED")

