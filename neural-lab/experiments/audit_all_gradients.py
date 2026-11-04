import torch
import torch.nn.functional as F

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()
model = LuciGPT(config)

tokens = torch.randint(0, config.vocab_size, (2, 32))
targets = torch.randint(0, config.vocab_size, (2, 32))

logits = model(tokens)

loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    targets.reshape(-1),
)

loss.backward()

print("LOSS:", loss.item())

print("\n========== GRADIENT AUDIT ==========")

for name, parameter in model.named_parameters():

    if parameter.grad is None:
        print(f"{name}: MISSING")
        continue

    grad = parameter.grad.detach()

    print(
        f"{name:<65} "
        f"norm={grad.norm().item():.8f} "
        f"mean={grad.mean().item():.8e} "
        f"max={grad.abs().max().item():.8e}"
    )

    assert torch.isfinite(grad).all(), (
        f"Non-finite gradient: {name}"
    )

print("\nAUDITORIA COMPLETA FINALIZADA")
