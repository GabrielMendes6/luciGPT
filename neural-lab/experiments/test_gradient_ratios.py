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

print("\n========== GRADIENT / WEIGHT RATIOS ==========")

for index, block in enumerate(model.transformer_blocks):

    print(f"\nBLOCK {index + 1:02d}")

    modules = {
        "Attention Query": block.attention.query,
        "Attention Key": block.attention.key,
        "Attention Value": block.attention.value,
        "Attention Output": block.attention.output_projection,
        "FFN Input": block.feed_forward.network[0],
        "FFN Output": block.feed_forward.network[2],
    }

    for name, module in modules.items():

        weight_norm = module.weight.detach().norm().item()
        gradient_norm = module.weight.grad.detach().norm().item()

        ratio = gradient_norm / max(weight_norm, 1e-12)

        print(
            f"{name:<20} "
            f"Weight: {weight_norm:.6f} | "
            f"Gradient: {gradient_norm:.6f} | "
            f"Ratio: {ratio:.6f}"
        )
