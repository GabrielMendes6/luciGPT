import torch
import torch.nn.functional as F

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()
model = LuciGPT(config)
model.train()

tokens = torch.randint(0, config.vocab_size, (2, 32))
targets = torch.randint(0, config.vocab_size, (2, 32))

branch_data = []


def capture_branch(name, block_index):
    def hook(module, inputs, output):
        output.retain_grad()

        branch_data.append({
            "block": block_index,
            "name": name,
            "output": output,
        })

    return hook


for index, block in enumerate(model.transformer_blocks):
    block.attention.register_forward_hook(
        capture_branch("Attention", index + 1)
    )

    block.feed_forward.register_forward_hook(
        capture_branch("FFN", index + 1)
    )


logits = model(tokens)

loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    targets.reshape(-1),
)

loss.backward()

print(f"LOSS: {loss.item():.6f}")
print("\n========== RESIDUAL BRANCH AUDIT ==========")

for item in branch_data:
    output = item["output"]
    gradient = output.grad

    output_rms = output.detach().pow(2).mean().sqrt().item()
    gradient_rms = gradient.detach().pow(2).mean().sqrt().item()

    print(
        f"BLOCK {item['block']:02d} | "
        f"{item['name']:<10} | "
        f"Output RMS: {output_rms:.8f} | "
        f"Gradient RMS: {gradient_rms:.8f}"
    )


print("\n========== RESIDUAL PROJECTION INITIALIZATION ==========")

for index, block in enumerate(model.transformer_blocks):
    attention_weight = block.attention.output_projection.weight
    ffn_weight = block.feed_forward.network[2].weight

    print(
        f"BLOCK {index + 1:02d} | "
        f"Attention std: {attention_weight.detach().std().item():.8f} | "
        f"FFN std: {ffn_weight.detach().std().item():.8f}"
    )

