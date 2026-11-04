import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()
model = LuciGPT(config)
model.eval()

tokens = torch.randint(0, config.vocab_size, (2, 32))

measurements = []


def rms(tensor):
    return tensor.detach().pow(2).mean().sqrt().item()


def capture_norm(name, block_index):
    def hook(module, inputs, output):
        measurements.append({
            "block": block_index,
            "name": name,
            "rms": rms(output),
            "mean": output.detach().mean().item(),
            "std": output.detach().std().item(),
        })

    return hook


handles = []

for index, block in enumerate(model.transformer_blocks):
    handles.append(
        block.norm1.register_forward_hook(
            capture_norm("Attention Pre-Norm", index + 1)
        )
    )

    handles.append(
        block.norm2.register_forward_hook(
            capture_norm("FFN Pre-Norm", index + 1)
        )
    )

handles.append(
    model.final_norm.register_forward_hook(
        capture_norm("Final Norm", 0)
    )
)

with torch.no_grad():
    logits = model(tokens)

for handle in handles:
    handle.remove()

print("========== NORMALIZATION AUDIT ==========")

for item in measurements:
    print(
        f"BLOCK {item['block']:02d} | "
        f"{item['name']:<20} | "
        f"RMS: {item['rms']:.6f} | "
        f"Mean: {item['mean']:.6f} | "
        f"Std: {item['std']:.6f}"
    )

print(
    f"\nLOGITS RMS: {rms(logits):.6f}"
)
