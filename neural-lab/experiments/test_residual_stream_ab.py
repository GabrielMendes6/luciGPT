import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()

tokens = torch.randint(0, config.vocab_size, (2, 32))


def evaluate(model, label):
    measurements = []

    def capture_input(module, inputs):
        x = inputs[0]
        measurements.append({
            "input": x.detach().pow(2).mean().sqrt().item()
        })

    def capture_output(module, inputs, output):
        x = output
        measurements[-1]["output"] = (
            x.detach().pow(2).mean().sqrt().item()
        )

    handles = []

    for block in model.transformer_blocks:
        handles.append(
            block.register_forward_pre_hook(capture_input)
        )
        handles.append(
            block.register_forward_hook(capture_output)
        )

    model.eval()

    with torch.no_grad():
        logits = model(tokens)

    for handle in handles:
        handle.remove()

    print(f"\n========== {label} ==========")

    for index, item in enumerate(measurements):
        print(
            f"BLOCK {index + 1:02d} | "
            f"Input RMS: {item['input']:.6f} | "
            f"Output RMS: {item['output']:.6f}"
        )

    print(
        f"FINAL LOGITS RMS: "
        f"{logits.pow(2).mean().sqrt().item():.6f}"
    )


# A: Baseline
torch.manual_seed(42)
baseline = LuciGPT(config)

evaluate(baseline, "BASELINE")


# B: Residual balanceado
torch.manual_seed(42)
balanced = LuciGPT(config)

residual_std = 0.02 / (2 * config.num_layers) ** 0.5

with torch.no_grad():
    for block in balanced.transformer_blocks:
        projection = block.feed_forward.network[2]

        torch.nn.init.normal_(
            projection.weight,
            mean=0.0,
            std=residual_std,
        )

        if projection.bias is not None:
            torch.nn.init.zeros_(projection.bias)

evaluate(balanced, "RESIDUAL BALANCEADO")

