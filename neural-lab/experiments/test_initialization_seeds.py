import gc
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


config = ModelConfig()

torch.manual_seed(7)
tokens = torch.randint(0, config.vocab_size, (2, 32))

seeds = [42, 123, 2026]


def evaluate(model):
    measurements = []

    def capture(module, inputs, output):
        measurements.append(
            output.detach().pow(2).mean().sqrt().item()
        )

    handles = [
        block.register_forward_hook(capture)
        for block in model.transformer_blocks
    ]

    model.eval()

    with torch.no_grad():
        logits = model(tokens)

    for handle in handles:
        handle.remove()

    return measurements, logits.pow(2).mean().sqrt().item()


results = {"baseline": [], "balanced": []}

for seed in seeds:
    # Baseline
    torch.manual_seed(seed)
    baseline = LuciGPT(config)

    baseline_rms, baseline_logits = evaluate(baseline)

    results["baseline"].append({
        "blocks": baseline_rms,
        "logits": baseline_logits,
    })

    del baseline
    gc.collect()

    # Residual balanceado
    torch.manual_seed(seed)
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

    balanced_rms, balanced_logits = evaluate(balanced)

    results["balanced"].append({
        "blocks": balanced_rms,
        "logits": balanced_logits,
    })

    del balanced
    gc.collect()

    print(f"\nSEED {seed}")
    print(
        f"Baseline final RMS: {baseline_rms[-1]:.6f} | "
        f"Balanced final RMS: {balanced_rms[-1]:.6f}"
    )
    print(
        f"Baseline logits RMS: {baseline_logits:.6f} | "
        f"Balanced logits RMS: {balanced_logits:.6f}"
    )


print("\n========== MÉDIAS DOS EXPERIMENTOS ==========")

for variant in ("baseline", "balanced"):
    final_values = [
        result["blocks"][-1]
        for result in results[variant]
    ]

    logit_values = [
        result["logits"]
        for result in results[variant]
    ]

    print(f"\n{variant.upper()}")
    print(f"Final RMS médio: {sum(final_values) / len(final_values):.6f}")
    print(f"Logits RMS médio: {sum(logit_values) / len(logit_values):.6f}")
