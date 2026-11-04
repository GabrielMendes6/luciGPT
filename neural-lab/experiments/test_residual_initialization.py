import torch
import torch.nn.functional as F

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()

tokens = torch.randint(0, config.vocab_size, (2, 32))
targets = torch.randint(0, config.vocab_size, (2, 32))


def evaluate_model(model, label):
    model.train()
    branch_outputs = {}

    def capture(name):
        def hook(module, inputs, output):
            output.retain_grad()
            branch_outputs[name] = output
        return hook

    handles = []

    for index, block in enumerate(model.transformer_blocks):
        handles.append(
            block.attention.register_forward_hook(
                capture(f"block_{index + 1}_attention")
            )
        )
        handles.append(
            block.feed_forward.register_forward_hook(
                capture(f"block_{index + 1}_ffn")
            )
        )

    logits = model(tokens)

    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    loss.backward()

    print(f"\n========== {label} ==========")
    print(f"LOSS: {loss.item():.6f}")

    for index, block in enumerate(model.transformer_blocks):
        attention = branch_outputs[f"block_{index + 1}_attention"]
        ffn = branch_outputs[f"block_{index + 1}_ffn"]

        attention_rms = attention.detach().pow(2).mean().sqrt().item()
        ffn_rms = ffn.detach().pow(2).mean().sqrt().item()

        attention_grad = attention.grad.detach().norm().item()
        ffn_grad = ffn.grad.detach().norm().item()

        print(
            f"BLOCK {index + 1:02d} | "
            f"Attention RMS: {attention_rms:.6f} | "
            f"FFN RMS: {ffn_rms:.6f} | "
            f"Attention Grad: {attention_grad:.6f} | "
            f"FFN Grad: {ffn_grad:.6f}"
        )

    for handle in handles:
        handle.remove()

    return loss.item()


# A: Baseline
torch.manual_seed(42)
baseline = LuciGPT(config)

baseline_loss = evaluate_model(baseline, "BASELINE")


# B: Residual balanceado
torch.manual_seed(42)
balanced = LuciGPT(config)

residual_std = 0.02 / (2 * config.num_layers) ** 0.5

with torch.no_grad():
    for block in balanced.transformer_blocks:
        output_projection = block.feed_forward.network[2]

        torch.nn.init.normal_(
            output_projection.weight,
            mean=0.0,
            std=residual_std,
        )

        if output_projection.bias is not None:
            torch.nn.init.zeros_(output_projection.bias)


balanced_loss = evaluate_model(balanced, "RESIDUAL BALANCEADO")


print("\n========== COMPARAÇÃO FINAL ==========")
print(f"Baseline loss: {baseline_loss:.6f}")
print(f"Balanced loss: {balanced_loss:.6f}")
print(f"Residual std: {residual_std:.8f}")
