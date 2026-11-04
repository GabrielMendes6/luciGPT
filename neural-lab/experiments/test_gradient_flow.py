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

activations = []


def capture_input(module, inputs):
    tensor = inputs[0]
    tensor.retain_grad()
    activations.append({
        "stage": "input",
        "tensor": tensor,
        "block": len(activations) // 2 + 1,
    })


def capture_output(module, inputs, output):
    output.retain_grad()
    activations.append({
        "stage": "output",
        "tensor": output,
        "block": len(activations) // 2 + 1,
    })


for block in model.transformer_blocks:
    block.register_forward_pre_hook(capture_input)
    block.register_forward_hook(capture_output)


logits = model(tokens)

loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    targets.reshape(-1),
)

loss.backward()

print(f"LOSS: {loss.item():.6f}")
print("\n========== GRADIENT FLOW ==========")

for index in range(config.num_layers):
    input_data = activations[index * 2]
    output_data = activations[index * 2 + 1]

    x_in = input_data["tensor"]
    x_out = output_data["tensor"]

    input_grad = x_in.grad.norm().item()
    output_grad = x_out.grad.norm().item()

    input_rms = x_in.detach().pow(2).mean().sqrt().item()
    output_rms = x_out.detach().pow(2).mean().sqrt().item()

    input_grad_rms = input_grad / (x_in.numel() ** 0.5)
    output_grad_rms = output_grad / (x_out.numel() ** 0.5)

    print(f"\nBLOCK {index + 1:02d}")
    print(f"Input activation RMS:  {input_rms:.6f}")
    print(f"Output activation RMS: {output_rms:.6f}")
    print(f"Input gradient RMS:    {input_grad_rms:.8f}")
    print(f"Output gradient RMS:   {output_grad_rms:.8f}")
