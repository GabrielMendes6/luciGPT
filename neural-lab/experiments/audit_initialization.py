import torch
import torch.nn as nn

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

config = ModelConfig()
model = LuciGPT(config)


def audit_module(name, weight):
    print(f"\n{name}")
    print(f"  Shape: {tuple(weight.shape)}")
    print(f"  Mean: {weight.mean().item():.6f}")
    print(f"  Std: {weight.std().item():.6f}")
    print(f"  Min: {weight.min().item():.6f}")
    print(f"  Max: {weight.max().item():.6f}")


print("========== EMBEDDINGS ==========")

audit_module(
    "Token Embedding",
    model.input_embedding.token_embedding.embedding.weight
)

audit_module(
    "Position Embedding",
    model.input_embedding.position_embedding.embedding.weight
)


print("\n========== ATTENTION ==========")

for index, block in enumerate(model.transformer_blocks):

    attention = block.attention

    print(f"\nTransformer Block {index + 1}")

    audit_module("Query", attention.query.weight)
    audit_module("Key", attention.key.weight)
    audit_module("Value", attention.value.weight)
    audit_module(
        "Output Projection",
        attention.output_projection.weight
    )


print("\n========== FEED FORWARD ==========")

audit_module(
    "FFN Layer 1",
    model.transformer_blocks[0].feed_forward.network[0].weight
)

audit_module(
    "FFN Layer 2",
    model.transformer_blocks[0].feed_forward.network[2].weight
)


print("\n========== LAYER NORMALIZATION ==========")

for name, module in model.named_modules():

    if isinstance(module, nn.LayerNorm):

        print(
            f"{name}: "
            f"weight_mean={module.weight.mean().item():.6f}, "
            f"bias_mean={module.bias.mean().item():.6f}"
        )


print("\n========== WEIGHT TYING ==========")

embedding = model.input_embedding.token_embedding.embedding.weight
lm_head = model.lm_head.weight

print("Mesmo objeto:", embedding is lm_head)
print("Mesmo endereço:", embedding.data_ptr() == lm_head.data_ptr())


print("\n========== MODEL PARAMETERS ==========")

total = sum(parameter.numel() for parameter in model.parameters())

trainable = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total:", f"{total:,}")
print("Treináveis:", f"{trainable:,}")
print("Vocab size:", config.vocab_size)
print("Layers:", config.num_layers)
print("Embedding dim:", config.embedding_dim)
print("Attention heads:", config.num_heads)
print("KV heads:", config.num_kv_heads)
print("FFN dim:", config.ffn_dim)
