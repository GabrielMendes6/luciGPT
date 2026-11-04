
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    config = ModelConfig(
        vocab_size=848,
        embedding_dim=128,
        max_seq_len=128,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_dim=512,
    )

    model = LuciGPT(config).eval()

    batch_size = 2
    sequence_length = 12
    prefill_length = 8

    token_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
    )

    with torch.inference_mode():

        # Referência: sequência completa sem cache.
        reference_logits = model(token_ids)

        # Prefill: primeiros 8 tokens.
        prefill_logits, cache = model(
            token_ids[:, :prefill_length],
            use_cache=True,
        )

        # Incremental: últimos 4 tokens.
        incremental_logits, cache = model(
            token_ids[:, prefill_length:],
            kv_cache=cache,
            use_cache=True,
        )

        combined_logits = torch.cat(
            [prefill_logits, incremental_logits],
            dim=1,
        )

    difference = (reference_logits - combined_logits).abs()

    print("===== MODEL CACHE EQUIVALENCE =====")
    print("Reference shape:", reference_logits.shape)
    print("Incremental shape:", combined_logits.shape)
    print("Cache final length:", cache.length)
    print("Maximum difference:", difference.max().item())
    print("Mean difference:", difference.mean().item())

    equivalent = torch.allclose(
        reference_logits,
        combined_logits,
        atol=1e-5,
        rtol=1e-4,
    )

    print("Equivalent:", equivalent)

    if not equivalent:
        raise AssertionError(
            "O modelo com cache divergiu do forward convencional."
        )


if __name__ == "__main__":
    main()

