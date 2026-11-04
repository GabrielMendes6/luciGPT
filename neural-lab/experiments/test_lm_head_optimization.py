
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


def main():
    torch.manual_seed(42)

    config = ModelConfig()
    model = LuciGPT(config)
    model.eval()

    batch_size = 2
    sequence_length = 128

    token_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
    )

    with torch.no_grad():
        full_logits = model(
            token_ids,
            logits_to_keep=None,
        )

        last_logits = model(
            token_ids,
            logits_to_keep=1,
        )

        last_five_logits = model(
            token_ids,
            logits_to_keep=5,
        )

    expected_last = full_logits[:, -1:, :]
    expected_last_five = full_logits[:, -5:, :]

    print("===== LM HEAD OPTIMIZATION =====")

    print("Full logits:", tuple(full_logits.shape))
    print("Last logits:", tuple(last_logits.shape))
    print("Last five:", tuple(last_five_logits.shape))

    max_diff = (
        expected_last - last_logits
    ).abs().max().item()

    mean_diff = (
        expected_last - last_logits
    ).abs().mean().item()

    print("\nMaximum difference:", max_diff)
    print("Mean difference:", mean_diff)

    assert torch.allclose(
        expected_last,
        last_logits,
        rtol=1e-5,
        atol=1e-6,
    ), "Logits da última posição divergiram!"

    assert torch.allclose(
        expected_last_five,
        last_five_logits,
        rtol=1e-5,
        atol=1e-6,
    ), "Logits das últimas cinco posições divergiram!"

    assert full_logits.shape == (
        batch_size,
        sequence_length,
        config.vocab_size,
    )

    assert last_logits.shape == (
        batch_size,
        1,
        config.vocab_size,
    )

    assert last_five_logits.shape == (
        batch_size,
        5,
        config.vocab_size,
    )

    print("\nRESULTADO: TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
