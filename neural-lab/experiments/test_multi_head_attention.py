
import torch

from architecture.config import ModelConfig
from architecture.multi_head_attention import MultiHeadCausalSelfAttention


def main():
    config = ModelConfig()

    attention = MultiHeadCausalSelfAttention(config)

    batch_size = 2
    sequence_length = 32

    x = torch.randn(
        batch_size,
        sequence_length,
        config.embedding_dim,
        requires_grad=True,
    )

    output = attention(x)

    print("Entrada:", x.shape)
    print("Saída:", output.shape)

    assert output.shape == (
        batch_size,
        sequence_length,
        config.embedding_dim,
    )

    # Verifica o fluxo de gradientes.
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert attention.query.weight.grad is not None
    assert attention.key.weight.grad is not None
    assert attention.value.weight.grad is not None

    print("Gradiente da entrada:", x.grad.shape)
    print("Gradiente Q:", attention.query.weight.grad.shape)
    print("Gradiente K:", attention.key.weight.grad.shape)
    print("Gradiente V:", attention.value.weight.grad.shape)

    # Verifica o limite de contexto.
    try:
        invalid_input = torch.randn(
            1,
            config.max_seq_len + 1,
            config.embedding_dim,
        )

        attention(invalid_input)

        raise AssertionError("O limite de contexto não foi respeitado")

    except ValueError:
        print("Limite de contexto: validado")

    print("GQA: TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
