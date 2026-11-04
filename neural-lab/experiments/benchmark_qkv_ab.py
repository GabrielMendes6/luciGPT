
import copy
import statistics
import time

import torch
import torch.nn as nn

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)
torch.set_num_threads(2)


PROMPT_LENGTH = 128
DECODE_STEPS = 64
REPETITIONS = 7
WARMUP_RUNS = 2


class SeparateQKVProjection(nn.Module):
    """
    Executa Q, K e V em três operações lineares independentes.
    Mantém a mesma ordem de saída da projeção fundida.
    """

    def __init__(
        self,
        embedding_dim: int,
        q_dim: int,
        kv_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            q_dim,
            device=device,
            dtype=dtype,
        )

        self.key = nn.Linear(
            embedding_dim,
            kv_dim,
            device=device,
            dtype=dtype,
        )

        self.value = nn.Linear(
            embedding_dim,
            kv_dim,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        return torch.cat((q, k, v), dim=-1)


def create_separate_model(fused_model: LuciGPT) -> LuciGPT:
    separate_model = copy.deepcopy(fused_model)

    with torch.no_grad():

        for block in separate_model.transformer_blocks:

            attention = block.attention
            fused = attention.qkv_projection

            separate = SeparateQKVProjection(
                embedding_dim=attention.embedding_dim,
                q_dim=attention.q_dim,
                kv_dim=attention.kv_dim,
                device=fused.weight.device,
                dtype=fused.weight.dtype,
            )

            q_dim = attention.q_dim
            kv_dim = attention.kv_dim

            separate.query.weight.copy_(
                fused.weight[:q_dim]
            )

            separate.key.weight.copy_(
                fused.weight[q_dim:q_dim + kv_dim]
            )

            separate.value.weight.copy_(
                fused.weight[q_dim + kv_dim:q_dim + 2 * kv_dim]
            )

            if fused.bias is not None:
                separate.query.bias.copy_(
                    fused.bias[:q_dim]
                )

                separate.key.bias.copy_(
                    fused.bias[q_dim:q_dim + kv_dim]
                )

                separate.value.bias.copy_(
                    fused.bias[q_dim + kv_dim:q_dim + 2 * kv_dim]
                )

            attention.qkv_projection = separate

    return separate_model.eval()


@torch.inference_mode()
def run_generation(model, prompt):

    logits, cache = model(
        prompt,
        use_cache=True,
    )

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
    ).unsqueeze(1)

    for _ in range(DECODE_STEPS):

        logits, cache = model(
            next_token,
            kv_cache=cache,
            use_cache=True,
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        ).unsqueeze(1)

    return cache.length


def benchmark_pair(
    fused_model,
    separate_model,
    prompt,
):

    results = {
        "FUSED": [],
        "SEPARATE": [],
    }

    models = {
        "FUSED": fused_model,
        "SEPARATE": separate_model,
    }

    # Aquecimento de ambas as implementações.
    for _ in range(WARMUP_RUNS):
        run_generation(fused_model, prompt)
        run_generation(separate_model, prompt)

    # Alterna a ordem para reduzir efeitos de carga e temperatura.
    for repetition in range(REPETITIONS):

        order = (
            ["FUSED", "SEPARATE"]
            if repetition % 2 == 0
            else ["SEPARATE", "FUSED"]
        )

        for name in order:

            start = time.perf_counter()

            final_length = run_generation(
                models[name],
                prompt,
            )

            elapsed = time.perf_counter() - start

            results[name].append(elapsed)

    print("\n===== RESULTADOS =====")

    summaries = {}

    for name in ("FUSED", "SEPARATE"):

        latencies = results[name]
        median = statistics.median(latencies)
        mean = statistics.mean(latencies)

        throughput = (
            PROMPT_LENGTH + DECODE_STEPS
        ) / median

        summaries[name] = {
            "median": median,
            "mean": mean,
            "throughput": throughput,
        }

        print(f"\n{name}")
        print("Latências:", [round(x, 4) for x in latencies])
        print(f"Mediana: {median:.4f}s")
        print(f"Média: {mean:.4f}s")
        print(f"Throughput: {throughput:.2f} tok/s")
        print(f"Cache final: {final_length}")

    fused_time = summaries["FUSED"]["median"]
    separate_time = summaries["SEPARATE"]["median"]

    print("\n===== COMPARAÇÃO A/B =====")

    print(f"QKV fundido: {fused_time:.4f}s")
    print(f"Q/K/V separados: {separate_time:.4f}s")

    print(
        f"Speedup (fundido/separado): "
        f"{separate_time / fused_time:.3f}x"
    )

    difference = (
        (fused_time - separate_time)
        / separate_time
    ) * 100

    print(
        f"Diferença de latência do fundido: "
        f"{difference:+.2f}%"
    )


def main():

    config = ModelConfig(
        vocab_size=16384,
        embedding_dim=768,
        max_seq_len=4096,
        num_heads=12,
        num_kv_heads=4,
        num_layers=12,
        ffn_dim=3072,
    )

    fused_model = LuciGPT(config).eval()

    separate_model = create_separate_model(
        fused_model
    )

    prompt = torch.randint(
        0,
        config.vocab_size,
        (1, PROMPT_LENGTH),
    )

    print("===== LUCI QKV A/B BENCHMARK =====")
    print("PyTorch:", torch.__version__)
    print("Threads:", torch.get_num_threads())
    print("Device:", next(fused_model.parameters()).device)
    print("Parameters:", sum(p.numel() for p in fused_model.parameters()))
    print("Prompt:", PROMPT_LENGTH)
    print("Decode:", DECODE_STEPS)
    print("Repetitions:", REPETITIONS)
    print("Warmups:", WARMUP_RUNS)

    # Verifica equivalência antes de medir desempenho.
    with torch.inference_mode():

        fused_logits = fused_model(
            prompt,
            use_cache=False,
        )

        separate_logits = separate_model(
            prompt,
            use_cache=False,
        )

        max_difference = (
            fused_logits - separate_logits
        ).abs().max().item()

        mean_difference = (
            fused_logits - separate_logits
        ).abs().mean().item()

    print("\n===== EQUIVALÊNCIA =====")
    print(f"Maximum difference: {max_difference:.10e}")
    print(f"Mean difference: {mean_difference:.10e}")

    if not torch.allclose(
        fused_logits,
        separate_logits,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise RuntimeError(
            "As implementações não são numericamente equivalentes."
        )

    print("Equivalent: True")

    benchmark_pair(
        fused_model,
        separate_model,
        prompt,
    )


if __name__ == "__main__":
    main()
