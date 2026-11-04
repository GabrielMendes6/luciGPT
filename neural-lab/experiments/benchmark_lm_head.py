
import statistics
import time

import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


# ============================================================
# CONFIGURAÇÃO
# ============================================================

torch.manual_seed(42)
torch.set_num_threads(2)

PROMPT_LENGTH = 128
DECODE_STEPS = 64
REPETITIONS = 7
WARMUP_RUNS = 2

CONFIG = ModelConfig()


# ============================================================
# MODELO E PROMPT
# ============================================================

model = LuciGPT(CONFIG).eval()

prompt = torch.randint(
    0,
    CONFIG.vocab_size,
    (1, PROMPT_LENGTH),
)


# ============================================================
# GERAÇÃO
# ============================================================

@torch.inference_mode()
def run_generation(
    target_model,
    input_prompt,
    logits_to_keep,
):
    logits, cache = target_model(
        input_prompt,
        use_cache=True,
        logits_to_keep=logits_to_keep,
    )

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
    ).unsqueeze(1)

    generated_tokens = [next_token]

    for _ in range(DECODE_STEPS):

        logits, cache = target_model(
            next_token,
            kv_cache=cache,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        ).unsqueeze(1)

        generated_tokens.append(next_token)

    generated = torch.cat(
        generated_tokens,
        dim=1,
    )

    return generated, cache.length


# ============================================================
# EQUIVALÊNCIA
# ============================================================

print("===== LUCI LM HEAD A/B BENCHMARK =====")

print(f"PyTorch: {torch.__version__}")
print(f"Threads: {torch.get_num_threads()}")
print("Device: cpu")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Prompt: {PROMPT_LENGTH}")
print(f"Decode: {DECODE_STEPS}")
print(f"Repetitions: {REPETITIONS}")
print(f"Warmups: {WARMUP_RUNS}")


baseline_tokens, baseline_cache_length = run_generation(
    model,
    prompt,
    logits_to_keep=None,
)

optimized_tokens, optimized_cache_length = run_generation(
    model,
    prompt,
    logits_to_keep=1,
)

equivalent = torch.equal(
    baseline_tokens,
    optimized_tokens,
)

print("\n===== EQUIVALÊNCIA =====")

print(
    "Baseline tokens:",
    tuple(baseline_tokens.shape),
)

print(
    "Optimized tokens:",
    tuple(optimized_tokens.shape),
)

print(
    "Tokens identical:",
    equivalent,
)

print(
    "Baseline cache:",
    baseline_cache_length,
)

print(
    "Optimized cache:",
    optimized_cache_length,
)

assert equivalent, (
    "As gerações divergiram entre baseline e otimizado."
)

assert baseline_cache_length == optimized_cache_length

assert baseline_cache_length == (
    PROMPT_LENGTH + DECODE_STEPS
)


# ============================================================
# BENCHMARK
# ============================================================

variants = {
    "BASELINE": None,
    "OPTIMIZED": 1,
}

results = {
    name: []
    for name in variants
}

# Aquecimento das duas variantes.
for _ in range(WARMUP_RUNS):

    for name, keep in variants.items():
        run_generation(
            model,
            prompt,
            logits_to_keep=keep,
        )


# Execuções alternadas para reduzir efeitos de carga.
for repetition in range(REPETITIONS):

    order = (
        ["BASELINE", "OPTIMIZED"]
        if repetition % 2 == 0
        else ["OPTIMIZED", "BASELINE"]
    )

    for name in order:

        start = time.perf_counter()

        generated, cache_length = run_generation(
            model,
            prompt,
            logits_to_keep=variants[name],
        )

        elapsed = time.perf_counter() - start

        assert cache_length == baseline_cache_length
        assert torch.equal(generated, baseline_tokens)

        results[name].append(elapsed)


# ============================================================
# RESULTADOS
# ============================================================

print("\n===== RESULTADOS =====")

medians = {}

for name, latencies in results.items():

    median = statistics.median(latencies)
    mean = statistics.mean(latencies)

    # Mantém a mesma convenção de throughput do benchmark
    # anterior: 64 operações de decode por tempo total.
    throughput = DECODE_STEPS / median

    medians[name] = median

    print(f"\n{name}")

    print(
        "Latências:",
        [round(value, 4) for value in latencies],
    )

    print(f"Mediana: {median:.4f}s")
    print(f"Média: {mean:.4f}s")
    print(f"Throughput: {throughput:.2f} steps/s")


# ============================================================
# COMPARAÇÃO
# ============================================================

baseline_median = medians["BASELINE"]
optimized_median = medians["OPTIMIZED"]

speedup = baseline_median / optimized_median

latency_difference = (
    (optimized_median - baseline_median)
    / baseline_median
) * 100

print("\n===== COMPARAÇÃO A/B =====")

print(f"Baseline: {baseline_median:.4f}s")
print(f"Otimizado: {optimized_median:.4f}s")

print(f"Speedup: {speedup:.3f}x")

print(
    "Diferença de latência:",
    f"{latency_difference:+.2f}%",
)

print("\nBENCHMARK CONCLUÍDO")
