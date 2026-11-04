
import statistics
import time
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT

torch.manual_seed(42)

PROMPT_LENGTH = 128
DECODE_STEPS = 64
REPETITIONS = 5

config = ModelConfig(
    vocab_size=16384,
    embedding_dim=768,
    max_seq_len=4096,
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
    ffn_dim=3072,
)

model = LuciGPT(config).eval()

prompt = torch.randint(
    0,
    config.vocab_size,
    (1, PROMPT_LENGTH),
)


@torch.inference_mode()
def run_generation(prefill_threads, decode_threads):
    torch.set_num_threads(prefill_threads)

    start = time.perf_counter()

    logits, cache = model(prompt, use_cache=True)

    prefill_elapsed = time.perf_counter() - start

    torch.set_num_threads(decode_threads)

    decode_start = time.perf_counter()

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

    decode_elapsed = time.perf_counter() - decode_start

    total_elapsed = time.perf_counter() - start

    return {
        "prefill": prefill_elapsed,
        "decode": decode_elapsed,
        "total": total_elapsed,
        "cache": cache.length,
    }


configurations = [
    ("2 -> 2", 2, 2),
    ("8 -> 8", 8, 8),
    ("8 -> 2", 8, 2),
]

results = []

print("===== LUCI HYBRID THREAD BENCHMARK =====")

for name, prefill_threads, decode_threads in configurations:
    # Aquecimento
    run_generation(prefill_threads, decode_threads)

    prefill_times = []
    decode_times = []
    total_times = []

    for _ in range(REPETITIONS):
        result = run_generation(
            prefill_threads,
            decode_threads,
        )

        prefill_times.append(result["prefill"])
        decode_times.append(result["decode"])
        total_times.append(result["total"])

    prefill_median = statistics.median(prefill_times)
    decode_median = statistics.median(decode_times)
    total_median = statistics.median(total_times)

    results.append({
        "name": name,
        "prefill": prefill_median,
        "decode": decode_median,
        "total": total_median,
    })

    print(f"\nConfiguração: {name}")
    print(f"Prefill: {prefill_median:.4f}s")
    print(f"Decode:  {decode_median:.4f}s")
    print(f"Total:   {total_median:.4f}s")
    print(f"Throughput total: {(PROMPT_LENGTH + DECODE_STEPS) / total_median:.2f} tok/s")

best = min(results, key=lambda item: item["total"])

print("\n===== MELHOR CONFIGURAÇÃO =====")
print(f"Configuração: {best['name']}")
print(f"Tempo total: {best['total']:.4f}s")
