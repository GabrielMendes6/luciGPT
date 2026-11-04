
import statistics
import time

import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

THREAD_COUNTS = [1, 2, 4, 6, 8, 10, 12]

PROMPT_LENGTH = 128
DECODE_STEPS = 64
REPETITIONS = 3

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
def run_decode():

    logits, cache = model(
        prompt,
        use_cache=True,
    )

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    for _ in range(DECODE_STEPS):

        logits, cache = model(
            next_token,
            kv_cache=cache,
            use_cache=True,
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

    return cache.length


print("===== LUCI CPU THREAD BENCHMARK =====")
print(f"Prompt: {PROMPT_LENGTH}")
print(f"Decode: {DECODE_STEPS}")
print(f"Repetições: {REPETITIONS}")

results = []

for thread_count in THREAD_COUNTS:

    torch.set_num_threads(thread_count)

    # Aquecimento.
    run_decode()

    latencies = []

    for _ in range(REPETITIONS):

        start = time.perf_counter()

        final_length = run_decode()

        elapsed = time.perf_counter() - start

        latencies.append(elapsed)

    median_latency = statistics.median(latencies)
    mean_latency = statistics.mean(latencies)

    throughput = DECODE_STEPS / median_latency

    results.append({
        "threads": thread_count,
        "median": median_latency,
        "mean": mean_latency,
        "throughput": throughput,
    })

    print(
        f"Threads: {thread_count:2d} | "
        f"Mediana: {median_latency:.4f}s | "
        f"Média: {mean_latency:.4f}s | "
        f"Throughput: {throughput:.2f} tok/s | "
        f"Cache: {final_length}"
    )

best = max(
    results,
    key=lambda result: result["throughput"],
)

print("\n===== MELHOR CONFIGURAÇÃO =====")
print(f"Threads: {best['threads']}")
print(f"Mediana: {best['median']:.4f}s")
print(f"Throughput: {best['throughput']:.2f} tokens/s")
