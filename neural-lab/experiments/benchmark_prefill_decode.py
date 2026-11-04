
import statistics
import time
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)

THREAD_COUNTS = [1, 2, 4, 6, 8]

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
def prefill():
    return model(prompt, use_cache=True)


@torch.inference_mode()
def decode(logits, cache):
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)

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


def measure_prefill():
    start = time.perf_counter()
    logits, cache = prefill()
    elapsed = time.perf_counter() - start

    return elapsed, logits, cache


def measure_decode():
    # O prefill acontece fora do cronômetro.
    logits, cache = prefill()

    start = time.perf_counter()
    final_length = decode(logits, cache)
    elapsed = time.perf_counter() - start

    return elapsed, final_length


print("===== LUCI PREFILL / DECODE BENCHMARK =====")
print(f"Prompt: {PROMPT_LENGTH}")
print(f"Decode: {DECODE_STEPS}")
print(f"Repetições: {REPETITIONS}")
print(f"Threads: {THREAD_COUNTS}")

results = []

for thread_count in THREAD_COUNTS:
    torch.set_num_threads(thread_count)

    # Aquecimento
    prefill()
    measure_decode()

    prefill_times = []
    decode_times = []

    for _ in range(REPETITIONS):
        elapsed, _, _ = measure_prefill()
        prefill_times.append(elapsed)

    for _ in range(REPETITIONS):
        elapsed, final_length = measure_decode()
        decode_times.append(elapsed)

    prefill_median = statistics.median(prefill_times)
    decode_median = statistics.median(decode_times)

    prefill_throughput = PROMPT_LENGTH / prefill_median
    decode_throughput = DECODE_STEPS / decode_median

    result = {
        "threads": thread_count,
        "prefill_median": prefill_median,
        "prefill_throughput": prefill_throughput,
        "decode_median": decode_median,
        "decode_throughput": decode_throughput,
        "cache_length": final_length,
    }

    results.append(result)

    print(f"\nThreads: {thread_count}")
    print(
        f"  Prefill: {prefill_median:.4f}s | "
        f"{prefill_throughput:.2f} tokens/s"
    )
    print(
        f"  Decode:  {decode_median:.4f}s | "
        f"{decode_throughput:.2f} tokens/s"
    )
    print(f"  Cache:   {final_length}")

print("\n===== MELHORES CONFIGURAÇÕES =====")

best_prefill = max(
    results,
    key=lambda item: item["prefill_throughput"],
)

best_decode = max(
    results,
    key=lambda item: item["decode_throughput"],
)

print(
    f"Prefill: {best_prefill['threads']} threads | "
    f"{best_prefill['prefill_throughput']:.2f} tok/s"
)

print(
    f"Decode:  {best_decode['threads']} threads | "
    f"{best_decode['decode_throughput']:.2f} tok/s"
)

