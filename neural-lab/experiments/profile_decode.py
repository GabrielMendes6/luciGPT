
import time

import torch
from torch.profiler import profile, ProfilerActivity

from architecture.config import ModelConfig
from architecture.model import LuciGPT


torch.manual_seed(42)
torch.set_num_threads(8)

config = ModelConfig(
    vocab_size=16384,
    embedding_dim=768,
    max_seq_len=4096,
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
    ffn_dim=3072,
)

PROMPT_LENGTH = 128
DECODE_STEPS = 64
BENCHMARK_REPETITIONS = 5

model = LuciGPT(config)
model.eval()

prompt = torch.randint(
    0,
    config.vocab_size,
    (1, PROMPT_LENGTH),
)


def prefill():
    logits, cache = model(
        prompt,
        use_cache=True,
    )

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    return next_token, cache


def decode_steps(next_token, cache, steps):
    for _ in range(steps):
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

    return next_token, cache


def get_cache_memory(cache):
    total_bytes = 0

    for key, value in zip(
        cache.key_buffers,
        cache.value_buffers,
    ):
        total_bytes += key.numel() * key.element_size()
        total_bytes += value.numel() * value.element_size()

    return total_bytes


print("===== LUCI DECODE PROFILER =====")
print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"Threads: {torch.get_num_threads()}")
print(f"Prompt: {PROMPT_LENGTH}")
print(f"Decode steps: {DECODE_STEPS}")

with torch.inference_mode():

    # ==========================================
    # 1. AQUECIMENTO
    # ==========================================

    for _ in range(2):
        next_token, cache = prefill()
        decode_steps(next_token, cache, 3)

    # ==========================================
    # 2. BENCHMARK SEM PROFILER
    # ==========================================

    latencies = []
    cache = None

    for repetition in range(BENCHMARK_REPETITIONS):

        # Prefill fora da medição.
        next_token, cache = prefill()

        start = time.perf_counter()

        next_token, cache = decode_steps(
            next_token,
            cache,
            DECODE_STEPS,
        )

        elapsed = time.perf_counter() - start

        latencies.append(elapsed)

        print(
            f"Repetição {repetition + 1}: "
            f"{elapsed:.4f}s | "
            f"{DECODE_STEPS / elapsed:.2f} tokens/s"
        )

    mean_latency = sum(latencies) / len(latencies)
    throughput = DECODE_STEPS / mean_latency

    print("\n===== RESULTADO DO BENCHMARK =====")
    print(f"Latência média: {mean_latency:.4f}s")
    print(f"Tempo médio por token: {mean_latency / DECODE_STEPS * 1000:.2f}ms")
    print(f"Throughput médio: {throughput:.2f} tokens/s")

    # ==========================================
    # 3. PROFILER
    # ==========================================

    # Cache novo: não reutilizar o cache do benchmark.
    next_token, cache = prefill()

    cache_memory = get_cache_memory(cache)

    print("\n===== KV CACHE =====")
    print(f"Comprimento após prefill: {cache.length}")
    print(f"Memória reservada: {cache_memory / 1024**2:.2f} MiB")

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        next_token, cache = decode_steps(
            next_token,
            cache,
            DECODE_STEPS,
        )

print("\n===== OPERADORES POR TEMPO DE CPU =====")

print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=25,
    )
)

print("\n===== OPERADORES POR MEMÓRIA =====")

print(
    prof.key_averages().table(
        sort_by="self_cpu_memory_usage",
        row_limit=15,
    )
)

# ==========================================
# 4. ANÁLISE DAS CONCATENAÇÕES
# ==========================================

cat_events = [
    event
    for event in prof.key_averages()
    if event.key == "aten::cat"
]

print("\n===== ANÁLISE DE ATEN::CAT =====")

if cat_events:
    for event in cat_events:
        print(f"Chamadas: {event.count}")
        print(
            f"Memória alocada: "
            f"{event.self_cpu_memory_usage / 1024**2:.2f} MiB"
        )
        print(
            f"Tempo CPU próprio: "
            f"{event.self_cpu_time_total / 1000:.2f} ms"
        )
else:
    print("Nenhuma operação aten::cat registrada.")

print("\n===== ESTADO FINAL DO CACHE =====")
print(f"Comprimento: {cache.length}")
print(f"Camadas: {len(cache)}")

print("\n===== PROFILING CONCLUÍDO =====")
