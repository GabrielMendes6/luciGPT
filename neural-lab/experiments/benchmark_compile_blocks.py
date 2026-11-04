
import statistics
import time
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT

torch.manual_seed(42)
torch.set_num_threads(2)

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


def compile_blocks(target_model):
    for index, block in enumerate(target_model.transformer_blocks):
        target_model.transformer_blocks[index] = torch.compile(
            block,
            backend="inductor",
        )

    return target_model


@torch.inference_mode()
def run_generation(target_model):
    logits, cache = target_model(prompt, use_cache=True)

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
    ).unsqueeze(1)

    for _ in range(DECODE_STEPS):
        logits, cache = target_model(
            next_token,
            kv_cache=cache,
            use_cache=True,
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        ).unsqueeze(1)

    return cache.length


def benchmark(name, target_model):
    run_generation(target_model)

    latencies = []

    for _ in range(REPETITIONS):
        start = time.perf_counter()

        final_length = run_generation(target_model)

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    median = statistics.median(latencies)

    throughput = (
        PROMPT_LENGTH + DECODE_STEPS
    ) / median

    print(f"\n{name}")
    print("Latências:", [round(x, 4) for x in latencies])
    print(f"Mediana: {median:.4f}s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"Cache: {final_length}")

    return median


print("===== LUCI BLOCK COMPILATION =====")

# Modelo eager
eager_time = benchmark("EAGER", model)

# Compilar blocos individualmente
print("\nCompilando Transformer Blocks...")

compile_start = time.perf_counter()

compiled_model = compile_blocks(model)

# Primeira execução compila os grafos necessários.
run_generation(compiled_model)

compile_elapsed = time.perf_counter() - compile_start

print(f"Compilação inicial: {compile_elapsed:.2f}s")

compiled_time = benchmark(
    "COMPILED BLOCKS",
    compiled_model,
)

print("\n===== COMPARAÇÃO =====")
print(f"Eager: {eager_time:.4f}s")
print(f"Compiled Blocks: {compiled_time:.4f}s")
print(f"Speedup: {eager_time / compiled_time:.2f}x")
print(
    f"Redução de latência: "
    f"{(1 - compiled_time / eager_time) * 100:.2f}%"
)
