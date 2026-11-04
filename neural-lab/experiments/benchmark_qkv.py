
import statistics
import time
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT

torch.manual_seed(42)
torch.set_num_threads(2)

PROMPT_LENGTH = 128
DECODE_STEPS = 64
REPETITIONS = 7

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
def run_generation():
    logits, cache = model(prompt, use_cache=True)

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


# Aquecimento
for _ in range(2):
    run_generation()

latencies = []

for _ in range(REPETITIONS):
    start = time.perf_counter()

    final_length = run_generation()

    elapsed = time.perf_counter() - start
    latencies.append(elapsed)

median = statistics.median(latencies)
mean = statistics.mean(latencies)

throughput = (
    PROMPT_LENGTH + DECODE_STEPS
) / median

print("===== LUCI QKV BENCHMARK =====")
print("PyTorch:", torch.__version__)
print("Threads:", torch.get_num_threads())
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Prompt:", PROMPT_LENGTH)
print("Decode:", DECODE_STEPS)
print("Repetitions:", REPETITIONS)

print("\nLatências:", [round(x, 4) for x in latencies])
print(f"Mediana: {median:.4f}s")
print(f"Média: {mean:.4f}s")
print(f"Throughput: {throughput:.2f} tok/s")
print(f"Cache final: {final_length}")
