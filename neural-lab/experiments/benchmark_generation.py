
import time
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT
from architecture.generation import generate


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

model = LuciGPT(config)
model.eval()

batch_size = 1
prompt_length = 32
new_tokens = 32
repetitions = 5

token_ids = torch.randint(
    0,
    config.vocab_size,
    (batch_size, prompt_length),
)


@torch.inference_mode()
def generate_without_cache(model, token_ids, max_new_tokens):
    generated = token_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(generated)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            (generated, next_token),
            dim=1,
        )

    return generated


def measure(label, function):
    function()  # aquecimento

    measurements = []

    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start

        measurements.append(elapsed)

    average = sum(measurements) / len(measurements)
    throughput = new_tokens / average

    print(f"\n{label}")
    print(f"Latências: {[round(x, 4) for x in measurements]}")
    print(f"Tempo médio: {average:.4f} s")
    print(f"Throughput: {throughput:.2f} tokens/s")

    return average


print("===== LUCI GENERATION BENCHMARK =====")
print(f"PyTorch: {torch.__version__}")
print(f"Threads: {torch.get_num_threads()}")
print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch: {batch_size}")
print(f"Prompt: {prompt_length} tokens")
print(f"Geração: {new_tokens} tokens")
print(f"Repetições: {repetitions}")

# Primeiro, comprova que os dois caminhos geram o mesmo resultado.
reference = generate_without_cache(
    model,
    token_ids,
    new_tokens,
)

cached = generate(
    model,
    token_ids,
    max_new_tokens=new_tokens,
)

assert torch.equal(reference, cached)

print("\nEquivalência: VALIDADA")

# Alterna a ordem para reduzir o efeito da ordem de execução.
results = {}

results["Sem KV Cache"] = measure(
    "SEM KV CACHE",
    lambda: generate_without_cache(
        model,
        token_ids,
        new_tokens,
    ),
)

results["Com KV Cache"] = measure(
    "COM KV CACHE",
    lambda: generate(
        model,
        token_ids,
        max_new_tokens=new_tokens,
    ),
)

speedup = results["Sem KV Cache"] / results["Com KV Cache"]

print("\n===== COMPARAÇÃO FINAL =====")
print(f"Sem cache: {results['Sem KV Cache']:.4f} s")
print(f"Com cache: {results['Com KV Cache']:.4f} s")
print(f"Speedup: {speedup:.2f}x")
print(f"Redução de latência: {(1 - 1 / speedup) * 100:.2f}%")

