
import time
import resource
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


# ============================================================
# CONFIGURAÇÃO
# ============================================================

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

BATCH_SIZE = 1
PROMPT_LENGTH = 32
NEW_TOKENS = 32
REPETITIONS = 5


# ============================================================
# MÉTRICAS DE MEMÓRIA
# ============================================================

def get_rss_mb():
    """RAM atualmente residente no processo."""

    with open("/proc/self/status", "r") as file:
        for line in file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024

    return 0.0


def get_peak_rss_mb():
    """Maior RSS registrado durante a vida do processo."""

    return resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024


def get_cache_size_mb(cache):
    """Tamanho exato dos tensores K/V."""

    total_bytes = sum(
        tensor.numel() * tensor.element_size()
        for key, value in cache
        for tensor in (key, value)
    )

    return total_bytes / (1024 ** 2)


def get_parameter_size_mb(model):
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    return total_bytes / (1024 ** 2)


# ============================================================
# INICIALIZAÇÃO
# ============================================================

print("===== LUCI INFERENCE BREAKDOWN =====")

model = LuciGPT(config)
model.eval()

prompt = torch.randint(
    0,
    config.vocab_size,
    (BATCH_SIZE, PROMPT_LENGTH),
)

print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"Threads: {torch.get_num_threads()}")
print(f"Parâmetros em memória: {get_parameter_size_mb(model):.2f} MB")

baseline_rss = get_rss_mb()

print(f"RSS após carregar modelo: {baseline_rss:.2f} MB")


# ============================================================
# 1. PREFILL
# ============================================================

@torch.inference_mode()
def run_prefill():
    return model(
        prompt,
        use_cache=True,
    )


# Aquecimento
_, _ = run_prefill()

prefill_times = []
last_logits = None
last_cache = None

for _ in range(REPETITIONS):
    start = time.perf_counter()

    logits, cache = run_prefill()

    elapsed = time.perf_counter() - start

    prefill_times.append(elapsed)

    last_logits = logits
    last_cache = cache

prefill_average = sum(prefill_times) / len(prefill_times)

cache_size = get_cache_size_mb(last_cache)
rss_after_prefill = get_rss_mb()

print("\n===== PREFILL =====")
print(f"Latências: {[round(x, 4) for x in prefill_times]}")
print(f"Tempo médio: {prefill_average:.4f} s")
print(f"Throughput: {PROMPT_LENGTH / prefill_average:.2f} tokens/s")
print(f"Logits: {tuple(last_logits.shape)}")
print(f"Camadas no cache: {len(last_cache)}")
print(f"Tamanho exato do KV Cache: {cache_size:.2f} MB")
print(f"RSS após prefill: {rss_after_prefill:.2f} MB")
print(f"RAM adicional aproximada: {rss_after_prefill - baseline_rss:.2f} MB")

# ============================================================
# 2. DECODE INCREMENTAL ISOLADO
# ============================================================

@torch.inference_mode()
def run_decode_only():

    # O prefill acontece fora da medição.
    logits, cache = model(
        prompt,
        use_cache=True,
    )

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    # Primeiro token já foi obtido pelo prefill.
    start = time.perf_counter()

    for _ in range(NEW_TOKENS - 1):

        logits, cache = model(
            next_token,
            past_key_values=cache,
            use_cache=True,
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

    elapsed = time.perf_counter() - start

    return elapsed, cache


# Aquecimento
run_decode_only()

decode_times = []
last_decode_cache = None

for _ in range(REPETITIONS):

    elapsed, cache = run_decode_only()

    decode_times.append(elapsed)
    last_decode_cache = cache

decode_average = sum(decode_times) / len(decode_times)

decode_throughput = (
    (NEW_TOKENS - 1) / decode_average
)

print("\n===== DECODE ISOLADO =====")
print(f"Latências: {[round(x, 4) for x in decode_times]}")
print(f"Tempo médio: {decode_average:.4f} s")
print(f"Tokens processados incrementalmente: {NEW_TOKENS - 1}")
print(f"Throughput: {decode_throughput:.2f} tokens/s")
print(f"Latência média por token: {decode_average / (NEW_TOKENS - 1) * 1000:.2f} ms")
print(f"Cache final: {get_cache_size_mb(last_decode_cache):.2f} MB")

# ============================================================
# 3. RESUMO DE MEMÓRIA
# ============================================================

print("\n===== MEMÓRIA =====")
print(f"RSS inicial do modelo: {baseline_rss:.2f} MB")
print(f"RSS após prefill: {rss_after_prefill:.2f} MB")
print(f"RSS atual: {get_rss_mb():.2f} MB")
print(f"Pico RSS do processo: {get_peak_rss_mb():.2f} MB")

print("\n===== BENCHMARK CONCLUÍDO =====")
