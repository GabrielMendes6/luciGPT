"""
Análise detalhada da configuração do modelo Luci-GPT.
Calcula parâmetros totais, memória necessária e compara diferentes configurações.
"""

import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from architecture.config import ModelConfig
from architecture.model import LuciGPT
import torch


def count_parameters(model):
    """Conta parâmetros totais e treináveis do modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory(config: ModelConfig, batch_size: int = 1, dtype_bytes: int = 4):
    """
    Estima memória necessária para treinamento (aproximação).

    dtype_bytes:
      - 4 para float32
      - 2 para float16/bfloat16
      - 1 para int8
    """
    # Parâmetros do modelo
    model = LuciGPT(config)
    total_params, _ = count_parameters(model)

    # Memória dos parâmetros (pesos)
    params_memory_mb = (total_params * dtype_bytes) / (1024 ** 2)

    # Memória dos gradientes (mesmo tamanho dos parâmetros)
    gradients_memory_mb = params_memory_mb

    # Memória dos optimizer states (Adam usa ~2x os parâmetros)
    optimizer_memory_mb = params_memory_mb * 2

    # Memória das ativações (depende de batch_size e seq_len)
    # Aproximação: ~4 bytes por ativação por camada
    seq_len = config.max_seq_len
    activations_per_layer = batch_size * seq_len * config.embedding_dim
    activations_memory_mb = (activations_per_layer * config.num_layers * dtype_bytes) / (1024 ** 2)

    # KV Cache durante inferência
    kv_cache_memory_mb = (
        2 *  # K e V
        config.num_layers *
        batch_size *
        config.max_seq_len *
        config.num_kv_heads *
        config.head_dim *
        dtype_bytes
    ) / (1024 ** 2)

    total_training_mb = params_memory_mb + gradients_memory_mb + optimizer_memory_mb + activations_memory_mb
    total_inference_mb = params_memory_mb + kv_cache_memory_mb

    return {
        'params_mb': params_memory_mb,
        'gradients_mb': gradients_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'activations_mb': activations_memory_mb,
        'kv_cache_mb': kv_cache_memory_mb,
        'total_training_mb': total_training_mb,
        'total_training_gb': total_training_mb / 1024,
        'total_inference_mb': total_inference_mb,
        'total_inference_gb': total_inference_mb / 1024,
    }


def analyze_config(name: str, config: ModelConfig, batch_size: int = 1):
    """Analisa uma configuração específica e imprime estatísticas."""
    print(f"\n{'=' * 80}")
    print(f"CONFIGURAÇÃO: {name}")
    print(f"{'=' * 80}")

    # Parâmetros da configuração
    print("\n📊 HIPERPARÂMETROS:")
    print(f"  • Vocabulário (vocab_size):        {config.vocab_size:,}")
    print(f"  • Dimensão de Embedding:            {config.embedding_dim}")
    print(f"  • Tamanho Máximo de Sequência:      {config.max_seq_len}")
    print(f"  • Número de Camadas:                {config.num_layers}")
    print(f"  • Número de Cabeças de Atenção:     {config.num_heads}")
    print(f"  • Número de Cabeças KV (GQA):       {config.num_kv_heads}")
    print(f"  • Dimensão por Cabeça:              {config.head_dim}")
    print(f"  • Dimensão Feed-Forward:            {config.ffn_dim}")
    print(f"  • Razão FFN/Embedding:              {config.ffn_dim / config.embedding_dim:.2f}x")

    # Contagem de parâmetros
    model = LuciGPT(config)
    total_params, trainable_params = count_parameters(model)

    print(f"\n🔢 PARÂMETROS DO MODELO:")
    print(f"  • Total de Parâmetros:              {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  • Parâmetros Treináveis:            {trainable_params:,} ({trainable_params / 1e6:.1f}M)")

    # Breakdown de parâmetros por componente
    embedding_params = config.vocab_size * config.embedding_dim
    lm_head_params = config.vocab_size * config.embedding_dim

    # Por camada de transformer
    # Atenção: Q, K, V projections + output projection
    attention_params_per_layer = (
        config.embedding_dim * (config.num_heads * config.head_dim) +  # Q
        config.embedding_dim * (config.num_kv_heads * config.head_dim) +  # K
        config.embedding_dim * (config.num_kv_heads * config.head_dim) +  # V
        (config.num_heads * config.head_dim) * config.embedding_dim  # O
    )

    # Feed-forward: 2 projeções lineares
    ffn_params_per_layer = (
        config.embedding_dim * config.ffn_dim +  # up projection
        config.ffn_dim * config.embedding_dim     # down projection
    )

    # Normalização (RMSNorm tem apenas 1 parâmetro por dimensão)
    norm_params_per_layer = config.embedding_dim * 2  # 2 LayerNorms por bloco

    layer_params = (attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer) * config.num_layers

    print(f"\n  Breakdown:")
    print(f"    - Embedding de Entrada:           {embedding_params:,} ({embedding_params / total_params * 100:.1f}%)")
    print(f"    - Camadas Transformer (total):    {layer_params:,} ({layer_params / total_params * 100:.1f}%)")
    print(f"    - LM Head (saída):                {lm_head_params:,} ({lm_head_params / total_params * 100:.1f}%)")

    # Estimativa de memória
    mem_fp32 = estimate_memory(config, batch_size=batch_size, dtype_bytes=4)
    mem_fp16 = estimate_memory(config, batch_size=batch_size, dtype_bytes=2)

    print(f"\n💾 MEMÓRIA ESTIMADA (batch_size={batch_size}):")
    print(f"\n  Float32 (FP32):")
    print(f"    • Treinamento:  {mem_fp32['total_training_gb']:.2f} GB")
    print(f"    • Inferência:   {mem_fp32['total_inference_gb']:.2f} GB")

    print(f"\n  Float16/BFloat16 (FP16):")
    print(f"    • Treinamento:  {mem_fp16['total_training_gb']:.2f} GB")
    print(f"    • Inferência:   {mem_fp16['total_inference_gb']:.2f} GB")

    print(f"\n  Breakdown (FP16):")
    print(f"    - Parâmetros:       {mem_fp16['params_mb']:.1f} MB")
    print(f"    - Gradientes:       {mem_fp16['gradients_mb']:.1f} MB")
    print(f"    - Optimizer State:  {mem_fp16['optimizer_mb']:.1f} MB")
    print(f"    - Ativações:        {mem_fp16['activations_mb']:.1f} MB")
    print(f"    - KV Cache:         {mem_fp16['kv_cache_mb']:.1f} MB")

    # Comparação com modelos conhecidos
    if 10e6 <= total_params <= 30e6:
        ref = "TinyLlama/GPT-2 Small"
    elif 100e6 <= total_params <= 200e6:
        ref = "GPT-2 Medium"
    elif 300e6 <= total_params <= 500e6:
        ref = "GPT-2 Large"
    elif 1e9 <= total_params <= 2e9:
        ref = "GPT-2 XL / Llama-1B"
    elif 6e9 <= total_params <= 8e9:
        ref = "Llama-7B"
    elif 12e9 <= total_params <= 14e9:
        ref = "Llama-13B"
    else:
        ref = "Custom size"

    print(f"\n🏷️  CATEGORIA: ~{ref}")

    return total_params, mem_fp16['total_training_gb'], mem_fp16['total_inference_gb']


print("=" * 80)
print("ANÁLISE DE CONFIGURAÇÕES DO MODELO LUCI-GPT")
print("=" * 80)

# Configuração padrão atual
config_default = ModelConfig(
    vocab_size=16384,
    embedding_dim=768,
    max_seq_len=4096,
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
    ffn_dim=3072
)

# Configurações alternativas

# Small: para testes rápidos e prototipagem
config_small = ModelConfig(
    vocab_size=16384,
    embedding_dim=512,
    max_seq_len=2048,
    num_heads=8,
    num_kv_heads=2,
    num_layers=8,
    ffn_dim=2048
)

# Medium: balanceado, bom para recursos limitados
config_medium = ModelConfig(
    vocab_size=16384,
    embedding_dim=768,
    max_seq_len=2048,
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
    ffn_dim=3072
)

# Large: melhor performance, mais recursos
config_large = ModelConfig(
    vocab_size=16384,
    embedding_dim=1024,
    max_seq_len=4096,
    num_heads=16,
    num_kv_heads=4,
    num_layers=24,
    ffn_dim=4096
)

# Micro: para debugging e experimentação
config_micro = ModelConfig(
    vocab_size=1024,
    embedding_dim=256,
    max_seq_len=512,
    num_heads=4,
    num_kv_heads=2,
    num_layers=4,
    ffn_dim=512
)

configs = [
    ("MICRO (Debug/Teste)", config_micro, 4),
    ("SMALL (Protótipo)", config_small, 2),
    ("MEDIUM (Balanceado)", config_medium, 2),
    ("DEFAULT (Configuração Atual)", config_default, 1),
    ("LARGE (Alta Performance)", config_large, 1),
]

results = []
for name, config, batch_size in configs:
    params, train_gb, infer_gb = analyze_config(name, config, batch_size)
    results.append((name, params, train_gb, infer_gb))

# Resumo comparativo
print(f"\n\n{'=' * 80}")
print("RESUMO COMPARATIVO")
print(f"{'=' * 80}")
print(f"\n{'Configuração':<30} {'Parâmetros':<15} {'Treino (GB)':<15} {'Inferência (GB)':<15}")
print("-" * 80)
for name, params, train_gb, infer_gb in results:
    print(f"{name:<30} {params/1e6:>10.1f}M     {train_gb:>10.2f}        {infer_gb:>10.2f}")

print(f"\n{'=' * 80}")
print("RECOMENDAÇÕES")
print(f"{'=' * 80}")
print("""
📌 VOCAB_SIZE (Tamanho do Vocabulário):
   • Seu tokenizer atual tem 847 tokens
   • Você configurou 16384 (16K) - isso está MUITO acima!
   • Recomendação: ajustar para 1024 ou 2048 (próximo ao tamanho real do seu BPE)
   • Impacto: vocabulário grande = mais parâmetros na camada de embedding e LM head

📌 EMBEDDING_DIM (Dimensão de Embedding/Hidden):
   • 512: modelos small/protótipos
   • 768: padrão GPT-2 base (bom balanceamento)
   • 1024+: modelos maiores, precisa mais dados e compute

📌 MAX_SEQ_LEN (Contexto Máximo):
   • 512-1024: suficiente para a maioria das tarefas
   • 2048: contexto confortável para documentos
   • 4096: contextos longos, mas quadruplica memória de atenção

📌 NUM_LAYERS (Profundidade):
   • 4-6: testes rápidos
   • 8-12: modelos práticos
   • 16-24: modelos grandes (requer mais dados)

📌 NUM_HEADS / NUM_KV_HEADS:
   • num_heads: geralmente 8, 12, ou 16
   • num_kv_heads < num_heads = Grouped Query Attention (GQA) - economiza memória!
   • Razão típica: 2x, 3x ou 4x (ex: 12 heads / 4 KV heads)

📌 FFN_DIM (Dimensão Feed-Forward):
   • Geralmente ~4x embedding_dim (ex: 768 → 3072)
   • Modelos recentes usam até 8x em arquiteturas SwiGLU

⚠️  ATENÇÃO: Seu vocab_size=16384 está desperdiçando ~15K embeddings não usadas!
   Ajuste para o tamanho real do seu tokenizer (847 → arredondar para 1024).
""")
