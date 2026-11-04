"""
Script de diagnóstico de hardware para recomendar configuração ideal da LUCI.
"""

import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

print("=" * 80)
print("DIAGNÓSTICO DE HARDWARE PARA LUCI-GPT")
print("=" * 80)

# CPU Info
print("\n🖥️  CPU:")
try:
    cpu_info = subprocess.check_output("lscpu", shell=True, text=True)
    for line in cpu_info.split("\n"):
        if "Model name" in line or "CPU(s):" in line or "Thread(s) per core" in line:
            print(f"  {line.strip()}")
except:
    print("  [Não foi possível detectar CPU]")

# RAM Info
print("\n💾 MEMÓRIA RAM:")
try:
    mem_info = subprocess.check_output("free -h", shell=True, text=True)
    lines = mem_info.split("\n")
    print(f"  {lines[0]}")
    print(f"  {lines[1]}")
except:
    print("  [Não foi possível detectar RAM]")

# GPU Info
print("\n🎮 GPU:")
try:
    # Tenta NVIDIA primeiro
    nvidia_info = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, text=True)
    gpus = nvidia_info.strip().split("\n")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")

    has_gpu = True
except:
    print("  ❌ Nenhuma GPU NVIDIA detectada")
    has_gpu = False

    # Tenta AMD
    try:
        amd_info = subprocess.check_output("lspci | grep -i vga", shell=True, text=True)
        print(f"  Info: {amd_info.strip()}")
    except:
        print("  💻 Usando apenas CPU")

# Disk Space
print("\n💿 ESPAÇO EM DISCO (para checkpoints e datasets):")
try:
    disk_info = subprocess.check_output("df -h .", shell=True, text=True)
    lines = disk_info.split("\n")
    print(f"  {lines[0]}")
    print(f"  {lines[1]}")
except:
    print("  [Não foi possível detectar espaço em disco]")

# PyTorch + CUDA
print("\n🔥 PYTORCH & CUDA:")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Número de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      VRAM: {props.total_memory / 1024**3:.1f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
except ImportError:
    print("  ⚠️  PyTorch não instalado")
except Exception as e:
    print(f"  Erro ao verificar PyTorch: {e}")

print("\n" + "=" * 80)
print("RECOMENDAÇÕES")
print("=" * 80)

# Determinar recomendação baseado no hardware detectado
try:
    import torch
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        if vram_gb >= 24:
            print("\n✅ HARDWARE: Excelente (GPU com 24GB+ VRAM)")
            print("\n📊 CONFIGURAÇÃO RECOMENDADA: LARGE")
            print("""
ModelConfig(
    vocab_size=16384,       # Vocabulário rico
    embedding_dim=1024,     # Alta dimensionalidade
    max_seq_len=4096,       # Contexto longo
    num_heads=16,
    num_kv_heads=4,
    num_layers=16,          # Profundo
    ffn_dim=4096
)
# ~200-250M parâmetros
# Treinamento: ~2.0 GB VRAM (batch_size=1, FP16)
# Você pode usar batch_size=4-8 e gradient accumulation
            """)

        elif vram_gb >= 12:
            print("\n✅ HARDWARE: Muito Bom (GPU com 12-24GB VRAM)")
            print("\n📊 CONFIGURAÇÃO RECOMENDADA: MEDIUM-LARGE")
            print("""
ModelConfig(
    vocab_size=8192,        # Vocabulário adequado
    embedding_dim=768,      # Padrão GPT-2
    max_seq_len=2048,       # Contexto confortável
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
    ffn_dim=3072
)
# ~85M parâmetros
# Treinamento: ~0.7 GB VRAM (batch_size=1, FP16)
# Você pode usar batch_size=8-16 com conforto
            """)

        elif vram_gb >= 6:
            print("\n✅ HARDWARE: Bom (GPU com 6-12GB VRAM)")
            print("\n📊 CONFIGURAÇÃO RECOMENDADA: MEDIUM")
            print("""
ModelConfig(
    vocab_size=8192,
    embedding_dim=512,      # Reduzido para caber melhor
    max_seq_len=1024,
    num_heads=8,
    num_kv_heads=2,
    num_layers=10,
    ffn_dim=2048
)
# ~40M parâmetros
# Treinamento: ~0.3 GB VRAM (batch_size=1, FP16)
# Você pode usar batch_size=4-8
            """)

        else:
            print(f"\n⚠️  HARDWARE: Limitado (GPU com {vram_gb:.1f}GB VRAM)")
            print("\n📊 CONFIGURAÇÃO RECOMENDADA: SMALL")
            print("""
ModelConfig(
    vocab_size=4096,        # Vocabulário menor
    embedding_dim=512,
    max_seq_len=1024,
    num_heads=8,
    num_kv_heads=2,
    num_layers=8,
    ffn_dim=2048
)
# ~30M parâmetros
# Treinamento: ~0.25 GB VRAM (batch_size=1, FP16)
# Você pode usar batch_size=2-4
            """)

    else:
        # Sem GPU
        print("\n⚠️  HARDWARE: CPU Only (sem GPU)")
        print("\n📊 CONFIGURAÇÃO RECOMENDADA: MICRO (para testes)")
        print("""
ModelConfig(
    vocab_size=2048,
    embedding_dim=256,
    max_seq_len=512,
    num_heads=4,
    num_kv_heads=2,
    num_layers=6,
    ffn_dim=1024
)
# ~10M parâmetros
# Treinamento em CPU será MUITO lento (dias/semanas)
# RECOMENDAÇÃO: Use Google Colab (grátis) ou alugue GPU (vast.ai, runpod.io)
        """)

except:
    print("\n⚠️  Não foi possível determinar capacidade da GPU")
    print("Execute este script com PyTorch instalado para recomendações precisas.")

print("\n" + "=" * 80)
print("💡 DICAS ADICIONAIS")
print("=" * 80)
print("""
1. VOCAB_SIZE (Tokenizer):
   - 4K-8K tokens: suficiente para português coloquial + técnico
   - 16K tokens: mais expressivo, mas aumenta parâmetros em ~15-20%

2. BATCH_SIZE (durante treino):
   - Comece com batch_size=1 e aumente gradualmente
   - Use gradient accumulation se VRAM for limitada
   - Monitore uso de memória com nvidia-smi

3. MIXED PRECISION (FP16/BF16):
   - SEMPRE use torch.cuda.amp para economizar memória
   - Reduz uso de VRAM em ~50%

4. GRADIENT CHECKPOINTING:
   - Se ainda assim não couber na VRAM, ative gradient checkpointing
   - Troca velocidade por memória (25% mais lento, 40% menos memória)

5. TEMPO ESTIMADO DE TREINO:
   - Pequeno (30M): ~2-5 dias em GPU moderna
   - Médio (85M): ~5-10 dias
   - Grande (200M+): ~2-4 semanas
""")
