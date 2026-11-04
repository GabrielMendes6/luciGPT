import sys
from pathlib import Path
import torch

sys.path.append(str(Path.cwd()))

from tokenizer.serialization import load_tokenizer
from architecture.config import ModelConfig
from architecture.model import LuciGPT
from architecture.generation import generate

print("=" * 60)
print("1. AUDITORIA DO TOKENIZER BPE DA LUCI")
print("=" * 60)
tokenizer_path = Path("checkpoints/tokenizer/luci_bpe_v2.json")
if tokenizer_path.exists():
    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = len(tokenizer.token_bytes)
    print(f"[OK] Tokenizer carregado de {tokenizer_path}")
    print(f"     Tamanho do Vocabulário: {vocab_size}")

    sample_text = "A inteligência artificial e os modelos de linguagem em português estão avançando rapidamente."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"     Texto original: {sample_text}")
    print(f"     Tokens gerados ({len(encoded)}): {encoded[:15]}...")
    print(f"     Decodificação correta? {decoded == sample_text}")
    print(f"     Texto decodificado: {decoded}")
else:
    print(f"[AVISO] Checkpoint do tokenizer não encontrado em {tokenizer_path}")

print("\n" + "=" * 60)
print("2. INSPEÇÃO DOS DATASETS E CORPUS")
print("=" * 60)
datasets_dir = Path("datasets")
for sub in datasets_dir.iterdir():
    if sub.is_dir():
        items = list(sub.iterdir())
        print(f"     Pasta 'datasets/{sub.name}': {len(items)} itens ({[x.name for x in items[:4]]})")

print("\n" + "=" * 60)
print("3. INICIALIZAÇÃO E TESTE DE FORWARD PASS DA LUCI-GPT")
print("=" * 60)
config = ModelConfig(
    vocab_size=847 if tokenizer_path.exists() else 1024,
    embedding_dim=256,
    max_seq_len=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers=4,
    ffn_dim=512
)
model = LuciGPT(config)
model.eval()

print(f"[OK] LuciGPT inicializada com sucesso!")
print(f"     Configuração: {config}")

# Teste de Forward Pass
batch_size = 2
seq_len = 32
dummy_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
with torch.no_grad():
    logits = model(dummy_tokens)
print(f"     Shape da entrada: {dummy_tokens.shape}")
print(f"     Shape dos logits de saída: {logits.shape}")
assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Shape inesperado: {logits.shape}"
print("[OK] Forward pass executado e validado com sucesso!")

print("\n" + "=" * 60)
print("4. TESTE DE GERAÇÃO E DECODIFICAÇÃO")
print("=" * 60)
prompt = "Luci, o modelo"
if tokenizer_path.exists():
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    print(f"     Prompt: '{prompt}' -> IDs: {prompt_ids}")

    # Gerando alguns tokens com a função de geração do repositório
    generated_ids = generate(
        model=model,
        token_ids=input_tensor,
        max_new_tokens=10
    )
    generated_list = generated_ids[0].tolist()
    decoded_output = tokenizer.decode(generated_list)
    print(f"     IDs gerados: {generated_list}")
    print(f"     Texto gerado (pós-geração): {decoded_output}")
    print("[OK] Teste de geração concluído com sucesso!")
else:
    print("[PULADO] Geração não testada por falta do tokenizer.")

print("\n" + "=" * 60)
print("TUDO PRONTO PARA INICIAR O TREINAMENTO DA LUCI-GPT!")
print("=" * 60)
