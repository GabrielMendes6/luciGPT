"""
Script para treinar o tokenizer BPE da LUCI com 8192 tokens.
Usa o dataset Carolina já processado.
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path.cwd()))

from tokenizer.bpe import BytePairTokenizer
from tokenizer.serialization import save_tokenizer


def load_training_texts(jsonl_path: Path, max_texts: int = None, max_chars: int = 10_000_000):
    """
    Carrega textos do dataset JSONL.

    Args:
        jsonl_path: Caminho para o arquivo train.jsonl
        max_texts: Número máximo de documentos (None = todos)
        max_chars: Número máximo de caracteres total (para não explodir RAM)
    """
    texts = []
    total_chars = 0

    print(f"📖 Carregando textos de {jsonl_path}...")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_texts and i >= max_texts:
                break

            if i % 1000 == 0:
                print(f"   Lidos {i:,} documentos ({total_chars:,} caracteres)...", end='\r')

            doc = json.loads(line)
            text = doc.get('text', '')

            if text:
                texts.append(text)
                total_chars += len(text)

            if total_chars >= max_chars:
                print(f"   Limite de {max_chars:,} caracteres atingido.")
                break

    print(f"   ✅ Carregados {len(texts):,} documentos ({total_chars:,} caracteres)")
    return texts


def train_bpe_tokenizer(
    texts: list[str],
    vocab_size: int = 8192,
    special_tokens: dict[str, int] = None
):
    """
    Treina o tokenizer BPE.

    Args:
        texts: Lista de textos para treinar
        vocab_size: Tamanho do vocabulário target
        special_tokens: Dicionário com tokens especiais {nome: id}
    """
    print(f"\n🔧 Treinando BPE Tokenizer (vocab_size={vocab_size})...")

    # Inicializa o tokenizer
    tokenizer = BytePairTokenizer(vocab_size=vocab_size)

    # Treina o BPE
    tokenizer.train(texts)

    # Adiciona tokens especiais se fornecidos
    if special_tokens:
        print(f"\n➕ Adicionando {len(special_tokens)} tokens especiais...")
        for token_name, token_id in special_tokens.items():
            tokenizer.add_special_token(token_name, token_id)
            print(f"   {token_name} -> ID {token_id}")

    print(f"\n✅ Treinamento concluído!")
    print(f"   Vocabulário final: {len(tokenizer.token_bytes)} tokens")
    print(f"   Merges realizados: {len(tokenizer.merges)}")

    return tokenizer


def test_tokenizer(tokenizer, test_texts: list[str]):
    """Testa o tokenizer com alguns textos."""
    print("\n🧪 TESTANDO TOKENIZER:")
    print("=" * 80)

    for i, text in enumerate(test_texts, 1):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        compression_ratio = len(text.encode('utf-8')) / len(encoded)

        print(f"\n[Teste {i}]")
        print(f"Texto original ({len(text)} chars):")
        print(f"  {text[:100]}...")
        print(f"Tokens gerados: {len(encoded)}")
        print(f"  {encoded[:20]}...")
        print(f"Compressão: {compression_ratio:.2f}x")
        print(f"Decodificação correta: {decoded == text}")


def main():
    print("=" * 80)
    print("TREINAMENTO DO TOKENIZER BPE DA LUCI")
    print("=" * 80)

    # Configuração
    VOCAB_SIZE = 8192
    TRAIN_DATA_PATH = Path("datasets/training/carolina/train.jsonl")
    OUTPUT_PATH = Path("checkpoints/tokenizer/luci_bpe_8k.json")

    # Tokens especiais (você pode customizar)
    SPECIAL_TOKENS = {
        "<|endoftext|>": VOCAB_SIZE,      # Final de documento
        "<|pad|>": VOCAB_SIZE + 1,         # Padding
        "<|unk|>": VOCAB_SIZE + 2,         # Token desconhecido (se precisar)
    }

    # Verifica se o dataset existe
    if not TRAIN_DATA_PATH.exists():
        print(f"❌ Erro: Dataset não encontrado em {TRAIN_DATA_PATH}")
        print("   Execute o script de preparação de dados primeiro.")
        return

    # Carrega textos de treino
    # Nota: Limitando a 10M caracteres para não explodir a RAM
    # Para treino completo, remova max_chars ou aumente o valor
    texts = load_training_texts(
        TRAIN_DATA_PATH,
        max_texts=None,          # Usa todos os documentos
        max_chars=50_000_000     # 50M caracteres (~50MB de texto)
    )

    if not texts:
        print("❌ Nenhum texto carregado. Verifique o dataset.")
        return

    # Treina o tokenizer
    tokenizer = train_bpe_tokenizer(
        texts=texts,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )

    # Testa o tokenizer
    test_samples = [
        "Olá, Luci! Como você está hoje?",
        "A inteligência artificial está revolucionando o mundo.",
        "Precisamos analisar os logs do servidor e verificar possíveis vulnerabilidades.",
        "O treinamento do modelo de linguagem está progredindo bem.",
    ]
    test_tokenizer(tokenizer, test_samples)

    # Salva o tokenizer
    print(f"\n💾 Salvando tokenizer em {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, str(OUTPUT_PATH))
    print(f"   ✅ Tokenizer salvo com sucesso!")

    # Estatísticas finais
    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    print(f"Vocabulário: {len(tokenizer.token_bytes)} tokens")
    print(f"Merges BPE: {len(tokenizer.merges)}")
    print(f"Tokens especiais: {len(SPECIAL_TOKENS)}")
    print(f"Arquivo salvo: {OUTPUT_PATH}")
    print(f"Tamanho do arquivo: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print("\n✅ Treinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
