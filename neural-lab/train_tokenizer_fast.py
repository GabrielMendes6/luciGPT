"""
Script otimizado e rápido para treinar o tokenizer BPE da LUCI com 4096 ou 8192 tokens.
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path.cwd()))

from tokenizer.bpe import BytePairTokenizer
from tokenizer.serialization import save_tokenizer


def main():
    print("=" * 80)
    print("TREINAMENTO RÁPIDO DO TOKENIZER BPE DA LUCI")
    print("=" * 80)

    VOCAB_SIZE = 4096  # Reduzido para 4K para treino super rápido e eficiente
    TRAIN_DATA_PATH = Path("datasets/training/carolina/train.jsonl")
    OUTPUT_PATH = Path(f"checkpoints/tokenizer/luci_bpe_{VOCAB_SIZE}.json")

    print(f"📖 Carregando dados de {TRAIN_DATA_PATH}...")
    texts = []
    total_chars = 0
    max_chars = 10_000_000  # 10M caracteres é mais que suficiente para um ótimo BPE

    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if total_chars >= max_chars:
                break
            try:
                doc = json.loads(line)
                text = doc.get('text', '')
                if text:
                    texts.append(text)
                    total_chars += len(text)
            except:
                continue

    print(f"   ✅ Carregados {len(texts):,} documentos ({total_chars:,} caracteres)")

    print(f"\n🔧 Treinando BytePairTokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = BytePairTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(texts)

    # Adiciona tokens especiais
    try:
        tokenizer.add_special_token("<|endoftext|>")
        tokenizer.add_special_token("<|pad|>")
    except Exception as e:
        print(f"   Nota sobre tokens especiais: {e}")

    print(f"   ✅ Treinamento concluído!")
    print(f"   Vocabulário final: {tokenizer.actual_vocab_size} tokens")

    # Testa tokenizer
    sample = "Olá, Luci! O sistema de inteligência artificial está funcionando perfeitamente."
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"\n🧪 Teste de codificação:")
    print(f"   Original: {sample}")
    print(f"   Tokens ({len(encoded)}): {encoded[:15]}...")
    print(f"   Decodificado: {decoded}")
    print(f"   Match: {decoded == sample}")

    # Salva
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, str(OUTPUT_PATH))
    print(f"\n💾 Tokenizer salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
