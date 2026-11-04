"""
Script otimizado com limite de caracteres bem restrito para teste imediato do BPE.
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path.cwd()))

from tokenizer.bpe import BytePairTokenizer
from tokenizer.serialization import save_tokenizer

def main():
    print("Iniciando treino rápido do BPE (2048 tokens, 500k chars)...")
    VOCAB_SIZE = 2048
    TRAIN_PATH = Path("datasets/training/carolina/train.jsonl")
    OUTPUT_PATH = Path(f"checkpoints/tokenizer/luci_bpe_{VOCAB_SIZE}.json")

    texts = []
    total = 0
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if total >= 500_000:
                break
            try:
                doc = json.loads(line)
                txt = doc.get('text', '')
                if txt:
                    texts.append(txt)
                    total += len(txt)
            except:
                continue

    print(f"Carregados {len(texts)} docs ({total} chars). Treinando BPE...")
    tokenizer = BytePairTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(texts)

    try:
        tokenizer.add_special_token("<|endoftext|>")
        tokenizer.add_special_token("<|pad|>")
    except:
        pass

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, str(OUTPUT_PATH))
    print(f"Sucesso! Salvo em {OUTPUT_PATH} (vocab size: {tokenizer.actual_vocab_size})")

if __name__ == '__main__':
    main()
