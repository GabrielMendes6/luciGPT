
from tokenizer.bpe import BytePairTokenizer
from datasets.portuguese_sample import PORTUGUESE_CORPUS


tokenizer = BytePairTokenizer(vocab_size=1024)

print("===== LUCI BPE TRAINING =====")

tokenizer.train(PORTUGUESE_CORPUS)

print("Vocabulário desejado:", tokenizer.vocab_size)
print("Vocabulário aprendido:", tokenizer.actual_vocab_size)
print("Merges realizados:", len(tokenizer.merges))

test_cases = [
    "Olá, LUCI!",
    "Inteligência artificial",
    "Programação em Python",
    "São João, coração e ação.",
    "🐺 Lobo em pele de cordeiro.",
    "A LUCI possui seu próprio tokenizer.",
]

print("\n===== TOKENIZATION =====")

for text in test_cases:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("\nTexto:", repr(text))
    print("IDs:", encoded)
    print("Quantidade de tokens:", len(encoded))
    print("Reconstrução correta:", decoded == text)

print("\n===== COMPARAÇÃO BYTE VS BPE =====")

for text in test_cases:
    byte_count = len(text.encode("utf-8"))
    bpe_count = len(tokenizer.encode(text))

    print(
        repr(text),
        "| Bytes:", byte_count,
        "| BPE:", bpe_count,
        "| Redução:",
        round((1 - bpe_count / byte_count) * 100, 2),
        "%"
    )
