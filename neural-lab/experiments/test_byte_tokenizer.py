from tokenizer.byte_tokenizer import ByteTokenizer


tokenizer = ByteTokenizer()

test_cases = [
    "Olá, LUCI!",
    "Inteligência artificial",
    "Programação em Python",
    "São João, coração e ação.",
    "🐺 Lobo em pele de cordeiro.",
    "Primeira linha\nSegunda linha",
]

print("===== LUCI BYTE TOKENIZER =====")
print("Vocabulário:", tokenizer.vocab_size)

for text in test_cases:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("\nTexto original:", repr(text))
    print("Tokens:", encoded)
    print("Quantidade de tokens:", len(encoded))
    print("Reconstrução correta:", decoded == text)

print("\n===== TESTE DE REVERSIBILIDADE =====")

original = "\n".join(test_cases)

encoded = tokenizer.encode(original)
decoded = tokenizer.decode(encoded)

print("Texto original preservado:", original == decoded)
print("Quantidade total de bytes:", len(encoded))