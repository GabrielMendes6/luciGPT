
from tokenizer.bpe import BytePairTokenizer
from tokenizer.serialization import (
    save_tokenizer,
    load_tokenizer,
)
from datasets.portuguese_sample import PORTUGUESE_CORPUS


TOKENIZER_PATH = "checkpoints/tokenizer/luci_bpe_v2.json"

print("===== TREINAMENTO =====")

tokenizer = BytePairTokenizer(vocab_size=1024)
tokenizer.train(PORTUGUESE_CORPUS)

print("Vocabulário:", tokenizer.actual_vocab_size)
print("Merges:", len(tokenizer.merges))

eos_id = tokenizer.add_special_token("<|endoftext|>")

print("EOS ID:", eos_id)
print("Vocabulário total:", tokenizer.actual_vocab_size)

print("\n===== SALVAMENTO =====")

save_tokenizer(tokenizer, TOKENIZER_PATH)

print("Tokenizer salvo:", TOKENIZER_PATH)

print("\n===== CARREGAMENTO =====")

loaded_tokenizer = load_tokenizer(TOKENIZER_PATH)

print("Vocabulário carregado:", loaded_tokenizer.actual_vocab_size)
print("Merges carregados:", len(loaded_tokenizer.merges))

print("\n===== COMPARAÇÃO =====")

test_cases = [
    "Olá, LUCI!",
    "Inteligência artificial",
    "Programação em Python",
    "São João, coração e ação.",
    "🐺 Lobo em pele de cordeiro.",
    "A LUCI possui seu próprio tokenizer.",
]

for text in test_cases:
    original_ids = tokenizer.encode(text, add_eos=True)
    loaded_ids = loaded_tokenizer.encode(text, add_eos=True)

    decoded = loaded_tokenizer.decode(
        loaded_ids,
        skip_special_tokens=True,
    )

    print(
        repr(text),
        "| IDs iguais:", original_ids == loaded_ids,
        "| Reconstrução:", decoded == text,
    )

assert tokenizer.token_bytes == loaded_tokenizer.token_bytes
assert tokenizer.merges == loaded_tokenizer.merges
assert loaded_tokenizer.eos_token_id == tokenizer.eos_token_id
assert loaded_tokenizer.actual_vocab_size == 848
assert loaded_tokenizer.special_tokens == tokenizer.special_tokens

print("\nPERSISTÊNCIA VALIDADA COM SUCESSO!")
