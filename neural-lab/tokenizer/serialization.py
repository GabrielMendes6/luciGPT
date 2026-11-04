
import json
from pathlib import Path

from tokenizer.bpe import BytePairTokenizer


FORMAT_VERSION = 2


def save_tokenizer(
    tokenizer: BytePairTokenizer,
    path: str | Path,
) -> None:
    if not tokenizer.trained:
        raise RuntimeError("Não é possível salvar um tokenizer não treinado.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "format_version": FORMAT_VERSION,
        "algorithm": "byte-level-bpe",
        "target_vocab_size": tokenizer.vocab_size,
        "base_vocab_size": len(tokenizer.token_bytes),
        "actual_vocab_size": tokenizer.actual_vocab_size,
        "token_bytes": {
            str(token_id): token.hex()
            for token_id, token in tokenizer.token_bytes.items()
        },
        "special_tokens": tokenizer.special_tokens,
        "merges": [
            {
                "left": left,
                "right": right,
            }
            for left, right in tokenizer.merges
        ],
    }

    with destination.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_tokenizer(
    path: str | Path,
) -> BytePairTokenizer:
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"Arquivo do tokenizer não encontrado: {source}"
        )

    with source.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError("Versão do formato incompatível.")

    if data.get("algorithm") != "byte-level-bpe":
        raise ValueError("Algoritmo incompatível.")

    tokenizer = BytePairTokenizer(
        vocab_size=data["target_vocab_size"]
    )

    token_bytes = {
        int(token_id): bytes.fromhex(token_hex)
        for token_id, token_hex in data["token_bytes"].items()
    }

    base_vocab_size = data["base_vocab_size"]
    actual_vocab_size = data["actual_vocab_size"]

    if set(token_bytes) != set(range(base_vocab_size)):
        raise ValueError("IDs do vocabulário inválidos ou incompletos.")

    if base_vocab_size < 256:
        raise ValueError("Vocabulário base inválido.")

    for token_id in range(256):
        if token_bytes[token_id] != bytes([token_id]):
            raise ValueError("Vocabulário base de bytes inválido.")

    special_tokens = {
        str(token): int(token_id)
        for token, token_id in data["special_tokens"].items()
    }

    expected_special_ids = set(
        range(base_vocab_size, base_vocab_size + len(special_tokens))
    )

    if set(special_tokens.values()) != expected_special_ids:
        raise ValueError("IDs dos tokens especiais inválidos.")

    if actual_vocab_size != base_vocab_size + len(special_tokens):
        raise ValueError("Tamanho total do vocabulário inconsistente.")

    merges = [
        (item["left"], item["right"])
        for item in data["merges"]
    ]

    if len(merges) != base_vocab_size - 256:
        raise ValueError("Quantidade de merges inconsistente.")

    for merge_index, (left, right) in enumerate(merges):
        new_id = 256 + merge_index

        if left not in token_bytes or right not in token_bytes:
            raise ValueError("Regra de merge referencia token inexistente.")

        expected_bytes = token_bytes[left] + token_bytes[right]

        if token_bytes.get(new_id) != expected_bytes:
            raise ValueError("Regra de merge incompatível com o vocabulário.")

    tokenizer.token_bytes = token_bytes
    tokenizer.special_tokens = special_tokens
    tokenizer.merges = merges
    tokenizer.trained = True

    return tokenizer
