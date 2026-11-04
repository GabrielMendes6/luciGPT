
from collections import Counter


class BytePairTokenizer:
    """
    Tokenizer BPE próprio da LUCI.

    IDs 0-255: bytes UTF-8.
    IDs 256+: tokens aprendidos pelo BPE.
    IDs posteriores: tokens especiais.
    """

    def __init__(self, vocab_size: int = 1024):
        if vocab_size < 256:
            raise ValueError(
                "O vocabulário precisa ter pelo menos 256 tokens."
            )

        self.vocab_size = vocab_size

        self.token_bytes: dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }

        self.merges: list[tuple[int, int]] = []
        self.special_tokens: dict[str, int] = {}

        self.trained = False

    @staticmethod
    def _merge_pair(
        sequence: list[int],
        pair: tuple[int, int],
        new_id: int,
    ) -> list[int]:
        result = []
        i = 0

        while i < len(sequence):
            if (
                i + 1 < len(sequence)
                and (sequence[i], sequence[i + 1]) == pair
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(sequence[i])
                i += 1

        return result

    def train(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("O corpus não pode estar vazio.")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Todos os documentos precisam ser strings.")

        sequences = [
            list(text.encode("utf-8"))
            for text in texts
        ]

        self.token_bytes = {
            i: bytes([i]) for i in range(256)
        }

        self.merges = []
        self.special_tokens = {}

        while (
            len(self.token_bytes) + len(self.special_tokens)
            < self.vocab_size
        ):
            pair_counts = Counter()

            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    pair_counts[
                        (sequence[i], sequence[i + 1])
                    ] += 1

            if not pair_counts:
                break

            best_pair, _ = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )

            left, right = best_pair
            new_id = len(self.token_bytes)

            self.token_bytes[new_id] = (
                self.token_bytes[left]
                + self.token_bytes[right]
            )

            self.merges.append(best_pair)

            sequences = [
                self._merge_pair(sequence, best_pair, new_id)
                for sequence in sequences
            ]

        self.trained = True

    def add_special_token(self, token: str) -> int:
        if not self.trained:
            raise RuntimeError("Treine o tokenizer antes.")

        if not isinstance(token, str) or not token:
            raise ValueError("O token especial precisa ser uma string.")

        if token in self.special_tokens:
            return self.special_tokens[token]

        token_id = len(self.token_bytes) + len(self.special_tokens)

        if token_id >= self.vocab_size:
            raise ValueError("O vocabulário máximo foi atingido.")

        self.special_tokens[token] = token_id

        return token_id

    @property
    def eos_token_id(self) -> int:
        if "<|endoftext|>" not in self.special_tokens:
            raise RuntimeError("Token EOS ainda não foi registrado.")

        return self.special_tokens["<|endoftext|>"]

    @property
    def actual_vocab_size(self) -> int:
        return len(self.token_bytes) + len(self.special_tokens)

    def encode(
        self,
        text: str,
        add_eos: bool = False,
    ) -> list[int]:
        if not self.trained:
            raise RuntimeError("O tokenizer ainda não foi treinado.")

        if not isinstance(text, str):
            raise TypeError("A entrada precisa ser uma string.")

        sequence = list(text.encode("utf-8"))

        for merge_index, pair in enumerate(self.merges):
            new_id = 256 + merge_index

            sequence = self._merge_pair(
                sequence,
                pair,
                new_id,
            )

        if add_eos:
            sequence.append(self.eos_token_id)

        return sequence

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        if not self.trained:
            raise RuntimeError("O tokenizer ainda não foi treinado.")

        special_by_id = {
            token_id: token
            for token, token_id in self.special_tokens.items()
        }

        result = []
        byte_buffer = bytearray()

        for token_id in token_ids:
            if token_id in self.token_bytes:
                byte_buffer.extend(self.token_bytes[token_id])

            elif token_id in special_by_id:
                if byte_buffer:
                    result.append(
                        bytes(byte_buffer).decode("utf-8", errors="replace")
                    )
                    byte_buffer.clear()

                if not skip_special_tokens:
                    result.append(special_by_id[token_id])

            else:
                # Ignora ou trata IDs inválidos gerados por pesos aleatórios
                continue

        if byte_buffer:
            result.append(bytes(byte_buffer).decode("utf-8", errors="replace"))

        return "".join(result)
