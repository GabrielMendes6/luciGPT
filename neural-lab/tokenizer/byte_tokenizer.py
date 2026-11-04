class ByteTokenizer:
    """
    Tokenizer básico da LUCI.

    Cada byte UTF-8 corresponde a um token.
    Vocabulário inicial: 256 tokens.
    """

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("A entrada precisa ser uma string.")

        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        if not all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token <= 255
            for token in token_ids
        ):
            raise ValueError("Token ID inválido.")

        return bytes(token_ids).decode("utf-8")