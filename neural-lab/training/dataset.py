
import torch

from torch.utils.data import Dataset

from tokenizer.bpe import BytePairTokenizer


class CausalLanguageModelDataset(Dataset):
    """
    Dataset para treinamento autoregressivo.

    Cada amostra contém:
    X: sequência de entrada.
    Y: sequência deslocada para prever o próximo token.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: BytePairTokenizer,
        sequence_length: int = 128,
    ):
        if not tokenizer.trained:
            raise RuntimeError("Tokenizer ainda não foi treinado.")

        if sequence_length <= 0:
            raise ValueError("sequence_length precisa ser positivo.")

        if not texts:
            raise ValueError("O dataset não pode estar vazio.")

        self.sequence_length = sequence_length

        all_tokens = []

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Todos os documentos precisam ser strings.")

            if not text:
                continue

            all_tokens.extend(
                tokenizer.encode(text, add_eos=True)
            )

        if len(all_tokens) < sequence_length + 1:
            raise ValueError(
                "Quantidade insuficiente de tokens para criar uma amostra."
            )

        self.tokens = torch.tensor(
            all_tokens,
            dtype=torch.long,
        )

        self.num_samples = (
            len(self.tokens) - 1
        ) // sequence_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        if index < 0 or index >= self.num_samples:
            raise IndexError("Índice fora dos limites do dataset.")

        start = index * self.sequence_length
        end = start + self.sequence_length

        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]

        return x, y
