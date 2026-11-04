
import torch

from architecture.model import LuciGPT
from tokenizer.bpe import BytePairTokenizer
from architecture.generation import generate


class LuciLanguageModel:
    def __init__(
        self,
        model: LuciGPT,
        tokenizer: BytePairTokenizer,
    ):
        if not tokenizer.trained:
            raise RuntimeError("Tokenizer ainda não foi treinado.")

        if model.config.vocab_size != tokenizer.actual_vocab_size:
            raise ValueError(
                "O vocabulário do modelo não corresponde ao tokenizer."
            )

        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text)

        return torch.tensor(
            [token_ids],
            dtype=torch.long,
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.squeeze(0).tolist()

        return self.tokenizer.decode(ids)

    def forward(self, text: str) -> torch.Tensor:
        tokens = self.encode(text)

        return self.model(tokens)
    
    def generate(
        self,
        text: str,
        max_new_tokens: int = 64,
    ) -> str:

        tokens = self.encode(text)

        generated = generate(
            model=self.model,
            token_ids=tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )

        return self.decode(generated)
    