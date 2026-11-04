
import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT
from architecture.language_model import LuciLanguageModel
from tokenizer.serialization import load_tokenizer


TOKENIZER_PATH = "checkpoints/tokenizer/luci_bpe_v2.json"

torch.manual_seed(42)

print("===== CARREGANDO TOKENIZER =====")

tokenizer = load_tokenizer(TOKENIZER_PATH)

print("Vocabulário:", tokenizer.actual_vocab_size)

print("\n===== INICIALIZANDO LUCI GPT =====")

config = ModelConfig(
    vocab_size=tokenizer.actual_vocab_size,
)

model = LuciGPT(config)

print("Parâmetros:", sum(
    parameter.numel()
    for parameter in model.parameters()
))

print("\n===== INTEGRANDO COMPONENTES =====")

language_model = LuciLanguageModel(
    model=model,
    tokenizer=tokenizer,
)

text = "Olá! A LUCI está aprendendo português."

tokens = language_model.encode(text)
logits = language_model.forward(text)
decoded = language_model.decode(tokens)

print("Texto original:", text)
print("Tokens:", tokens.tolist())
print("Formato dos logits:", logits.shape)
print("Texto reconstruído:", decoded)
print("Reconstrução correta:", decoded == text)

