import torch
from tokenizer.serialization import load_tokenizer
from datasets.portuguese_sample import PORTUGUESE_CORPUS
from training.dataset import CausalLanguageModelDataset


TOKENIZER_PATH = "checkpoints/tokenizer/luci_bpe_v2.json"

print("===== CARREGANDO TOKENIZER =====")

tokenizer = load_tokenizer(TOKENIZER_PATH)

print("Vocabulário:", tokenizer.actual_vocab_size)
print("EOS ID:", tokenizer.eos_token_id)

print("\n===== CRIANDO DATASET =====")

dataset = CausalLanguageModelDataset(
    texts=PORTUGUESE_CORPUS,
    tokenizer=tokenizer,
    sequence_length=128,
)

print("Quantidade de amostras:", len(dataset))
print("Quantidade total de tokens:", len(dataset.tokens))

print("\n===== INSPECIONANDO AMOSTRA =====")

x, y = dataset[0]

print("Formato X:", x.shape)
print("Formato Y:", y.shape)

print("Primeiros tokens X:", x[:15].tolist())
print("Primeiros tokens Y:", y[:15].tolist())

print("\n===== VALIDANDO DESLOCAMENTO =====")

assert x.shape == (128,)
assert y.shape == (128,)
#assert torch_equal := (x[1:] == y[:-1]).all().item()]
torch_equal = torch.equal(x[1:], y[:-1])
assert torch_equal

print("Deslocamento correto:", torch_equal)

print("\n===== VALIDANDO EOS =====")

eos_count = (dataset.tokens == tokenizer.eos_token_id).sum().item()

print("Quantidade de EOS:", eos_count)
assert eos_count == len(PORTUGUESE_CORPUS)

print("\nDATASET VALIDADO COM SUCESSO!")
