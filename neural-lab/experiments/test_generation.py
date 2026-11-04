import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT
from architecture.language_model import LuciLanguageModel
from tokenizer.serialization import load_tokenizer


TOKENIZER_PATH = "checkpoints/tokenizer/luci_bpe_v2.json"

torch.manual_seed(42)

print("===== CARREGANDO TOKENIZER =====")

tokenizer = load_tokenizer(TOKENIZER_PATH)

config = ModelConfig(
    vocab_size=tokenizer.actual_vocab_size,
    embedding_dim=128,
    max_seq_len=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
    ffn_dim=512,
)

model = LuciGPT(config)

language_model = LuciLanguageModel(
    model=model,
    tokenizer=tokenizer,
)

print("Vocabulário:", tokenizer.actual_vocab_size)
print("Parâmetros:", sum(p.numel() for p in model.parameters()))


print("\n===== TESTE 1: GERAÇÃO =====")

prompt = "Olá! A LUCI"

result = language_model.generate(
    prompt,
    max_new_tokens=10,
)

print("Prompt:", prompt)
print("Resultado:", result)
print("Geração executada:", result.startswith(prompt))


print("\n===== TESTE 2: LIMITE DE TOKENS =====")

tokens = language_model.encode(prompt)

from architecture.generation import generate

generated = generate(
    model=model,
    token_ids=tokens,
    max_new_tokens=5,
)

new_tokens = generated.size(1) - tokens.size(1)

print("Tokens solicitados:", 5)
print("Tokens gerados:", new_tokens)

assert new_tokens == 5


print("\n===== TESTE 3: EOS =====")

model.lm_head.weight.data.zero_()
model.lm_head.weight.data[0].fill_(1.0)

with torch.no_grad():
    model.lm_head.weight.data[0].fill_(1.0)

eos_id = tokenizer.eos_token_id

# Força o modelo a escolher EOS.
original_forward = model.forward

def forced_eos_forward(token_ids, **kwargs):
    result = original_forward(token_ids, **kwargs)
    
    if kwargs.get("use_cache", False):
        logits, past_key_values = result
    else:
        logits = result
        past_key_values = None
    
    logits.fill_(-1000)
    logits[:, :, eos_id] = 1000
    
    if kwargs.get("use_cache", False):
        return logits, past_key_values

    return logits

model.forward = forced_eos_forward

eos_result = generate(
    model=model,
    token_ids=tokens,
    eos_token_id=eos_id,
    max_new_tokens=10,
)

model.forward = original_forward

eos_generated = eos_result.size(1) - tokens.size(1)

print("EOS ID:", eos_id)
print("Tokens gerados até EOS:", eos_generated)

assert eos_generated == 1


print("\n===== TESTE 4: ESTADO DO MODELO =====")

model.train()

language_model.generate(
    prompt,
    max_new_tokens=2,
)

print("Modelo em treinamento antes/depois:", model.training)

assert model.training is True


print("\n===== TESTES CONCLUÍDOS =====")
