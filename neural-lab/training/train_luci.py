"""
Script completo de treinamento da LUCI-GPT (Pretraining no Corpus Carolina).
Otimizado para rodar em GPUs no Kaggle (T4/P100 com FP16/AMP).
"""

import os
import sys
import json
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path.cwd()))

from architecture.config import ModelConfig
from architecture.model import LuciGPT
from tokenizer.serialization import load_tokenizer


class CarolinaDataset(Dataset):
    """Dataset para carregar textos tokenizados do Corpus Carolina."""
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        print(f"📖 Carregando dataset de {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if text:
                        # Codifica o texto
                        tokens = self.tokenizer.encode(text)
                        if len(tokens) > 10:  # Descarta textos muito curtos
                            self.samples.append(tokens)
                except Exception:
                    continue

        print(f"   ✅ Carregados {len(self.samples):,} documentos válidos.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        # Trunca ou faz padding para max_seq_len + 1 (para input e target)
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[:self.max_seq_len + 1]
        else:
            # Padding com ID 0 ou equivalente se necessário (aqui usamos pad simples)
            pad_id = 0
            tokens = tokens + [pad_id] * (self.max_seq_len + 1 - len(tokens))

        chunk = torch.tensor(tokens, dtype=torch.long)
        x = chunk[:-1]  # Entrada
        y = chunk[1:]   # Alvo (target deslocado de 1)
        return x, y


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """Cosine learning rate scheduler com warmup."""
    # 1. Warmup linear
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2. Se passou do limite de decay, retorna min_lr
    if it > lr_decay_iters:
        return min_lr
    # 3. Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train():
    print("=" * 80)
    print("TREINAMENTO DA LUCI-GPT (PRETRAINING)")
    print("=" * 80)

    # Configuração de Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Dispositivo de treinamento: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Hiperparâmetros
    batch_size = 4             # Ajuste para 2 ou 1 se faltar VRAM
    gradient_accumulation_steps = 8  # Simula batch maior (effective batch = 4 * 8 = 32)
    max_iters = 5000           # Número total de iterações
    learning_rate = 3e-4       # Learning rate padrão para GPT
    min_lr = 3e-5
    warmup_iters = 200
    lr_decay_iters = 5000
    eval_interval = 250
    save_interval = 500

    # Carrega Tokenizer
    tokenizer_path = Path("checkpoints/tokenizer/luci_bpe_v2.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer não encontrado em {tokenizer_path}")

    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = len(tokenizer.token_bytes)
    print(f"✅ Tokenizer carregado! Vocab size: {vocab_size}")

    # Configuração do Modelo
    config = ModelConfig(
        vocab_size=vocab_size,
        embedding_dim=768,
        max_seq_len=512,       # Reduzido para 512 para treino rápido em 16GB VRAM
        num_heads=12,
        num_kv_heads=4,
        num_layers=12,
        ffn_dim=3072
    )

    print(f"\n🧠 Inicializando LuciGPT...")
    model = LuciGPT(config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parâmetros do Modelo: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Dataset e DataLoader
    train_path = Path("datasets/training/carolina/train.jsonl")
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset de treino não encontrado em {train_path}")

    dataset = CarolinaDataset(str(train_path), tokenizer, max_seq_len=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    # Otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    # Mixed Precision Scaler (para GPUs NVIDIA)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Diretório de Checkpoints
    os.makedirs("checkpoints/model", exist_ok=True)

    print("\n🚀 INICIANDO LOOP DE TREINAMENTO...")
    model.train()
    iter_num = 0
    running_loss = 0.0
    t0 = time.time()

    data_iter = iter(dataloader)

    while iter_num < max_iters:
        # Ajusta Learning Rate
        lr = get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        # Gradient Accumulation Loop
        for micro_step in range(gradient_accumulation_steps):
            try:
                X, Y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                X, Y = next(data_iter)

            X, Y = X.to(device), Y.to(device)

            # Forward pass com Mixed Precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(X)
                # Reshape para calcular Cross Entropy Loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                    ignore_index=0
                )
                loss = loss / gradient_accumulation_steps

            loss_accum += loss.item()

            # Backward pass
            scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        iter_num += 1
        running_loss = loss_accum * gradient_accumulation_steps

        # Logging
        if iter_num % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"Iter {iter_num:4d}/{max_iters} | Loss: {running_loss:.4f} | LR: {lr:.5f} | Time: {dt:.2f}s")

        # Salva Checkpoint
        if iter_num % save_interval == 0 or iter_num == max_iters:
            ckpt_path = f"checkpoints/model/luci_gpt_iter_{iter_num}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'iter': iter_num,
                'loss': running_loss
            }, ckpt_path)
            print(f"💾 Checkpoint salvo em {ckpt_path}")

    print("\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")


if __name__ == "__main__":
    train()
