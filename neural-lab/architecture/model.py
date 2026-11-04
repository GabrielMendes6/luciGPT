
import torch
import torch.nn as nn

from architecture.config import ModelConfig
from architecture.embeddings import InputEmbedding
from architecture.transformer_block import TransformerBlock
from architecture.kv_cache import KVCache


class LuciGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.input_embedding = InputEmbedding(config)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embedding_dim)

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # Compartilha os pesos entre embeddings e saída.
        self.lm_head.weight = (
            self.input_embedding.token_embedding.embedding.weight
        )

        self.initialize_weights()

    def initialize_weights(self):
        std = 0.02
        residual_std = std / (2 * self.config.num_layers) ** 0.5

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Inicialização das projeções residuais.
        for block in self.transformer_blocks:

            nn.init.normal_(
                block.attention.output_projection.weight,
                mean=0.0,
                std=residual_std,
            )

            nn.init.normal_(
                block.feed_forward.network[2].weight,
                mean=0.0,
                std=residual_std,
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        logits_to_keep: int | None = None,
    ):

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids precisa ter formato [batch, sequence]."
            )

        if token_ids.size(0) == 0 or token_ids.size(1) == 0:
            raise ValueError("O batch e a sequência não podem estar vazios.")

        if kv_cache is not None and not use_cache:
            raise ValueError("kv_cache exige use_cache=True.")

        if kv_cache is not None:

            if not isinstance(kv_cache, KVCache):
                raise TypeError("kv_cache precisa ser uma instância de KVCache.")

            if len(kv_cache) != self.config.num_layers:
                raise ValueError(
                    "Quantidade de camadas do KV Cache incompatível."
                )

            if kv_cache.max_length != self.config.max_seq_len:
                raise ValueError(
                    "Capacidade do KV Cache incompatível com o modelo."
                )

            if kv_cache.key_buffers[0].size(0) != token_ids.size(0):
                raise ValueError("Batch do KV Cache incompatível.")

            position_offset = kv_cache.length

        else:
            position_offset = 0

        total_length = position_offset + token_ids.size(1)

        if total_length > self.config.max_seq_len:
            raise ValueError(
                f"Sequência excede o limite de "
                f"{self.config.max_seq_len} tokens."
            )

        x = self.input_embedding(
            token_ids,
            position_offset=position_offset,
        )

        if use_cache:

            if kv_cache is None:
                kv_cache = KVCache.create(
                    config=self.config,
                    batch_size=token_ids.size(0),
                    device=x.device,
                    dtype=x.dtype,
                )

            if kv_cache.device != x.device:
                raise ValueError(
                    "KV Cache e modelo precisam estar no mesmo device."
                )

            if kv_cache.dtype != x.dtype:
                raise ValueError(
                    "KV Cache e entrada precisam ter o mesmo dtype."
                )

        for layer_index, block in enumerate(self.transformer_blocks):

            if use_cache:
                key_cache = kv_cache.key_buffers[layer_index]
                value_cache = kv_cache.value_buffers[layer_index]
            else:
                key_cache = None
                value_cache = None

            x = block(
                x,
                key_cache=key_cache,
                value_cache=value_cache,
                cache_position=position_offset,
            )

        if use_cache:
            kv_cache.advance(token_ids.size(1))

        x = self.final_norm(x)
        
        if logits_to_keep is not None:

            if logits_to_keep <= 0:
                raise ValueError(
                    "logits_to_keep precisa ser maior que zero."
                )

            if logits_to_keep > x.size(1):
                raise ValueError(
                    "logits_to_keep não pode exceder o tamanho da sequência."
                )

            x = x[:, -logits_to_keep:, :]

        logits = self.lm_head(x)

        if use_cache:
            return logits, kv_cache

        return logits
