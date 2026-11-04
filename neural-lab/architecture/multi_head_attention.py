
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.config import ModelConfig


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len

        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        # Projeção QKV fundida.
        self.qkv_projection = nn.Linear(
            self.embedding_dim,
            self.q_dim + (2 * self.kv_dim),
        )

        self.output_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "A entrada deve ter formato [batch, sequence, embedding]."
            )

        batch_size, sequence_length, embedding_dim = x.shape

        if sequence_length == 0:
            raise ValueError("A sequência não pode estar vazia.")

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Dimensão esperada: {self.embedding_dim}; "
                f"recebida: {embedding_dim}"
            )

        if cache_position < 0:
            raise ValueError("cache_position não pode ser negativo.")

        end_position = cache_position + sequence_length

        if end_position > self.max_seq_len:
            raise ValueError("Sequência excede o contexto máximo.")

        if (key_cache is None) != (value_cache is None):
            raise ValueError(
                "key_cache e value_cache devem ser fornecidos juntos."
            )

        # Uma única operação linear produz Q, K e V.
        qkv = self.qkv_projection(x)

        q, k, v = torch.split(
            qkv,
            [self.q_dim, self.kv_dim, self.kv_dim],
            dim=-1,
        )

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_mask = None

        if key_cache is not None:

            expected_shape = (
                batch_size,
                self.num_kv_heads,
                self.max_seq_len,
                self.head_dim,
            )

            if key_cache.shape != expected_shape:
                raise ValueError(
                    f"Formato inválido para key_cache: "
                    f"{tuple(key_cache.shape)}; esperado: {expected_shape}"
                )

            if value_cache.shape != expected_shape:
                raise ValueError(
                    f"Formato inválido para value_cache: "
                    f"{tuple(value_cache.shape)}; esperado: {expected_shape}"
                )

            if key_cache.device != k.device or value_cache.device != v.device:
                raise ValueError(
                    "Cache e entrada precisam estar no mesmo device."
                )

            if key_cache.dtype != k.dtype or value_cache.dtype != v.dtype:
                raise ValueError(
                    "Cache e entrada precisam ter o mesmo dtype."
                )

            key_cache[:, :, cache_position:end_position, :].copy_(k)
            value_cache[:, :, cache_position:end_position, :].copy_(v)

            k = key_cache[:, :, :end_position, :]
            v = value_cache[:, :, :end_position, :]

            if cache_position == 0:
                is_causal = True

            elif sequence_length == 1:
                is_causal = False

            else:
                query_positions = torch.arange(
                    cache_position,
                    end_position,
                    device=x.device,
                )

                key_positions = torch.arange(
                    end_position,
                    device=x.device,
                )

                attention_mask = (
                    key_positions.unsqueeze(0)
                    <= query_positions.unsqueeze(1)
                )

                is_causal = False

        else:
            if cache_position != 0:
                raise ValueError(
                    "cache_position deve ser zero quando não há cache."
                )

            is_causal = True

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            enable_gqa=True,
        )

        attended = attended.transpose(1, 2).contiguous()

        attended = attended.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        return self.output_projection(attended)
