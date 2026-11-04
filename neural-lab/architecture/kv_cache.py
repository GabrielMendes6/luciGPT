
from dataclasses import dataclass

import torch

from architecture.config import ModelConfig


@dataclass
class KVCache:
    key_buffers: tuple[torch.Tensor, ...]
    value_buffers: tuple[torch.Tensor, ...]
    length: int
    max_length: int

    @classmethod
    def create(
        cls,
        config: ModelConfig,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":

        if batch_size <= 0:
            raise ValueError("batch_size precisa ser positivo.")

        key_buffers = []
        value_buffers = []

        for _ in range(config.num_layers):
            shape = (
                batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
            )

            key_buffers.append(
                torch.empty(shape, device=device, dtype=dtype)
            )

            value_buffers.append(
                torch.empty(shape, device=device, dtype=dtype)
            )

        return cls(
            key_buffers=tuple(key_buffers),
            value_buffers=tuple(value_buffers),
            length=0,
            max_length=config.max_seq_len,
        )

    def __post_init__(self):
        if len(self.key_buffers) != len(self.value_buffers):
            raise ValueError("Quantidade de buffers K/V incompatível.")

        if not self.key_buffers:
            raise ValueError("O cache precisa ter pelo menos uma camada.")

        if not 0 <= self.length <= self.max_length:
            raise ValueError("Comprimento inválido do KV Cache.")

    def __len__(self):
        return len(self.key_buffers)

    def __getitem__(
        self,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_layer(layer_index)

    def get_layer_buffers(
        self,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return (
            self.key_buffers[layer_index],
            self.value_buffers[layer_index],
        )

    def get_layer(
        self,
        layer_index: int,
        length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        visible_length = self.length if length is None else length

        if not 0 <= visible_length <= self.max_length:
            raise ValueError("Comprimento solicitado inválido.")

        key, value = self.get_layer_buffers(layer_index)

        return (
            key[:, :, :visible_length, :],
            value[:, :, :visible_length, :],
        )

    def write_layer(
        self,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start: int,
    ) -> None:

        if key.shape != value.shape:
            raise ValueError("Cache K/V possui dimensões incompatíveis.")

        if key.ndim != 4:
            raise ValueError("K/V precisam ter quatro dimensões.")

        if start != self.length:
            raise ValueError("Posição de escrita diferente do comprimento atual.")

        end = start + key.size(2)

        if end > self.max_length:
            raise ValueError("Cache excede o contexto máximo.")

        key_buffer, value_buffer = self.get_layer_buffers(layer_index)

        if key.shape[:2] != key_buffer.shape[:2]:
            raise ValueError("Dimensões de batch ou cabeças incompatíveis.")

        if key.size(3) != key_buffer.size(3):
            raise ValueError("Dimensão das cabeças incompatível.")

        if key.device != key_buffer.device or value.device != value_buffer.device:
            raise ValueError("Cache e entrada precisam estar no mesmo device.")

        if key.dtype != key_buffer.dtype or value.dtype != value_buffer.dtype:
            raise ValueError("Cache e entrada precisam ter o mesmo dtype.")

        key_buffer[:, :, start:end, :].copy_(key)
        value_buffer[:, :, start:end, :].copy_(value)

    def advance(self, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length precisa ser positivo.")

        new_length = self.length + sequence_length

        if new_length > self.max_length:
            raise ValueError("Cache excede o contexto máximo.")

        self.length = new_length

    @property
    def device(self):
        return self.key_buffers[0].device

    @property
    def dtype(self):
        return self.key_buffers[0].dtype

    def reset(self):
        self.length = 0
