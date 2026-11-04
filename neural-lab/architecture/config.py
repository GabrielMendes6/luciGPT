"""
Configuração otimizada da LUCI para treinamento no Kaggle (GPU T4 / P100 - 16GB VRAM).
Configuração MEDIUM adaptada para o tokenizer BPE de 847 tokens (ou expansível para 2048/4096).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 847         # Ajustado para o tokenizer BPE real (luci_bpe_v2.json)
    embedding_dim: int = 768      # Padrão GPT-2 (excelente equilíbrio qualidade/performance)
    max_seq_len: int = 2048       # Contexto confortável para conversas e documentos

    num_heads: int = 12           # 12 cabeças de atenção (768 / 12 = 64 dim por cabeça)
    num_kv_heads: int = 4         # Grouped Query Attention (GQA) para eficiência em inferência
    num_layers: int = 12          # 12 blocos de Transformer (profundidade ideal para ~90M params)

    ffn_dim: int = 3072           # 4x embedding_dim (padrão SwiGLU / Feed-Forward)

    def __post_init__(self):
        if self.vocab_size < 256:
            raise ValueError("vocab_size deve ser >= 256")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser positivo")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len deve ser positivo")

        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Número de cabeças deve ser positivo")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim deve ser divisível por num_heads"
            )

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads deve ser divisível por num_kv_heads"
            )

        if self.num_layers <= 0:
            raise ValueError("num_layers deve ser positivo")

        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim deve ser positivo")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads

    @property
    def queries_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads
