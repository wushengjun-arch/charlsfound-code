"""Small Transformer encoder for CHARLS person-wave masked value modeling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CharlsFoundLiteConfig:
    value_vocab_size: int
    trait_vocab_size: int
    module_vocab_size: int
    wave_vocab_size: int
    max_sequence_length: int
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1


class CharlsFoundLite(nn.Module):
    """Transformer encoder over fixed-order CHARLS trait/value tokens."""

    def __init__(self, config: CharlsFoundLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.value_embedding = nn.Embedding(
            config.value_vocab_size, config.hidden_size, padding_idx=0
        )
        self.trait_embedding = nn.Embedding(config.trait_vocab_size, config.hidden_size)
        self.module_embedding = nn.Embedding(config.module_vocab_size, config.hidden_size)
        self.wave_embedding = nn.Embedding(config.wave_vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.input_norm = nn.LayerNorm(config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.value_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def encode(
        self,
        value_ids: torch.Tensor,
        trait_ids: torch.Tensor,
        module_ids: torch.Tensor,
        wave_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = value_ids.shape
        positions = (
            torch.arange(sequence_length, device=value_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        wave = self.wave_embedding(wave_token_ids).unsqueeze(1)
        token_embeddings = (
            self.value_embedding(value_ids)
            + self.trait_embedding(trait_ids)
            + self.module_embedding(module_ids)
            + self.position_embedding(positions)
            + wave
        )
        token_embeddings = self.input_norm(token_embeddings)
        cls = self.cls_token.expand(batch_size, -1, -1) + wave
        encoded = self.encoder(torch.cat([cls, token_embeddings], dim=1))
        encoded = self.output_norm(encoded)
        return encoded[:, 0], encoded[:, 1:]

    def forward(
        self,
        value_ids: torch.Tensor,
        trait_ids: torch.Tensor,
        module_ids: torch.Tensor,
        wave_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, token_hidden = self.encode(value_ids, trait_ids, module_ids, wave_token_ids)
        return self.lm_head(token_hidden)
