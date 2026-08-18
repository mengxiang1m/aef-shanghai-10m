from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import input_specs, target_specs


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )


class MultiScaleBlock(nn.Module):
    """Precision path plus low-resolution context path, inspired by AEF STP blocks."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.precision = nn.Sequential(
            ConvNormAct(width, width),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.context = nn.Sequential(
            nn.AvgPool2d(4, 4),
            ConvNormAct(width, width * 2),
            nn.Conv2d(width * 2, width, 1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width, 1), nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        precision = self.precision(x)
        context = self.context(x)
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.gelu(self.norm(x + precision + self.gate(x) * context))


class Encoder(nn.Module):
    def __init__(self, inputs: dict[str, dict[str, Any]], model_config: dict[str, Any]):
        super().__init__()
        stem_width = int(model_config["stem_width"])
        width = int(model_config["width"])
        self.input_names = list(inputs)
        self.stems = nn.ModuleDict(
            {
                name: nn.Sequential(
                    ConvNormAct(int(spec["channels"]), stem_width),
                    ConvNormAct(stem_width, stem_width),
                )
                for name, spec in inputs.items()
            }
        )
        self.fuse = ConvNormAct(stem_width * len(inputs), width, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                MultiScaleBlock(width, float(model_config.get("dropout", 0.0)))
                for _ in range(int(model_config["depth"]))
            ]
        )
        self.embedding = nn.Conv2d(width, int(model_config["embedding_dim"]), 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self.stems[name](inputs[name]) for name in self.input_names]
        embedding = self.embedding(self.blocks(self.fuse(torch.cat(features, dim=1))))
        return F.normalize(embedding, dim=1, eps=1e-6)


class ImplicitDecoder(nn.Sequential):
    def __init__(self, embedding_dim: int, hidden: int, out_channels: int):
        super().__init__(
            nn.Conv2d(embedding_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )


class AEFLite(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        inputs = input_specs(config)
        targets = target_specs(config)
        model_config = config["model"]
        self.encoder = Encoder(inputs, model_config)
        embedding_dim = int(model_config["embedding_dim"])
        decoder_width = int(model_config["decoder_width"])
        self.decoders = nn.ModuleDict(
            {
                name: ImplicitDecoder(embedding_dim, decoder_width, int(spec["channels"]))
                for name, spec in targets.items()
            }
        )

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: decoder(embedding) for name, decoder in self.decoders.items()}

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        embedding = self.encode(inputs)
        return {"embedding": embedding, "predictions": self.decode(embedding)}


class DownstreamModel(nn.Module):
    """Per-pixel linear probes (or fine-tuning heads) over pretrained embeddings."""

    def __init__(self, encoder: Encoder, config: dict[str, Any], target_names: list[str]):
        super().__init__()
        self.encoder = encoder
        self.encoder_frozen = False
        embedding_dim = int(config["model"]["embedding_dim"])
        specs = target_specs(config, target_names)
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(embedding_dim, int(spec["channels"]), 1)
                for name, spec in specs.items()
            }
        )

    def set_encoder_frozen(self, frozen: bool) -> None:
        self.encoder_frozen = frozen
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        embedding = self.encoder(inputs)
        return {
            "embedding": embedding,
            "predictions": {name: head(embedding) for name, head in self.heads.items()},
        }
