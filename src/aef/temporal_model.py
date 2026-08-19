from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def temporal_pixel_tokens(features: torch.Tensor) -> torch.Tensor:
    """Map (B,T,H,W,C) to independent per-pixel temporal sequences.

    Keeping this as a named, tested operation prevents the unsafe direct ``view``
    that mixed time and spatial indices in the unofficial implementation.
    """

    if features.ndim != 5:
        raise ValueError(f"Expected (B,T,H,W,C), got {tuple(features.shape)}")
    return rearrange(features, "b t h w c -> (b h w) t c")


def restore_temporal_pixels(tokens: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[0] != batch * height * width:
        raise ValueError("Temporal token shape is incompatible with B,H,W")
    return rearrange(tokens, "(b h w) t c -> b t c h w", b=batch, h=height, w=width)


class SinusoidalTimeCode(nn.Module):
    """Stable sinusoidal encoding for float day offsets, not epoch milliseconds."""

    def __init__(self, dimension: int, min_period_days: float = 1.0, max_period_days: float = 3650.0):
        super().__init__()
        if dimension < 2:
            raise ValueError("Time-code dimension must be at least 2")
        half = dimension // 2
        periods = torch.logspace(
            math.log10(min_period_days), math.log10(max_period_days), steps=half
        )
        self.register_buffer("angular_frequency", 2.0 * math.pi / periods, persistent=False)
        self.dimension = dimension

    def forward(self, day_offsets: torch.Tensor) -> torch.Tensor:
        angles = day_offsets.to(torch.float32).unsqueeze(-1) * self.angular_frequency
        code = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if code.shape[-1] < self.dimension:
            code = F.pad(code, (0, self.dimension - code.shape[-1]))
        return code


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class SourceProjector(nn.Module):
    """Paper's H projector: source-specific mapping to the L/2 precision grid."""

    def __init__(self, input_channels: int, precision_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, precision_dim, 5, stride=2, padding=2),
            nn.GroupNorm(_groups(precision_dim), precision_dim),
            nn.GELU(),
            nn.Conv2d(precision_dim, precision_dim, 3, padding=1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected source frames (B,T,C,H,W), got {tuple(frames.shape)}")
        batch, time, channels, height, width = frames.shape
        projected = self.net(frames.reshape(batch * time, channels, height, width))
        return projected.reshape(batch, time, projected.shape[1], projected.shape[2], projected.shape[3])


def _safe_attention(
    attention: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    invalid: torch.Tensor,
) -> torch.Tensor:
    """Attention that stays finite for rows whose complete sequence is masked."""

    invalid = invalid.bool()
    all_invalid = invalid.all(dim=1)
    if bool(all_invalid.any()):
        invalid = invalid.clone()
        key = key.clone()
        value = value.clone()
        invalid[all_invalid, 0] = False
        key[all_invalid, 0] = 0
        value[all_invalid, 0] = 0
    output = attention(query, key, value, key_padding_mask=invalid, need_weights=False)[0]
    if bool(all_invalid.any()):
        output = output.masked_fill(all_invalid[:, None, None], 0)
    return output


class PrecisionOperator(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(dimension), dimension)
        self.net = nn.Sequential(
            nn.Conv2d(dimension, dimension * 2, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dimension * 2, dimension, 3, padding=1),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = features.shape
        x = features.reshape(batch * time, channels, height, width)
        x = x + self.net(self.norm(x))
        return x.reshape(batch, time, channels, height, width) * valid.unsqueeze(2)


class TimeOperator(nn.Module):
    """Time-axial attention independently at every L/8 spatial position."""

    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.time_code = SinusoidalTimeCode(dimension)
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, features: torch.Tensor, times: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = features.shape
        if times.shape != (batch, time) or valid.shape != (batch, time, height, width):
            raise ValueError("TimeOperator received inconsistent feature/time/mask shapes")
        x = temporal_pixel_tokens(rearrange(features, "b t c h w -> b t h w c"))
        time_code = self.time_code(times)
        time_code = rearrange(
            time_code[:, :, None, None, :].expand(-1, -1, height, width, -1),
            "b t h w c -> (b h w) t c",
        )
        valid_tokens = rearrange(valid, "b t h w -> (b h w) t")
        encoded = self.norm1(x + time_code)
        x = x + _safe_attention(self.attention, encoded, encoded, encoded, ~valid_tokens)
        x = x + self.mlp(self.norm2(x))
        x = x * valid_tokens.unsqueeze(-1)
        return restore_temporal_pixels(x, batch, height, width)


class SpaceOperator(nn.Module):
    """ViT-like spatial attention independently for every frame at L/16."""

    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = features.shape
        x = rearrange(features, "b t c h w -> (b t) (h w) c")
        valid_tokens = rearrange(valid, "b t h w -> (b t) (h w)")
        encoded = self.norm1(x)
        x = x + _safe_attention(self.attention, encoded, encoded, encoded, ~valid_tokens)
        x = x + self.mlp(self.norm2(x))
        x = x * valid_tokens.unsqueeze(-1)
        return rearrange(x, "(b t) (h w) c -> b t c h w", b=batch, t=time, h=height, w=width)


class LearnedExchange(nn.Module):
    """Learned pyramid exchange valid for every source/target scale ratio."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
            nn.GroupNorm(_groups(output_dim), output_dim),
        )

    def forward(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        batch, time, channels, height, width = features.shape
        x = features.reshape(batch * time, channels, height, width)
        if (height, width) != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.project(x)
        return x.reshape(batch, time, x.shape[1], size[0], size[1])


class STPBlock(nn.Module):
    PATHS = ("precision", "time", "space")

    def __init__(self, dimensions: Mapping[str, int], heads: Mapping[str, int], dropout: float):
        super().__init__()
        self.precision = PrecisionOperator(dimensions["precision"], dropout)
        self.time = TimeOperator(dimensions["time"], heads["time"], dropout)
        self.space = SpaceOperator(dimensions["space"], heads["space"], dropout)
        self.exchanges = nn.ModuleDict(
            {
                f"{source}_to_{target}": LearnedExchange(dimensions[source], dimensions[target])
                for source in self.PATHS
                for target in self.PATHS
            }
        )

    def forward(
        self,
        paths: dict[str, torch.Tensor],
        times: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        operated = {
            "precision": self.precision(paths["precision"], masks["precision"]),
            "time": self.time(paths["time"], times, masks["time"]),
            "space": self.space(paths["space"], masks["space"]),
        }
        exchanged: dict[str, torch.Tensor] = {}
        for target in self.PATHS:
            size = operated[target].shape[-2:]
            updates = [
                self.exchanges[f"{source}_to_{target}"](operated[source], size)
                for source in self.PATHS
            ]
            exchanged[target] = F.gelu(torch.stack(updates).sum(dim=0)) * masks[target].unsqueeze(2)
        return exchanged


def _resize_valid(valid: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, time, height, width = valid.shape
    resized = F.interpolate(
        valid.to(torch.float32).reshape(batch * time, 1, height, width),
        size=size,
        mode="nearest",
    )
    return resized.reshape(batch, time, size[0], size[1]).bool()


class STPEncoder(nn.Module):
    def __init__(
        self,
        source_channels: Mapping[str, int],
        precision_dim: int = 64,
        time_dim: int = 128,
        space_dim: int = 192,
        depth: int = 4,
        time_heads: int = 4,
        space_heads: int = 6,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.source_names = tuple(source_channels)
        self.projectors = nn.ModuleDict(
            {name: SourceProjector(channels, precision_dim) for name, channels in source_channels.items()}
        )
        self.source_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1, 1, precision_dim, 1, 1)) for name in source_channels}
        )
        self.to_time = LearnedExchange(precision_dim, time_dim)
        self.to_space = LearnedExchange(precision_dim, space_dim)
        dimensions = {"precision": precision_dim, "time": time_dim, "space": space_dim}
        heads = {"time": time_heads, "space": space_heads}
        self.blocks = nn.ModuleList([STPBlock(dimensions, heads, dropout) for _ in range(depth)])
        self.final_time = LearnedExchange(time_dim, precision_dim)
        self.final_space = LearnedExchange(space_dim, precision_dim)
        self.final_norm = nn.GroupNorm(_groups(precision_dim), precision_dim)

    def forward(
        self,
        sources: Mapping[str, torch.Tensor],
        source_times: Mapping[str, torch.Tensor],
        source_valid: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, times, validity = [], [], []
        spatial_shape: tuple[int, int] | None = None
        for name in self.source_names:
            frames = sources[name]
            batch, count, _, height, width = frames.shape
            if source_times[name].shape != (batch, count):
                raise ValueError(f"{name}: expected timestamps {(batch, count)}")
            valid = source_valid[name]
            if valid.shape == (batch, count):
                valid = valid[:, :, None, None].expand(batch, count, height, width)
            if valid.shape != (batch, count, height, width):
                raise ValueError(f"{name}: invalid mask shape {tuple(valid.shape)}")
            projected = self.projectors[name](frames) + self.source_embeddings[name]
            if spatial_shape is None:
                spatial_shape = projected.shape[-2:]
            elif projected.shape[-2:] != spatial_shape:
                raise ValueError("All source projectors must yield the same L/2 grid")
            features.append(projected)
            times.append(source_times[name].to(torch.float32))
            validity.append(_resize_valid(valid, projected.shape[-2:]))

        precision = torch.cat(features, dim=1)
        merged_times = torch.cat(times, dim=1)
        merged_valid = torch.cat(validity, dim=1)
        precision = precision * merged_valid.unsqueeze(2)
        precision_size = precision.shape[-2:]
        time_size = (max(1, precision_size[0] // 4), max(1, precision_size[1] // 4))
        space_size = (max(1, precision_size[0] // 8), max(1, precision_size[1] // 8))
        paths = {
            "precision": precision,
            "time": self.to_time(precision, time_size),
            "space": self.to_space(precision, space_size),
        }
        masks = {
            "precision": merged_valid,
            "time": _resize_valid(merged_valid, time_size),
            "space": _resize_valid(merged_valid, space_size),
        }
        for block in self.blocks:
            paths = block(paths, merged_times, masks)
        fused = paths["precision"]
        fused = fused + self.final_time(paths["time"], precision_size)
        fused = fused + self.final_space(paths["space"], precision_size)
        batch, time, channels, height, width = fused.shape
        fused = self.final_norm(fused.reshape(batch * time, channels, height, width))
        fused = fused.reshape(batch, time, channels, height, width) * merged_valid.unsqueeze(2)
        return fused, merged_times, merged_valid


class TemporalSummarizer(nn.Module):
    """Valid-period-conditioned, mask-safe time-axial attention pooling."""

    def __init__(self, dimension: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.time_code = SinusoidalTimeCode(dimension)
        self.period_query = nn.Sequential(
            nn.Linear(2 * dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.key_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        features: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        valid_period: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, channels, height, width = features.shape
        if valid_period.shape != (batch, 2):
            raise ValueError(f"Expected valid period (B,2), got {tuple(valid_period.shape)}")
        # The exact inverse pair of temporal_pixel_tokens; no direct view is allowed here.
        tokens = temporal_pixel_tokens(rearrange(features, "b t c h w -> b t h w c"))
        time_code = self.time_code(times)
        time_code = rearrange(
            time_code[:, :, None, None, :].expand(-1, -1, height, width, -1),
            "b t h w c -> (b h w) t c",
        )
        keys = self.key_norm(tokens + time_code)
        period_code = self.time_code(valid_period)
        query = self.period_query(period_code.flatten(1))
        query = rearrange(
            query[:, None, None, :].expand(-1, height, width, -1),
            "b h w c -> (b h w) 1 c",
        )
        valid_tokens = rearrange(valid, "b t h w -> (b h w) t")
        summary = _safe_attention(self.attention, query, keys, tokens, ~valid_tokens).squeeze(1)
        return rearrange(summary, "(b h w) c -> b c h w", b=batch, h=height, w=width)


class VMFMeanBottleneck(nn.Module):
    """Dense S^63 mean-direction bottleneck; stochastic sampling is optional."""

    def __init__(self, input_dim: int, embedding_dim: int = 64, concentration: float = 8000.0):
        super().__init__()
        self.mean = nn.Conv2d(input_dim, embedding_dim, 1)
        self.concentration = float(concentration)

    def mean_direction(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mean(features), dim=1, eps=1e-6)

    def sample(self, mean: torch.Tensor) -> torch.Tensor:
        """High-concentration tangent-normal approximation to a VMF sample."""

        noise = torch.randn_like(mean) / math.sqrt(self.concentration)
        noise = noise - (noise * mean).sum(dim=1, keepdim=True) * mean
        return F.normalize(mean + noise, dim=1, eps=1e-6)

    def forward(self, features: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mean = self.mean_direction(features)
        return self.sample(mean) if sample and self.training else mean


class ConditionalDecoder(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, hidden: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim + condition_dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, output_channels, 1),
        )

    def forward(self, embedding: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = condition[:, :, None, None].expand(-1, -1, embedding.shape[-2], embedding.shape[-1])
        return self.net(torch.cat((embedding, condition), dim=1))


@dataclass(frozen=True)
class TemporalModelConfig:
    precision_dim: int = 64
    time_dim: int = 128
    space_dim: int = 192
    depth: int = 4
    time_heads: int = 4
    space_heads: int = 6
    embedding_dim: int = 64
    decoder_width: int = 128
    dropout: float = 0.05
    concentration: float = 8000.0


class AEFTemporal(nn.Module):
    """Resource-scaled AEF topology with STP and valid-period summarization."""

    def __init__(
        self,
        source_channels: Mapping[str, int],
        target_channels: Mapping[str, int],
        config: TemporalModelConfig = TemporalModelConfig(),
    ):
        super().__init__()
        self.encoder = STPEncoder(
            source_channels,
            config.precision_dim,
            config.time_dim,
            config.space_dim,
            config.depth,
            config.time_heads,
            config.space_heads,
            config.dropout,
        )
        self.summarizer = TemporalSummarizer(
            config.precision_dim, config.time_heads, config.dropout
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(config.precision_dim, config.precision_dim, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.bottleneck = VMFMeanBottleneck(
            config.precision_dim, config.embedding_dim, config.concentration
        )
        condition_dim = 32
        # Decoder time is normalized to [0,1), unlike encoder time measured in days.
        self.decoder_time_code = SinusoidalTimeCode(
            condition_dim, min_period_days=1.0 / 365.0, max_period_days=1.0
        )
        self.decoders = nn.ModuleDict(
            {
                name: ConditionalDecoder(
                    config.embedding_dim, condition_dim, config.decoder_width, channels
                )
                for name, channels in target_channels.items()
            }
        )

    def encode(
        self,
        sources: Mapping[str, torch.Tensor],
        source_times: Mapping[str, torch.Tensor],
        source_valid: Mapping[str, torch.Tensor],
        valid_period: torch.Tensor,
        sample_bottleneck: bool = False,
    ) -> torch.Tensor:
        sequence, times, valid = self.encoder(sources, source_times, source_valid)
        summary = self.summarizer(sequence, times, valid, valid_period)
        return self.bottleneck(self.upsample(summary), sample=sample_bottleneck)

    def encode_mean_and_sample(
        self,
        sources: Mapping[str, torch.Tensor],
        source_times: Mapping[str, torch.Tensor],
        source_valid: Mapping[str, torch.Tensor],
        valid_period: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, times, valid = self.encoder(sources, source_times, source_valid)
        summary = self.summarizer(sequence, times, valid, valid_period)
        features = self.upsample(summary)
        mean = self.bottleneck.mean_direction(features)
        sample = self.bottleneck.sample(mean) if self.training else mean
        return mean, sample

    def decode(
        self,
        embedding: torch.Tensor,
        valid_period: torch.Tensor,
        target_times: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        duration = (valid_period[:, 1] - valid_period[:, 0]).clamp_min(1e-3)
        predictions = {}
        for name, decoder in self.decoders.items():
            target_time = target_times[name].to(torch.float32)
            relative_time = (target_time - valid_period[:, 0]) / duration
            predictions[name] = decoder(embedding, self.decoder_time_code(relative_time))
        return predictions

    def forward(
        self,
        sources: Mapping[str, torch.Tensor],
        source_times: Mapping[str, torch.Tensor],
        source_valid: Mapping[str, torch.Tensor],
        valid_period: torch.Tensor,
        target_times: Mapping[str, torch.Tensor],
    ) -> dict[str, object]:
        embedding, reconstruction_latent = self.encode_mean_and_sample(
            sources, source_times, source_valid, valid_period
        )
        return {
            "embedding": embedding,
            "reconstruction_latent": reconstruction_latent,
            "predictions": self.decode(reconstruction_latent, valid_period, target_times),
        }


class TemporalDownstreamModel(nn.Module):
    """Dense linear probes over valid-period-conditioned AEFTemporal mean embeddings."""

    def __init__(self, backbone: AEFTemporal, target_channels: Mapping[str, int]):
        super().__init__()
        self.backbone = backbone
        embedding_dim = backbone.bottleneck.mean.out_channels
        self.heads = nn.ModuleDict(
            {name: nn.Conv2d(embedding_dim, channels, 1) for name, channels in target_channels.items()}
        )
        self.backbone_frozen = False

    def set_backbone_frozen(self, frozen: bool) -> None:
        self.backbone_frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(
        self,
        sources: Mapping[str, torch.Tensor],
        source_times: Mapping[str, torch.Tensor],
        source_valid: Mapping[str, torch.Tensor],
        valid_period: torch.Tensor,
    ) -> dict[str, object]:
        if self.backbone_frozen:
            with torch.no_grad():
                embedding = self.backbone.encode(
                    sources, source_times, source_valid, valid_period
                )
        else:
            embedding = self.backbone.encode(sources, source_times, source_valid, valid_period)
        return {
            "embedding": embedding,
            "predictions": {name: head(embedding) for name, head in self.heads.items()},
        }
