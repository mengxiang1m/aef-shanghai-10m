from __future__ import annotations

import torch

from aef.temporal_model import (
    AEFTemporal,
    TemporalModelConfig,
    TemporalDownstreamModel,
    TemporalSummarizer,
    VMFMeanBottleneck,
    restore_temporal_pixels,
    temporal_pixel_tokens,
)


def test_vmf_bottleneck_separates_mean_direction_from_training_sample():
    torch.manual_seed(11)
    bottleneck = VMFMeanBottleneck(8, embedding_dim=64, concentration=8000.0).train()
    features = torch.randn(2, 8, 4, 4)
    mean = bottleneck.mean_direction(features)
    sample = bottleneck.sample(mean)
    assert torch.allclose(torch.linalg.vector_norm(mean, dim=1), torch.ones(2, 4, 4), atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(sample, dim=1), torch.ones(2, 4, 4), atol=1e-5)
    assert not torch.equal(mean, sample)


def test_temporal_token_rearrange_never_mixes_pixels_and_time():
    batch, time, height, width, channels = 2, 3, 2, 3, 1
    values = torch.empty(batch, time, height, width, channels)
    for b in range(batch):
        for t in range(time):
            for h in range(height):
                for w in range(width):
                    values[b, t, h, w, 0] = 1000 * b + 100 * t + 10 * h + w

    tokens = temporal_pixel_tokens(values)
    assert tokens.shape == (batch * height * width, time, channels)
    for b in range(batch):
        for h in range(height):
            for w in range(width):
                row = b * height * width + h * width + w
                expected = torch.tensor([1000 * b + 100 * t + 10 * h + w for t in range(time)])
                assert torch.equal(tokens[row, :, 0], expected)

    restored = restore_temporal_pixels(tokens, batch, height, width)
    assert torch.equal(restored, values.permute(0, 1, 4, 2, 3))


def test_temporal_summarizer_ignores_masked_frames_and_handles_all_invalid():
    torch.manual_seed(3)
    summarizer = TemporalSummarizer(8, heads=2).eval()
    features = torch.randn(1, 3, 8, 2, 2)
    times = torch.tensor([[0.0, 31.0, 60.0]])
    valid = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    valid[:, 2] = False
    period = torch.tensor([[0.0, 365.0]])

    changed = features.clone()
    changed[:, 2] = 1_000_000.0
    first = summarizer(features, times, valid, period)
    second = summarizer(changed, times, valid, period)
    assert torch.allclose(first, second, atol=1e-5)

    empty = summarizer(features, times, torch.zeros_like(valid), period)
    assert torch.isfinite(empty).all()
    assert torch.equal(empty, torch.zeros_like(empty))


def test_stp_accepts_asynchronous_source_time_axes_and_returns_dense_embedding():
    torch.manual_seed(7)
    config = TemporalModelConfig(
        precision_dim=16,
        time_dim=24,
        space_dim=32,
        depth=2,
        time_heads=4,
        space_heads=4,
        embedding_dim=64,
        decoder_width=16,
        dropout=0.0,
    )
    model = AEFTemporal(
        source_channels={"s1": 3, "s2": 4},
        target_channels={"s1": 3, "s2": 4, "building": 1},
        config=config,
    ).eval()
    sources = {
        "s1": torch.randn(2, 5, 3, 32, 32),
        "s2": torch.randn(2, 2, 4, 32, 32),
    }
    times = {
        "s1": torch.tensor([[0, 31, 60, 91, 121], [0, 31, 60, 91, 121]]),
        "s2": torch.tensor([[60, 213], [60, 213]]),
    }
    valid = {
        "s1": torch.ones(2, 5, 32, 32, dtype=torch.bool),
        "s2": torch.ones(2, 2, 32, 32, dtype=torch.bool),
    }
    valid["s1"][:, 3] = False
    period = torch.tensor([[0.0, 366.0], [0.0, 366.0]])
    target_times = {
        "s1": torch.tensor([91.0, 91.0]),
        "s2": torch.tensor([213.0, 213.0]),
        "building": torch.tensor([183.0, 183.0]),
    }

    output = model(sources, times, valid, period, target_times)
    assert output["embedding"].shape == (2, 64, 32, 32)
    assert torch.allclose(
        torch.linalg.vector_norm(output["embedding"], dim=1),
        torch.ones(2, 32, 32),
        atol=1e-4,
    )
    assert output["predictions"]["s1"].shape == (2, 3, 32, 32)
    assert output["predictions"]["s2"].shape == (2, 4, 32, 32)
    assert output["predictions"]["building"].shape == (2, 1, 32, 32)

    probe = TemporalDownstreamModel(model, {"building": 1, "dem": 1})
    probe.set_backbone_frozen(True)
    probe.train()
    probe_output = probe(sources, times, valid, period)
    assert probe_output["predictions"]["building"].shape == (2, 1, 32, 32)
    assert not probe.backbone.training
    assert all(not parameter.requires_grad for parameter in probe.backbone.parameters())
