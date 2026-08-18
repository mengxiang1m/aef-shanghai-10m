from __future__ import annotations

import numpy as np
import torch

from aef.data import AlignedRasterDataset
from aef.losses import batch_uniformity_loss, consistency_loss, reconstruction_loss
from aef.model import AEFLite, DownstreamModel
from aef.training import sample_dataset


def make_config(root):
    layers = {
        "s2": {"path": "s2.npy", "channels": 5, "role": "input_target", "transform": "log1p", "normalization": "standard", "loss": "smooth_l1", "weight": 1.0},
        "s1": {"path": "s1.npy", "channels": 2, "role": "input_target", "transform": "identity", "normalization": "standard", "loss": "smooth_l1", "weight": 1.0},
        "building": {"path": "building.npy", "channels": 1, "role": "target", "transform": "identity", "normalization": "none", "loss": "bce", "weight": 1.0},
        "lst": {"path": "lst.npy", "channels": 1, "role": "target", "transform": "identity", "normalization": "standard", "loss": "smooth_l1", "weight": 1.0},
        "dem": {"path": "dem.npy", "channels": 1, "role": "target", "transform": "identity", "normalization": "standard", "loss": "smooth_l1", "weight": 1.0},
        "ndvi": {"path": "ndvi.npy", "channels": 1, "role": "target", "transform": "identity", "normalization": "standard", "loss": "smooth_l1", "weight": 1.0},
    }
    return {
        "seed": 1,
        "data": {"root": str(root), "patch_size": 32, "stride": 32, "split_block_size": 32, "split_ratios": [1.0, 0.0, 0.0], "layers": layers},
        "model": {"stem_width": 8, "width": 16, "embedding_dim": 64, "depth": 2, "decoder_width": 16, "dropout": 0.0},
    }


def test_data_model_loss_and_probe(tmp_path):
    rng = np.random.default_rng(0)
    channels = {"s2": 5, "s1": 2, "building": 1, "lst": 1, "dem": 1, "ndvi": 1}
    for name, count in channels.items():
        array = rng.random((count, 64, 64), dtype=np.float32)
        if name == "building":
            array = (array > 0.5).astype(np.float32)
        np.save(tmp_path / f"{name}.npy", array)
    config = make_config(tmp_path)
    stats = {
        name: {"mean": [0.0] * count, "std": [1.0] * count} for name, count in channels.items()
    }
    dataset = AlignedRasterDataset(config, "train", stats)
    sample = dataset[0]
    inputs = {name: tensor.unsqueeze(0) for name, tensor in sample["inputs"].items()}
    targets = {name: tensor.unsqueeze(0) for name, tensor in sample["targets"].items()}
    masks = {name: tensor.unsqueeze(0) for name, tensor in sample["masks"].items()}
    model = AEFLite(config)
    output = model(inputs)
    assert output["embedding"].shape == (1, 64, 32, 32)
    assert torch.allclose(torch.linalg.vector_norm(output["embedding"], dim=1), torch.ones(1, 32, 32), atol=1e-4)
    loss, parts = reconstruction_loss(output["predictions"], targets, masks, config["data"]["layers"])
    assert torch.isfinite(loss) and set(parts) == set(channels)
    probe = DownstreamModel(model.encoder, config, ["building", "lst", "dem", "ndvi"])
    assert set(probe(inputs)["predictions"]) == {"building", "lst", "dem", "ndvi"}
    probe.set_encoder_frozen(True)
    probe.train()
    assert not probe.encoder.training
    assert all(not parameter.requires_grad for parameter in probe.encoder.parameters())


def test_aef_objectives_handle_missing_targets_and_dense_uniformity():
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    missing = torch.zeros_like(prediction, dtype=torch.bool)
    valid = torch.ones_like(prediction, dtype=torch.bool)
    total, parts = reconstruction_loss(
        {"missing": prediction, "valid": prediction},
        {"missing": target, "valid": target},
        {"missing": missing, "valid": valid},
        {
            "missing": {"loss": "l1", "weight": 100.0},
            "valid": {"loss": "l1", "weight": 1.0},
        },
    )
    assert torch.allclose(total, torch.tensor(2.5))
    assert torch.allclose(parts["missing"], torch.tensor(0.0))
    total.backward()
    assert prediction.grad is not None

    embedding = torch.zeros(2, 2, 1, 2)
    embedding[0, 0] = 1.0
    embedding[1, 1, 0, 0] = 1.0
    embedding[1, 0, 0, 1] = 1.0
    assert torch.allclose(batch_uniformity_loss(embedding), torch.tensor(0.5))
    assert torch.allclose(consistency_loss(embedding, embedding), torch.tensor(0.0))


def test_balanced_bce_keeps_rare_positive_gradient():
    prediction = torch.zeros(1, 1, 1, 4, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss, _ = reconstruction_loss(
        {"building": prediction},
        {"building": target},
        {"building": mask},
        {"building": {"loss": "balanced_bce", "weight": 1.0}},
    )
    loss.backward()
    assert prediction.grad[0, 0, 0, 0].abs() > prediction.grad[0, 0, 0, 1].abs()


def test_limited_validation_subset_is_reproducible_and_distributed():
    dataset = list(range(100))
    first = sample_dataset(dataset, 10, seed=7)
    second = sample_dataset(dataset, 10, seed=7)
    assert first.indices == second.indices
    assert first.indices != list(range(10))
