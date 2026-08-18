from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from .losses import batch_uniformity_loss, consistency_loss, reconstruction_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: ({name: value.to(device) for name, value in group.items()} if isinstance(group, dict) else group)
        for key, group in batch.items()
    }


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
        generator=generator,
    )


def sample_dataset(dataset, samples: int | None, seed: int = 0):
    """Return a reproducible spatially distributed subset for limited validation runs."""
    if samples is None or samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:samples].tolist()
    return Subset(dataset, indices)


def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def run_epoch(
    model,
    loader,
    layer_specs,
    device,
    optimizer=None,
    scaler=None,
    uniformity_weight: float = 0.0,
    grad_clip: float = 0.0,
    amp: bool = True,
    consistency_weight: float = 0.0,
    source_drop_probability: float = 0.3,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(
            tqdm(loader, leave=False, desc="train" if training else "validate")
        ):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                output = model(batch["inputs"])
                recon, parts = reconstruction_loss(
                    output["predictions"], batch["targets"], batch["masks"], layer_specs
                )
                uniformity = batch_uniformity_loss(output["embedding"])
                consistency = output["embedding"].new_zeros(())
                if training and consistency_weight > 0:
                    perturbed = {name: value.clone() for name, value in batch["inputs"].items()}
                    if "s1" in perturbed:
                        keep = (
                            torch.rand(
                                perturbed["s1"].shape[0], 1, 1, 1, device=device
                            )
                            >= source_drop_probability
                        )
                        perturbed["s1"] = perturbed["s1"] * keep
                    student = model(perturbed)["embedding"]
                    consistency = consistency_loss(output["embedding"], student)
                loss = (
                    recon
                    + uniformity_weight * uniformity
                    + consistency_weight * consistency
                )
            if training:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            batch_size = next(iter(batch["inputs"].values())).shape[0]
            metrics = {
                "loss": loss,
                "reconstruction": recon,
                "uniformity": uniformity,
                "consistency": consistency,
                **parts,
            }
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            count += batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    config: dict[str, Any],
    stats: dict[str, Any],
    metrics: dict[str, float],
    kind: str,
    scheduler=None,
    scaler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    torch.save(
        {
            "kind": kind,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "config": clean_config,
            "stats": stats,
            "metrics": metrics,
        },
        path.with_suffix(path.suffix + ".tmp"),
    )
    path.with_suffix(path.suffix + ".tmp").replace(path)


def write_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
