from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aef.config import load_config
from aef.data import load_stats
from aef.losses import batch_uniformity_loss, consistency_loss, reconstruction_loss
from aef.temporal_data import TemporalShanghaiDataset
from aef.temporal_model import AEFTemporal, TemporalModelConfig
from aef.training import make_scheduler, save_checkpoint, seed_everything, write_history


def make_temporal_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    """Respawn workers each epoch so dataset.set_epoch changes held-out frames."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=shuffle and len(dataset) >= batch_size,
        generator=generator,
    )


def move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    return value.to(device) if torch.is_tensor(value) else value


def perturb_inputs(
    sources: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    probability: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    student_sources = {name: value.clone() for name, value in sources.items()}
    student_valid = {name: value.clone() for name, value in valid.items()}
    for name in student_sources:
        batch, time = student_sources[name].shape[:2]
        keep = torch.rand(batch, time, device=student_sources[name].device) >= probability
        # The held-out frame is already invalid. Keep at least one remaining valid frame per source.
        available = student_valid[name].flatten(2).any(dim=2)
        for row in range(batch):
            if not bool((keep[row] & available[row]).any()) and bool(available[row].any()):
                keep[row, torch.nonzero(available[row], as_tuple=False)[0, 0]] = True
        student_sources[name] *= keep[:, :, None, None, None]
        student_valid[name] &= keep[:, :, None, None]
    return student_sources, student_valid


def run_epoch(
    model: AEFTemporal,
    loader,
    target_specs: dict[str, dict[str, Any]],
    device: torch.device,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    uniformity_weight: float = 0.0,
    consistency_weight: float = 0.0,
    frame_drop_probability: float = 0.3,
    grad_clip: float = 1.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, raw_batch in enumerate(
            tqdm(loader, leave=False, desc="train-stp" if training else "validate-stp")
        ):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                output = model(
                    batch["sources"],
                    batch["source_times"],
                    batch["source_valid"],
                    batch["valid_period"],
                    batch["target_times"],
                )
                reconstruction, parts = reconstruction_loss(
                    output["predictions"], batch["targets"], batch["masks"], target_specs
                )
                uniformity = batch_uniformity_loss(output["embedding"])
                consistency = output["embedding"].new_zeros(())
                if training and consistency_weight > 0:
                    student_sources, student_valid = perturb_inputs(
                        batch["sources"], batch["source_valid"], frame_drop_probability
                    )
                    student = model.encode(
                        student_sources,
                        batch["source_times"],
                        student_valid,
                        batch["valid_period"],
                    )
                    consistency = consistency_loss(output["embedding"], student)
                loss = (
                    reconstruction
                    + uniformity_weight * uniformity
                    + consistency_weight * consistency
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            batch_size = batch["valid_period"].shape[0]
            metrics = {
                "loss": loss,
                "reconstruction": reconstruction,
                "uniformity": uniformity,
                "consistency": consistency,
                **parts,
            }
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            examples += batch_size
    return {name: value / max(examples, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Shanghai AEF STP temporal model")
    parser.add_argument("--config", default="configs/shanghai_stp_temporal.yaml")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))
    stats = load_stats(config["data"]["stats_file"])
    train_config = config["temporal_train"]
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.output_dir is not None:
        train_config["output_dir"] = str(Path(args.output_dir).resolve())

    train_set = TemporalShanghaiDataset(config, "train", stats)
    val_set = TemporalShanghaiDataset(config, "val", stats)
    workers = int(config["temporal_data"].get("num_workers", 0))
    train_loader = make_temporal_loader(
        train_set,
        int(train_config["batch_size"]),
        workers,
        True,
        int(config.get("seed", 0)),
    )
    val_loader = make_temporal_loader(
        val_set, int(train_config["batch_size"]), workers, False, int(config.get("seed", 0)) + 1
    )

    allowed = {field.name for field in fields(TemporalModelConfig)}
    model_config = TemporalModelConfig(
        **{key: value for key, value in config["temporal_model"].items() if key in allowed}
    )
    target_specs = config["temporal_targets"]
    model = AEFTemporal(
        {"s1": 3, "s2": 4},
        {name: int(spec["channels"]) for name, spec in target_specs.items()},
        model_config,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = make_scheduler(
        optimizer, int(train_config["epochs"]), int(train_config.get("warmup_epochs", 0))
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(train_config.get("amp", True)) and device.type == "cuda"
    )
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best = float("inf")

    if args.resume:
        resume_path = output_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best = min(float(item["val"]["loss"]) for item in history)

    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_set.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            target_specs,
            device,
            optimizer,
            scaler,
            bool(train_config.get("amp", True)),
            float(train_config.get("uniformity_weight", 0.0)),
            float(train_config.get("consistency_weight", 0.0)),
            float(train_config.get("frame_drop_probability", 0.3)),
            float(train_config.get("grad_clip", 1.0)),
            args.max_train_batches,
        )
        validation_limit = (
            args.max_val_batches
            if args.max_val_batches is not None
            else int(train_config.get("validation_batches", 64))
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            target_specs,
            device,
            amp=bool(train_config.get("amp", True)),
            max_batches=validation_limit,
        )
        scheduler.step()
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, epoch + 1, config, stats,
            val_metrics, "stp_temporal", scheduler, scaler
        )
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch + 1, config, stats,
                val_metrics, "stp_temporal", scheduler, scaler
            )
        write_history(output_dir / "history.json", history)

    train_set.close()
    val_set.close()


if __name__ == "__main__":
    main()
