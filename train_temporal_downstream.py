from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from aef.config import load_config
from aef.data import load_stats
from aef.losses import reconstruction_loss
from aef.temporal_data import TemporalShanghaiDataset
from aef.temporal_model import AEFTemporal, TemporalDownstreamModel, TemporalModelConfig
from aef.training import make_scheduler, save_checkpoint, seed_everything, write_history
from train_temporal import make_temporal_loader, move_batch


def run_epoch(
    model: TemporalDownstreamModel,
    loader,
    target_specs: dict[str, dict[str, Any]],
    device: torch.device,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, raw_batch in enumerate(
            tqdm(loader, leave=False, desc="probe" if training else "validate-probe")
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
                )
                loss, parts = reconstruction_loss(
                    output["predictions"], batch["targets"], batch["masks"], target_specs
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = batch["valid_period"].shape[0]
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * batch_size
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            examples += batch_size
    return {name: value / max(examples, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dense linear probes over AEFTemporal")
    parser.add_argument("--config", default="configs/shanghai_stp_temporal.yaml")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))
    settings = config["temporal_downstream"]
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.output_dir is not None:
        settings["output_dir"] = str(Path(args.output_dir).resolve())
    checkpoint_path = Path(args.checkpoint or settings["pretrained_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "stp_temporal":
        raise ValueError(f"Expected stp_temporal checkpoint, got {checkpoint.get('kind')}")

    stats = load_stats(config["data"]["stats_file"])
    train_set = TemporalShanghaiDataset(config, "train", stats, hold_out_frames=False)
    val_set = TemporalShanghaiDataset(config, "val", stats, hold_out_frames=False)
    workers = int(config["temporal_data"].get("num_workers", 0))
    batch_size = int(settings["batch_size"])
    train_loader = make_temporal_loader(
        train_set, batch_size, workers, True, int(config.get("seed", 0))
    )
    val_loader = make_temporal_loader(
        val_set, batch_size, workers, False, int(config.get("seed", 0)) + 1
    )

    allowed = {field.name for field in fields(TemporalModelConfig)}
    model_config = TemporalModelConfig(
        **{key: value for key, value in config["temporal_model"].items() if key in allowed}
    )
    all_specs = config["temporal_targets"]
    target_names = list(settings["targets"])
    target_specs = {name: all_specs[name] for name in target_names}
    backbone = AEFTemporal(
        {"s1": 3, "s2": 4},
        {name: int(spec["channels"]) for name, spec in all_specs.items()},
        model_config,
    )
    backbone.load_state_dict(checkpoint["model_state"])
    model = TemporalDownstreamModel(
        backbone, {name: int(target_specs[name]["channels"]) for name in target_names}
    )
    model.set_backbone_frozen(bool(settings.get("freeze_backbone", True)))
    device = torch.device(args.device)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, int(settings["epochs"]), 0)
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(settings.get("amp", True)) and device.type == "cuda"
    )
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best = float("inf")

    if args.resume:
        resume_path = output_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        if resume.get("scheduler_state"):
            scheduler.load_state_dict(resume["scheduler_state"])
        if resume.get("scaler_state"):
            scaler.load_state_dict(resume["scaler_state"])
        start_epoch = int(resume["epoch"])
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best = min(float(item["val"]["loss"]) for item in history)

    for epoch in range(start_epoch, int(settings["epochs"])):
        train_metrics = run_epoch(
            model,
            train_loader,
            target_specs,
            device,
            optimizer,
            scaler,
            bool(settings.get("amp", True)),
            args.max_train_batches,
        )
        val_limit = (
            args.max_val_batches
            if args.max_val_batches is not None
            else int(settings.get("validation_batches", 92))
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            target_specs,
            device,
            amp=bool(settings.get("amp", True)),
            max_batches=val_limit,
        )
        scheduler.step()
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, epoch + 1, config, stats,
            val_metrics, "stp_temporal_downstream", scheduler, scaler
        )
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch + 1, config, stats,
                val_metrics, "stp_temporal_downstream", scheduler, scaler
            )
        write_history(output_dir / "history.json", history)

    train_set.close()
    val_set.close()


if __name__ == "__main__":
    main()
