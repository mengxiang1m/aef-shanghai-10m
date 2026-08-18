from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aef.config import load_config, target_specs
from aef.data import AlignedRasterDataset, load_stats
from aef.model import AEFLite
from aef.training import (
    make_loader,
    make_scheduler,
    run_epoch,
    sample_dataset,
    save_checkpoint,
    seed_everything,
    write_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AEF-Lite with multi-source reconstruction")
    parser.add_argument("--config", default="configs/shanghai_10m.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override total epoch count")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))
    stats = load_stats(config["data"]["stats_file"])
    train_set = AlignedRasterDataset(config, "train", stats)
    val_set = AlignedRasterDataset(config, "val", stats)
    if not train_set:
        raise RuntimeError("Training split is empty; reduce split_block_size or inspect split ratios")

    train_config = config["train"]
    if args.output_dir is not None:
        train_config["output_dir"] = str(Path(args.output_dir).resolve())
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    workers = int(config["data"].get("num_workers", 0))
    train_loader = make_loader(
        train_set, int(train_config["batch_size"]), workers, True, int(config.get("seed", 0))
    )
    val_samples = (
        args.max_val_batches * int(train_config["batch_size"])
        if args.max_val_batches is not None
        else None
    )
    val_view = sample_dataset(val_set, val_samples, int(config.get("seed", 0)) + 1)
    val_loader = (
        make_loader(val_view, int(train_config["batch_size"]), workers, False)
        if val_set
        else None
    )
    device = torch.device(args.device)
    model = AEFLite(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_config["learning_rate"]), weight_decay=float(train_config["weight_decay"])
    )
    scheduler = make_scheduler(
        optimizer, int(train_config["epochs"]), int(train_config.get("warmup_epochs", 0))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(train_config.get("amp", True)) and device.type == "cuda")
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = target_specs(config)
    history: list[dict] = []
    best = float("inf")
    start_epoch = 0
    resume_path = None
    if args.resume:
        resume_path = output_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history = [record for record in history if int(record["epoch"]) <= start_epoch]
        if history:
            best = min(float(record["val"]["loss"]) for record in history)
        print(f"resumed {resume_path} at epoch {start_epoch}")
    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_metrics = run_epoch(
            model,
            train_loader,
            specs,
            device,
            optimizer=optimizer,
            scaler=scaler,
            uniformity_weight=float(train_config.get("uniformity_weight", 0.0)),
            grad_clip=float(train_config.get("grad_clip", 0.0)),
            amp=bool(train_config.get("amp", True)),
            consistency_weight=float(train_config.get("consistency_weight", 0.0)),
            source_drop_probability=float(
                train_config.get("source_drop_probability", 0.3)
            ),
            max_batches=args.max_train_batches,
        )
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                specs,
                device,
                amp=bool(train_config.get("amp", True)),
                max_batches=None,
            )
            if val_loader is not None
            else train_metrics
        )
        scheduler.step()
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(record)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch + 1,
            config,
            stats,
            val_metrics,
            "pretrain",
            scheduler,
            scaler,
        )
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch + 1,
                config,
                stats,
                val_metrics,
                "pretrain",
                scheduler,
                scaler,
            )
        write_history(output_dir / "history.json", history)


if __name__ == "__main__":
    main()
