from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aef.config import load_config, target_specs
from aef.data import AlignedRasterDataset
from aef.model import AEFLite, DownstreamModel
from aef.training import (
    make_loader,
    run_epoch,
    sample_dataset,
    save_checkpoint,
    seed_everything,
    write_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train downstream pixel-wise probes on AEF embeddings")
    parser.add_argument("--config", default="configs/shanghai_10m.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))
    downstream = config["downstream"]
    if args.epochs is not None:
        downstream["epochs"] = args.epochs
    if args.batch_size is not None:
        downstream["batch_size"] = args.batch_size
    if args.output_dir is not None:
        downstream["output_dir"] = str(Path(args.output_dir).resolve())
    checkpoint_path = args.checkpoint or downstream["pretrained_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = checkpoint["stats"]
    backbone = AEFLite(config)
    backbone.load_state_dict(checkpoint["model_state"])
    target_names = list(downstream["targets"])
    model = DownstreamModel(backbone.encoder, config, target_names)
    model.set_encoder_frozen(bool(downstream.get("freeze_encoder", True)))
    device = torch.device(args.device)
    model.to(device)
    train_set = AlignedRasterDataset(config, "train", stats)
    val_set = AlignedRasterDataset(config, "val", stats)
    workers = int(config["data"].get("num_workers", 0))
    train_loader = make_loader(
        train_set,
        int(downstream["batch_size"]),
        workers,
        True,
        int(config.get("seed", 0)),
    )
    val_samples = (
        args.max_val_batches * int(downstream["batch_size"])
        if args.max_val_batches is not None
        else None
    )
    val_view = sample_dataset(val_set, val_samples, int(config.get("seed", 0)) + 1)
    val_loader = (
        make_loader(val_view, int(downstream["batch_size"]), workers, False)
        if val_set
        else None
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(downstream["learning_rate"]), weight_decay=float(downstream["weight_decay"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(downstream.get("amp", True)) and device.type == "cuda")
    output_dir = Path(downstream["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = target_specs(config, target_names)
    history: list[dict] = []
    best = float("inf")
    start_epoch = 0
    if args.resume:
        resume_path = output_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state"])
        if resume_checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history = [record for record in history if int(record["epoch"]) <= start_epoch]
        if history:
            best = min(float(record["val"]["loss"]) for record in history)
        print(f"resumed {resume_path} at epoch {start_epoch}")
    for epoch in range(start_epoch, int(downstream["epochs"])):
        train_metrics = run_epoch(
            model,
            train_loader,
            specs,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(downstream.get("amp", True)),
            max_batches=args.max_train_batches,
        )
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                specs,
                device,
                amp=bool(downstream.get("amp", True)),
                max_batches=None,
            )
            if val_loader is not None
            else train_metrics
        )
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(record)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, config, stats, val_metrics, "downstream")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1, config, stats, val_metrics, "downstream")
        write_history(output_dir / "history.json", history)


if __name__ == "__main__":
    main()
