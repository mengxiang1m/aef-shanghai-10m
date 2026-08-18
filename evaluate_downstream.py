from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aef.config import load_config
from aef.data import AlignedRasterDataset, invert_transform
from aef.model import AEFLite, DownstreamModel
from aef.training import make_loader, move_batch


def restore_scale(name, values, stats, spec):
    if spec.get("normalization", "standard") == "standard":
        mean = torch.as_tensor(stats[name]["mean"], device=values.device)[None, :, None, None]
        std = torch.as_tensor(stats[name]["std"], device=values.device)[None, :, None, None]
        values = values * std + mean
    return torch.from_numpy(
        invert_transform(values.detach().cpu().numpy(), spec.get("transform", "identity"))
    ).to(values.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate downstream probes on the spatial test split")
    parser.add_argument("--config", default="configs/shanghai_10m.yaml")
    parser.add_argument("--checkpoint", default="artifacts/downstream/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = checkpoint["stats"]
    targets = list(config["downstream"]["targets"])
    backbone = AEFLite(config)
    model = DownstreamModel(backbone.encoder, config, targets)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = AlignedRasterDataset(config, "test", stats)
    loader = make_loader(dataset, args.batch_size, int(config["data"].get("num_workers", 0)), False)

    accumulators = {
        name: {"n": 0, "absolute": 0.0, "squared": 0.0, "sum": 0.0, "sum_squared": 0.0}
        for name in targets
    }
    building_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move_batch(batch, device)
            predictions = model(batch["inputs"])["predictions"]
            for name in targets:
                mask = batch["masks"][name].bool()
                if not bool(mask.any()):
                    continue
                if config["data"]["layers"][name]["loss"] in {"bce", "balanced_bce"}:
                    predicted = torch.sigmoid(predictions[name]) >= 0.5
                    actual = batch["targets"][name] >= 0.5
                    building_counts["tp"] += int((predicted & actual & mask).sum())
                    building_counts["tn"] += int((~predicted & ~actual & mask).sum())
                    building_counts["fp"] += int((predicted & ~actual & mask).sum())
                    building_counts["fn"] += int((~predicted & actual & mask).sum())
                    continue
                prediction = restore_scale(
                    name, predictions[name], stats, config["data"]["layers"][name]
                )[mask]
                target = restore_scale(
                    name, batch["targets"][name], stats, config["data"]["layers"][name]
                )[mask]
                error = prediction - target
                acc = accumulators[name]
                acc["n"] += target.numel()
                acc["absolute"] += float(error.abs().sum())
                acc["squared"] += float(error.square().sum())
                acc["sum"] += float(target.sum())
                acc["sum_squared"] += float(target.square().sum())

    results = {}
    for name, acc in accumulators.items():
        if acc["n"]:
            denominator = acc["sum_squared"] - acc["sum"] ** 2 / acc["n"]
            results[name] = {
                "count": acc["n"],
                "mae": acc["absolute"] / acc["n"],
                "rmse": (acc["squared"] / acc["n"]) ** 0.5,
                "r2": 1.0 - acc["squared"] / max(denominator, 1e-12),
            }
    if sum(building_counts.values()):
        tp, tn, fp, fn = (building_counts[key] for key in ("tp", "tn", "fp", "fn"))
        results["building"] = {
            **building_counts,
            "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
            "iou": tp / max(tp + fp + fn, 1),
        }
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    dataset.close()


if __name__ == "__main__":
    main()
