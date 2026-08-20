from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from aef.config import load_config
from aef.data import invert_transform
from aef.temporal_data import TemporalShanghaiDataset
from aef.temporal_model import AEFTemporal, TemporalDownstreamModel, TemporalModelConfig
from train_temporal import make_temporal_loader, move_batch


def restore_scale(
    name: str,
    values: torch.Tensor,
    stats: dict[str, Any],
    spec: dict[str, Any],
) -> torch.Tensor:
    if spec.get("normalization", "standard") == "standard":
        mean = torch.as_tensor(
            stats[name]["mean"], dtype=values.dtype, device=values.device
        )[None, :, None, None]
        std = torch.as_tensor(
            stats[name]["std"], dtype=values.dtype, device=values.device
        )[None, :, None, None]
        values = values * std + mean
    restored = invert_transform(
        values.detach().float().cpu().numpy(), spec.get("transform", "identity")
    )
    return torch.from_numpy(restored).to(values.device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen AEFTemporal probes on the held-out spatial test split"
    )
    parser.add_argument("--config", default="configs/shanghai_stp_temporal.yaml")
    parser.add_argument(
        "--checkpoint", default="artifacts/stp_temporal_downstream/best.pt"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", default="artifacts/stp_temporal_downstream/test_metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "stp_temporal_downstream":
        raise ValueError(
            f"Expected stp_temporal_downstream checkpoint, got {checkpoint.get('kind')}"
        )

    settings = config["temporal_downstream"]
    target_names = list(settings["targets"])
    target_specs = {
        name: config["temporal_data"]["static_targets"][name] for name in target_names
    }
    all_specs = config["temporal_targets"]
    allowed = {field.name for field in fields(TemporalModelConfig)}
    model_config = TemporalModelConfig(
        **{key: value for key, value in config["temporal_model"].items() if key in allowed}
    )
    backbone = AEFTemporal(
        {"s1": 3, "s2": 4},
        {name: int(spec["channels"]) for name, spec in all_specs.items()},
        model_config,
    )
    model = TemporalDownstreamModel(
        backbone, {name: int(all_specs[name]["channels"]) for name in target_names}
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()

    stats = checkpoint["stats"]
    dataset = TemporalShanghaiDataset(config, "test", stats, hold_out_frames=False)
    loader = make_temporal_loader(
        dataset,
        args.batch_size,
        int(config["temporal_data"].get("num_workers", 0)),
        False,
        int(config.get("seed", 0)) + 2,
    )
    regression = {
        name: {"count": 0, "absolute": 0.0, "squared": 0.0, "sum": 0.0, "sum_squared": 0.0}
        for name in target_names
        if all_specs[name]["loss"] not in {"bce", "balanced_bce"}
    }
    classification = {
        name: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for name in target_names
        if all_specs[name]["loss"] in {"bce", "balanced_bce"}
    }

    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(tqdm(loader, desc="test-probe")):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move_batch(raw_batch, device)
            predictions = model(
                batch["sources"],
                batch["source_times"],
                batch["source_valid"],
                batch["valid_period"],
            )["predictions"]
            for name in target_names:
                mask = batch["masks"][name].bool()
                if not bool(mask.any()):
                    continue
                if name in classification:
                    predicted = torch.sigmoid(predictions[name]) >= 0.5
                    actual = batch["targets"][name] >= 0.5
                    counts = classification[name]
                    counts["tp"] += int((predicted & actual & mask).sum())
                    counts["tn"] += int((~predicted & ~actual & mask).sum())
                    counts["fp"] += int((predicted & ~actual & mask).sum())
                    counts["fn"] += int((~predicted & actual & mask).sum())
                    continue
                prediction = restore_scale(name, predictions[name], stats, target_specs[name])[mask]
                target = restore_scale(name, batch["targets"][name], stats, target_specs[name])[mask]
                error = prediction - target
                acc = regression[name]
                acc["count"] += target.numel()
                acc["absolute"] += float(error.abs().sum())
                acc["squared"] += float(error.square().sum())
                acc["sum"] += float(target.sum())
                acc["sum_squared"] += float(target.square().sum())

    results: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": "test",
        "metrics": {},
    }
    for name, acc in regression.items():
        count = int(acc["count"])
        if not count:
            continue
        denominator = acc["sum_squared"] - acc["sum"] ** 2 / count
        results["metrics"][name] = {
            "count": count,
            "mae": acc["absolute"] / count,
            "rmse": (acc["squared"] / count) ** 0.5,
            "r2": 1.0 - acc["squared"] / max(denominator, 1e-12),
        }
    for name, counts in classification.items():
        tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        results["metrics"][name] = {
            **counts,
            "pixel_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "balanced_accuracy": 0.5 * (recall + tn / max(tn + fp, 1)),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "iou": tp / max(tp + fp + fn, 1),
        }

    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    dataset.close()


if __name__ == "__main__":
    main()
