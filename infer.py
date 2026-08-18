from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aef.config import input_specs, load_config, target_specs
from aef.data import RasterSource, apply_transform, invert_transform
from aef.model import AEFLite, DownstreamModel


class OutputWriter:
    def __init__(self, output_dir: Path, name: str, channels: int, reference: RasterSource):
        self.dataset = None
        self.array = None
        if reference.kind == "rasterio":
            import rasterio

            self.dataset = rasterio.open(
                output_dir / f"{name}.tif",
                "w",
                driver="GTiff",
                height=reference.height,
                width=reference.width,
                count=channels,
                dtype="float32",
                crs=reference.crs,
                transform=reference.transform,
                nodata=np.nan,
                tiled=True,
                compress="deflate",
                predictor=3,
            )
        else:
            self.array = np.lib.format.open_memmap(
                output_dir / f"{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(channels, reference.height, reference.width),
            )

    def write(self, array: np.ndarray, y: int, x: int) -> None:
        height, width = array.shape[-2:]
        if self.dataset is not None:
            from rasterio.windows import Window

            self.dataset.write(array.astype(np.float32), window=Window(x, y, width, height))
        else:
            self.array[:, y : y + height, x : x + width] = array

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.close()
        if self.array is not None:
            self.array.flush()


def normalize_input(
    raw: np.ndarray, name: str, spec: dict[str, Any], stats: dict[str, Any], nodata
) -> np.ndarray:
    valid = np.isfinite(raw)
    if nodata is not None and np.isfinite(nodata):
        valid &= raw != nodata
    array = apply_transform(raw, spec.get("transform", "identity"))
    if spec.get("normalization", "standard") == "standard":
        mean = np.asarray(stats[name]["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(stats[name]["std"], dtype=np.float32)[:, None, None]
        array = np.clip((array - mean) / np.maximum(std, 1e-6), -6.0, 6.0)
    return np.where(valid, array, 0.0).astype(np.float32)


def restore_prediction(
    array: np.ndarray, name: str, spec: dict[str, Any], stats: dict[str, Any]
) -> np.ndarray:
    if spec["loss"] == "bce":
        return (1.0 / (1.0 + np.exp(-array))).astype(np.float32)
    if spec.get("normalization", "standard") == "standard":
        mean = np.asarray(stats[name]["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(stats[name]["std"], dtype=np.float32)[:, None, None]
        array = array * std + mean
    return invert_transform(array, spec.get("transform", "identity")).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiled AEF-Lite inference")
    parser.add_argument("--config", default="configs/shanghai_10m.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/predictions")
    parser.add_argument("--embedding", action="store_true", help="also export the 64-band embedding")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = checkpoint.get("stats")
    if not stats:
        raise RuntimeError("Checkpoint has no normalization statistics; use a trained checkpoint")
    if checkpoint.get("kind") == "downstream":
        base = AEFLite(config)
        names = list(config["downstream"]["targets"])
        model = DownstreamModel(base.encoder, config, names)
        specs = target_specs(config, names)
    else:
        model = AEFLite(config)
        specs = target_specs(config)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()

    root = Path(config["data"]["root"])
    source_specs = input_specs(config)
    sources = {
        name: RasterSource(root / spec["path"], int(spec["channels"]))
        for name, spec in source_specs.items()
    }
    reference = next(iter(sources.values()))
    for name, source in sources.items():
        if (source.height, source.width, source.crs, source.transform) != (
            reference.height,
            reference.width,
            reference.crs,
            reference.transform,
        ):
            raise ValueError(f"Input grid mismatch for {name}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        name: OutputWriter(output_dir, name, int(spec["channels"]), reference)
        for name, spec in specs.items()
    }
    if args.embedding:
        writers["embedding"] = OutputWriter(
            output_dir, "embedding", int(config["model"]["embedding_dim"]), reference
        )
    patch = int(config["data"]["patch_size"])
    try:
        with torch.no_grad():
            for y in range(0, reference.height, patch):
                for x in range(0, reference.width, patch):
                    height = min(patch, reference.height - y)
                    width = min(patch, reference.width - x)
                    inputs = {}
                    for name, source in sources.items():
                        raw = source.read(y, x, height, width)
                        array = normalize_input(raw, name, source_specs[name], stats, source.nodata)
                        padded = np.zeros((array.shape[0], patch, patch), dtype=np.float32)
                        padded[:, :height, :width] = array
                        inputs[name] = torch.from_numpy(padded).unsqueeze(0).to(device)
                    output = model(inputs)
                    for name, prediction in output["predictions"].items():
                        array = prediction[0, :, :height, :width].cpu().numpy()
                        writers[name].write(restore_prediction(array, name, specs[name], stats), y, x)
                    if args.embedding:
                        embedding = output["embedding"][0, :, :height, :width].cpu().numpy()
                        writers["embedding"].write(embedding, y, x)
    finally:
        for writer in writers.values():
            writer.close()
        for source in sources.values():
            source.close()
    print(f"wrote predictions to {output_dir}")


if __name__ == "__main__":
    main()
