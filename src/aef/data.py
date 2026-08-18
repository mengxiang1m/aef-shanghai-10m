from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import input_specs, target_specs


def apply_transform(array: np.ndarray, name: str) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if name == "identity":
        return array
    if name == "log1p":
        return np.log1p(np.maximum(array, 0.0))
    if name == "signed_log1p":
        return np.sign(array) * np.log1p(np.abs(array))
    raise ValueError(f"Unknown transform: {name}")


def invert_transform(array: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return array
    if name == "log1p":
        return np.expm1(array)
    if name == "signed_log1p":
        return np.sign(array) * np.expm1(np.abs(array))
    raise ValueError(f"Unknown transform: {name}")


class RasterSource:
    """Lazy CHW raster reader for GeoTIFF or NumPy arrays."""

    def __init__(self, path: Path, expected_channels: int):
        self.path = path
        self.expected_channels = expected_channels
        self._dataset = None
        self._array = None
        self.kind = "rasterio" if path.suffix.lower() in {".tif", ".tiff"} else "numpy"
        if self.kind == "rasterio":
            try:
                import rasterio
            except ImportError as error:
                raise ImportError(
                    "GeoTIFF input requires rasterio. Install with: pip install rasterio"
                ) from error
            with rasterio.open(path) as dataset:
                self.channels = dataset.count
                self.height = dataset.height
                self.width = dataset.width
                self.nodata = dataset.nodata
                self.crs = dataset.crs
                self.transform = dataset.transform
        else:
            array = np.load(path, mmap_mode="r")
            array = self._as_chw(array)
            self.channels, self.height, self.width = array.shape
            self.nodata = None
            self.crs = None
            self.transform = None

        if self.channels != expected_channels:
            raise ValueError(
                f"{path}: configured for {expected_channels} channels, found {self.channels}"
            )

    def _as_chw(self, array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return array[None]
        if array.ndim != 3:
            raise ValueError(f"{self.path}: expected a 2D/3D array, got {array.shape}")
        if array.shape[0] == self.expected_channels:
            return array
        if array.shape[-1] == self.expected_channels:
            return np.moveaxis(array, -1, 0)
        raise ValueError(f"{self.path}: cannot infer channel axis from {array.shape}")

    def _open(self) -> None:
        if self.kind == "rasterio" and self._dataset is None:
            import rasterio

            self._dataset = rasterio.open(self.path)
        elif self.kind == "numpy" and self._array is None:
            self._array = self._as_chw(np.load(self.path, mmap_mode="r"))

    def read(self, y: int, x: int, height: int, width: int) -> np.ndarray:
        self._open()
        if self.kind == "rasterio":
            from rasterio.windows import Window

            return self._dataset.read(window=Window(x, y, width, height)).astype(np.float32)
        return np.asarray(self._array[:, y : y + height, x : x + width], dtype=np.float32)

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None


@dataclass(frozen=True)
class WindowIndex:
    y: int
    x: int


def _positions(length: int, patch: int, stride: int) -> list[int]:
    if length < patch:
        raise ValueError(f"Raster dimension {length} is smaller than patch size {patch}")
    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _split_for(y: int, x: int, block: int, seed: int, ratios: list[float]) -> str:
    block_id = f"{y // block}:{x // block}:{seed}".encode()
    value = int.from_bytes(hashlib.blake2b(block_id, digest_size=8).digest(), "big") / 2**64
    train_cut = ratios[0]
    val_cut = ratios[0] + ratios[1]
    return "train" if value < train_cut else "val" if value < val_cut else "test"


class AlignedRasterDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str, stats: dict[str, Any] | None = None):
        self.config = config
        self.split = split
        self.layer_specs = config["data"]["layers"]
        root = Path(config["data"]["root"])
        self.sources = {
            name: RasterSource(root / spec["path"], int(spec["channels"]))
            for name, spec in self.layer_specs.items()
        }
        self._validate_alignment()
        first = next(iter(self.sources.values()))
        self.height, self.width = first.height, first.width
        data_config = config["data"]
        self.patch_size = int(data_config["patch_size"])
        stride = int(data_config.get("stride", self.patch_size))
        block = int(data_config["split_block_size"])
        ratios = list(data_config["split_ratios"])
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("data.split_ratios must sum to 1")
        seed = int(config.get("seed", 0))
        self.windows = [
            WindowIndex(y, x)
            for y in _positions(self.height, self.patch_size, stride)
            for x in _positions(self.width, self.patch_size, stride)
            if _split_for(y, x, block, seed, ratios) == split
        ]
        self.stats = stats or {}
        self.inputs = input_specs(config)
        self.targets = target_specs(config)

    def _validate_alignment(self) -> None:
        items = list(self.sources.items())
        ref_name, ref = items[0]
        for name, source in items[1:]:
            if (source.height, source.width) != (ref.height, ref.width):
                raise ValueError(
                    f"Raster shape mismatch: {ref_name}={(ref.height, ref.width)}, "
                    f"{name}={(source.height, source.width)}"
                )
            if ref.crs is not None and (source.crs != ref.crs or source.transform != ref.transform):
                raise ValueError(f"GeoTIFF grid mismatch between {ref_name} and {name}")

    def __len__(self) -> int:
        return len(self.windows)

    def _prepare(self, name: str, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spec = self.layer_specs[name]
        mask = np.isfinite(raw)
        source = self.sources[name]
        if source.nodata is not None and np.isfinite(source.nodata):
            mask &= raw != source.nodata
        array = apply_transform(raw, spec.get("transform", "identity"))
        normalization = spec.get("normalization", "standard")
        if normalization == "standard":
            if name not in self.stats:
                raise RuntimeError(f"Missing statistics for layer '{name}'")
            mean = np.asarray(self.stats[name]["mean"], dtype=np.float32)[:, None, None]
            std = np.asarray(self.stats[name]["std"], dtype=np.float32)[:, None, None]
            array = (array - mean) / np.maximum(std, 1e-6)
            array = np.clip(array, -6.0, 6.0)
        elif normalization != "none":
            raise ValueError(f"Unknown normalization: {normalization}")
        array = np.where(mask, array, 0.0).astype(np.float32)
        return array, mask.astype(np.bool_)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        prepared = {}
        masks = {}
        for name, source in self.sources.items():
            raw = source.read(window.y, window.x, self.patch_size, self.patch_size)
            prepared[name], masks[name] = self._prepare(name, raw)
        inputs = {name: torch.from_numpy(prepared[name]) for name in self.inputs}
        targets = {name: torch.from_numpy(prepared[name]) for name in self.targets}
        target_masks = {name: torch.from_numpy(masks[name]) for name in self.targets}
        return {
            "inputs": inputs,
            "targets": targets,
            "masks": target_masks,
            "window": torch.tensor([window.y, window.x]),
        }

    def close(self) -> None:
        for source in self.sources.values():
            source.close()


def load_stats(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_raw_training_windows(
    config: dict[str, Any], max_windows: int | None = None
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    dataset = AlignedRasterDataset(config, "train", stats={})
    try:
        windows = dataset.windows
        if max_windows is not None and len(windows) > max_windows:
            rng = np.random.default_rng(int(config.get("seed", 0)))
            selected = rng.choice(len(windows), size=max_windows, replace=False)
            windows = [windows[i] for i in selected]
        for window in windows:
            for name, source in dataset.sources.items():
                spec = dataset.layer_specs[name]
                raw = source.read(window.y, window.x, dataset.patch_size, dataset.patch_size)
                valid = np.isfinite(raw)
                if source.nodata is not None and np.isfinite(source.nodata):
                    valid &= raw != source.nodata
                yield name, apply_transform(raw, spec.get("transform", "identity")), valid
    finally:
        dataset.close()
