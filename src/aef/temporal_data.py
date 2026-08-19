from __future__ import annotations

import calendar
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import RasterSource, WindowIndex, _positions, _split_for, apply_transform


S1_MONTHS = tuple(f"2024{month:02d}" for month in range(1, 13))
S2_MONTHS = ("202403", "202408", "202411")


def month_day_offset(month: str) -> float:
    year = int(month[:4])
    number = int(month[4:])
    return float((date(year, number, 1) - date(year, 1, 1)).days)


@dataclass(frozen=True)
class CanonicalGrid:
    width: int
    height: int
    crs: Any
    transform: Any


class GeoWindowSource:
    """Raster window reader with an optional explicit WarpedVRT to a canonical grid."""

    def __init__(
        self,
        path: Path,
        channels: int,
        grid: CanonicalGrid,
        resampling: str,
    ):
        import rasterio

        self.path = path
        self.channels = channels
        self.grid = grid
        self.resampling = resampling
        self._source = None
        self._reader = None
        with rasterio.open(path) as dataset:
            if dataset.count != channels:
                raise ValueError(f"{path}: expected {channels} bands, got {dataset.count}")
            self.nodata = dataset.nodata
            self.aligned = (
                dataset.width == grid.width
                and dataset.height == grid.height
                and dataset.crs == grid.crs
                and dataset.transform == grid.transform
            )

    def _open(self) -> None:
        if self._reader is not None:
            return
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT

        self._source = rasterio.open(self.path)
        if self.aligned:
            self._reader = self._source
        else:
            self._reader = WarpedVRT(
                self._source,
                crs=self.grid.crs,
                transform=self.grid.transform,
                width=self.grid.width,
                height=self.grid.height,
                src_nodata=self.nodata,
                nodata=self.nodata,
                resampling=getattr(Resampling, self.resampling),
            )

    def read(self, window: WindowIndex, size: int) -> np.ndarray:
        from rasterio.windows import Window

        self._open()
        return self._reader.read(window=Window(window.x, window.y, size, size)).astype(np.float32)

    def close(self) -> None:
        if self._reader is not None and self._reader is not self._source:
            self._reader.close()
        if self._source is not None:
            self._source.close()
        self._reader = None
        self._source = None


def _normalize(
    raw: np.ndarray,
    transform: str,
    stats: dict[str, list[float]],
    valid: np.ndarray,
) -> np.ndarray:
    array = apply_transform(raw, transform)
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    array = np.clip((array - mean) / np.maximum(std, 1e-6), -6.0, 6.0)
    return np.where(valid[None], array, 0.0).astype(np.float32)


class TemporalShanghaiDataset(Dataset):
    """Windowed S1/S2 sequences with source-wise held-out reconstruction frames."""

    def __init__(self, config: dict[str, Any], split: str, stats: dict[str, Any]):
        import rasterio

        self.config = config
        self.split = split
        data = config["temporal_data"]
        self.temporal_root = Path(data["temporal_root"])
        self.static_root = Path(data["static_root"])
        self.patch_size = int(data["patch_size"])
        self.base_seed = int(config.get("seed", 0))
        self.epoch = 0
        self.stats = stats

        canonical_path = self.static_root / data.get("canonical_raster", "S2.tif")
        with rasterio.open(canonical_path) as canonical:
            self.grid = CanonicalGrid(
                canonical.width, canonical.height, canonical.crs, canonical.transform
            )

        self.source_months = {"s1": S1_MONTHS, "s2": S2_MONTHS}
        self.source_times = {
            name: np.asarray([month_day_offset(month) for month in months], dtype=np.float32)
            for name, months in self.source_months.items()
        }
        self.frames = {
            "s1": [
                GeoWindowSource(
                    self.temporal_root / f"s1_{month}_asc171_vv_vh_angle_utm51_10m.tif",
                    3,
                    self.grid,
                    "bilinear",
                )
                for month in S1_MONTHS
            ],
            "s2": [
                GeoWindowSource(
                    self.temporal_root / f"s2_{month}_raw_utm51_10m.tif",
                    4,
                    self.grid,
                    "bilinear",
                )
                for month in S2_MONTHS
            ],
        }
        self.valid_flags = {
            "s1": [
                GeoWindowSource(
                    self.temporal_root / f"s1_valid_flag_{month}_asc171_10m.tif",
                    1,
                    self.grid,
                    "nearest",
                )
                for month in S1_MONTHS
            ],
            "s2": [
                GeoWindowSource(
                    self.temporal_root / f"s2_valid_flag_{month}_10m.tif",
                    1,
                    self.grid,
                    "nearest",
                )
                for month in S2_MONTHS
            ],
        }
        self.cloud_flags = [
            GeoWindowSource(
                self.temporal_root / f"s2_cloud_suspect_flag_{month}_10m.tif",
                1,
                self.grid,
                "nearest",
            )
            for month in S2_MONTHS
        ]

        self.static_specs = data["static_targets"]
        self.static_sources = {
            name: RasterSource(self.static_root / spec["path"], int(spec["channels"]))
            for name, spec in self.static_specs.items()
        }
        for name, source in self.static_sources.items():
            if (
                source.width != self.grid.width
                or source.height != self.grid.height
                or source.crs != self.grid.crs
                or source.transform != self.grid.transform
            ):
                raise ValueError(f"Static target {name} is not on the canonical S2 grid")

        ratios = list(data["split_ratios"])
        stride = int(data.get("stride", self.patch_size))
        block = int(data["split_block_size"])
        self.windows = [
            WindowIndex(y, x)
            for y in _positions(self.grid.height, self.patch_size, stride)
            for x in _positions(self.grid.width, self.patch_size, stride)
            if _split_for(y, x, block, self.base_seed, ratios) == split
        ]
        required_s2 = int(data.get("require_s2_frames", 0))
        if required_s2 > 0:
            self.windows = self._filter_s2_center_coverage(self.windows, required_s2, stride)

    def _filter_s2_center_coverage(
        self, windows: list[WindowIndex], required_frames: int, sampling_stride: int
    ) -> list[WindowIndex]:
        """Remove empty tile-edge windows using a cheap center-pixel coverage proxy."""

        import rasterio
        from rasterio.enums import Resampling

        coarse_height = max(1, int(np.ceil(self.grid.height / sampling_stride)))
        coarse_width = max(1, int(np.ceil(self.grid.width / sampling_stride)))
        rows = np.asarray([window.y + self.patch_size // 2 for window in windows])
        cols = np.asarray([window.x + self.patch_size // 2 for window in windows])
        coarse_rows = np.minimum(rows * coarse_height // self.grid.height, coarse_height - 1)
        coarse_cols = np.minimum(cols * coarse_width // self.grid.width, coarse_width - 1)
        frame_count = np.zeros(len(windows), dtype=np.uint8)
        for valid_source, cloud_source in zip(self.valid_flags["s2"], self.cloud_flags):
            with rasterio.open(valid_source.path) as valid_dataset:
                valid = valid_dataset.read(
                    1,
                    out_shape=(coarse_height, coarse_width),
                    resampling=Resampling.nearest,
                )
            with rasterio.open(cloud_source.path) as cloud_dataset:
                cloud = cloud_dataset.read(
                    1,
                    out_shape=(coarse_height, coarse_width),
                    resampling=Resampling.nearest,
                )
            frame_count += ((valid[coarse_rows, coarse_cols] == 1) & (cloud[coarse_rows, coarse_cols] == 0))
        return [window for window, count in zip(windows, frame_count) if count >= required_frames]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.windows)

    def _rng(self, index: int, source: str) -> np.random.Generator:
        digest = hashlib.blake2b(
            f"{self.base_seed}:{self.epoch}:{index}:{source}".encode(), digest_size=8
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def _read_sequence(
        self, name: str, window: WindowIndex
    ) -> tuple[np.ndarray, np.ndarray]:
        arrays, masks = [], []
        for index, (frame, flag) in enumerate(zip(self.frames[name], self.valid_flags[name])):
            raw = frame.read(window, self.patch_size)
            valid = flag.read(window, self.patch_size)[0] == 1
            valid &= np.isfinite(raw).all(axis=0)
            if frame.nodata is not None and np.isfinite(frame.nodata):
                valid &= (raw != frame.nodata).all(axis=0)
            if name == "s2":
                valid &= self.cloud_flags[index].read(window, self.patch_size)[0] == 0
            arrays.append(
                _normalize(
                    raw,
                    "log1p" if name == "s2" else "identity",
                    self.stats[name],
                    valid,
                )
            )
            masks.append(valid)
        return np.stack(arrays), np.stack(masks)

    def _choose_held_out(self, valid: np.ndarray, index: int, source: str) -> int:
        candidates = np.flatnonzero(valid.reshape(valid.shape[0], -1).any(axis=1))
        if not len(candidates):
            candidates = np.arange(valid.shape[0])
        return int(self._rng(index, source).choice(candidates))

    def _static_target(self, name: str, window: WindowIndex) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.static_sources[name]
        raw = source.read(window.y, window.x, self.patch_size, self.patch_size)
        valid = np.isfinite(raw)
        if source.nodata is not None and np.isfinite(source.nodata):
            valid &= raw != source.nodata
        spec = self.static_specs[name]
        array = apply_transform(raw, spec.get("transform", "identity"))
        if spec.get("normalization", "standard") == "standard":
            mean = np.asarray(self.stats[name]["mean"], dtype=np.float32)[:, None, None]
            std = np.asarray(self.stats[name]["std"], dtype=np.float32)[:, None, None]
            array = np.clip((array - mean) / np.maximum(std, 1e-6), -6.0, 6.0)
        array = np.where(valid, array, 0).astype(np.float32)
        return torch.from_numpy(array), torch.from_numpy(valid)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        sources: dict[str, torch.Tensor] = {}
        source_valid: dict[str, torch.Tensor] = {}
        source_times: dict[str, torch.Tensor] = {}
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        target_times: dict[str, torch.Tensor] = {}

        for name in ("s1", "s2"):
            sequence, valid = self._read_sequence(name, window)
            held_out = self._choose_held_out(valid, index, name)
            targets[name] = torch.from_numpy(sequence[held_out].copy())
            masks[name] = torch.from_numpy(
                np.broadcast_to(valid[held_out][None], sequence[held_out].shape).copy()
            )
            target_times[name] = torch.tensor(self.source_times[name][held_out])
            sequence[held_out] = 0
            valid[held_out] = False
            sources[name] = torch.from_numpy(sequence)
            source_valid[name] = torch.from_numpy(valid)
            source_times[name] = torch.from_numpy(self.source_times[name].copy())

        for name in self.static_sources:
            targets[name], masks[name] = self._static_target(name, window)
            target_times[name] = torch.tensor(183.0)

        return {
            "sources": sources,
            "source_times": source_times,
            "source_valid": source_valid,
            "valid_period": torch.tensor([0.0, 366.0]),
            "target_times": target_times,
            "targets": targets,
            "masks": masks,
            "window": torch.tensor([window.y, window.x]),
        }

    def close(self) -> None:
        all_sources: Iterable[Any] = (
            [source for values in self.frames.values() for source in values]
            + [source for values in self.valid_flags.values() for source in values]
            + self.cloud_flags
            + list(self.static_sources.values())
        )
        for source in all_sources:
            source.close()


def temporal_worker_init(_: int) -> None:
    """Datasets are pickled before workers open GDAL handles; kept for explicitness."""
