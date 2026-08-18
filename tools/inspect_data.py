from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aef.config import load_config
from aef.data import AlignedRasterDataset, iter_raw_training_windows


class RunningStats:
    def __init__(self, channels: int):
        self.count = np.zeros(channels, dtype=np.int64)
        self.sum = np.zeros(channels, dtype=np.float64)
        self.squared_sum = np.zeros(channels, dtype=np.float64)

    def update(self, array: np.ndarray, valid: np.ndarray) -> None:
        for channel in range(array.shape[0]):
            values = array[channel][valid[channel]].astype(np.float64)
            self.count[channel] += values.size
            self.sum[channel] += values.sum()
            self.squared_sum[channel] += np.square(values).sum()

    def result(self) -> dict[str, list[float]]:
        count = np.maximum(self.count, 1)
        mean = self.sum / count
        variance = np.maximum(self.squared_sum / count - np.square(mean), 1e-12)
        return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "count": self.count.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate aligned rasters and compute train-only statistics")
    parser.add_argument("--config", default="configs/shanghai_10m.yaml")
    parser.add_argument("--max-windows", type=int, default=2048)
    args = parser.parse_args()
    config = load_config(args.config)
    probe = AlignedRasterDataset(config, "train", stats={})
    print(f"grid={probe.height}x{probe.width}, patch={probe.patch_size}")
    for name, source in probe.sources.items():
        print(f"{name}: {source.channels} bands, {source.path}")
    for split in ("train", "val", "test"):
        dataset = AlignedRasterDataset(config, split, stats={})
        print(f"{split}: {len(dataset)} windows")
        dataset.close()
    probe.close()

    accumulators = {
        name: RunningStats(int(spec["channels"])) for name, spec in config["data"]["layers"].items()
    }
    for name, array, valid in iter_raw_training_windows(config, args.max_windows):
        accumulators[name].update(array, valid)
    stats = {name: accumulator.result() for name, accumulator in accumulators.items()}
    output = Path(config["data"]["stats_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

