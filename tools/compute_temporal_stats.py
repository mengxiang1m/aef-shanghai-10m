from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aef.config import load_config
from aef.data import apply_transform, load_stats
from aef.temporal_data import TemporalShanghaiDataset


class Moments:
    def __init__(self, channels: int):
        self.count = np.zeros(channels, dtype=np.int64)
        self.total = np.zeros(channels, dtype=np.float64)
        self.squares = np.zeros(channels, dtype=np.float64)

    def add(self, values: np.ndarray, valid: np.ndarray) -> None:
        for channel in range(values.shape[0]):
            selected = values[channel][valid]
            if selected.size:
                selected = selected.astype(np.float64)
                self.count[channel] += selected.size
                self.total[channel] += selected.sum()
                self.squares[channel] += np.square(selected).sum()

    def result(self) -> dict[str, list[float] | list[int]]:
        mean = self.total / np.maximum(self.count, 1)
        variance = self.squares / np.maximum(self.count, 1) - np.square(mean)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(variance, 1e-12)).tolist(),
            "count": self.count.tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate valid-pixel S1/S2 temporal statistics")
    parser.add_argument("--config", default="configs/shanghai_stp_temporal.yaml")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--output", default="artifacts/shanghai_temporal_stats.json")
    args = parser.parse_args()

    config = load_config(args.config)
    base_stats = load_stats(config["data"].get("base_stats_file", config["data"]["stats_file"]))
    dataset = TemporalShanghaiDataset(config, "train", base_stats)
    rng = np.random.default_rng(int(config.get("seed", 0)))
    count = min(args.windows, len(dataset))
    selected = rng.choice(len(dataset), size=count, replace=False)
    moments = {"s1": Moments(3), "s2": Moments(4)}

    try:
        for sample_index, dataset_index in enumerate(selected, start=1):
            window = dataset.windows[int(dataset_index)]
            for name in ("s1", "s2"):
                for frame_index, (frame, flag) in enumerate(
                    zip(dataset.frames[name], dataset.valid_flags[name])
                ):
                    raw = frame.read(window, dataset.patch_size)
                    valid = flag.read(window, dataset.patch_size)[0] == 1
                    valid &= np.isfinite(raw).all(axis=0)
                    if frame.nodata is not None and np.isfinite(frame.nodata):
                        valid &= (raw != frame.nodata).all(axis=0)
                    if name == "s2":
                        valid &= (
                            dataset.cloud_flags[frame_index].read(window, dataset.patch_size)[0]
                            == 0
                        )
                    transformed = apply_transform(
                        raw, "log1p" if name == "s2" else "identity"
                    )
                    moments[name].add(transformed, valid)
            if sample_index % 16 == 0 or sample_index == count:
                print(f"statistics windows: {sample_index}/{count}", flush=True)
    finally:
        dataset.close()

    output = dict(base_stats)
    output.update({name: value.result() for name, value in moments.items()})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path.resolve())
    print(json.dumps({name: output[name] for name in ("s1", "s2")}, indent=2))


if __name__ == "__main__":
    main()
