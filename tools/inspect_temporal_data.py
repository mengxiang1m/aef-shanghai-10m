from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


S1_MONTHS = tuple(f"2024{month:02d}" for month in range(1, 13))
S2_MONTHS = ("202403", "202408", "202411")


def expected_files() -> set[str]:
    names: set[str] = set()
    for month in S1_MONTHS:
        names.update(
            {
                f"s1_{month}_asc171_vv_vh_angle_utm51_10m.tif",
                f"s1_valid_flag_{month}_asc171_10m.tif",
                f"s1_metadata_{month}_asc171_10m.csv",
            }
        )
    for month in S2_MONTHS:
        names.update(
            {
                f"s2_{month}_raw_utm51_10m.tif",
                f"s2_valid_flag_{month}_10m.tif",
                f"s2_cloud_suspect_flag_{month}_10m.tif",
                f"s2_metadata_{month}_10m.csv",
            }
        )
    return names


def expected_channels(name: str) -> int:
    if re.fullmatch(r"s1_\d{6}_asc171_vv_vh_angle_utm51_10m\.tif", name):
        return 3
    if re.fullmatch(r"s2_\d{6}_raw_utm51_10m\.tif", name):
        return 4
    return 1


def sample_windows(width: int, height: int) -> list[Window]:
    size = min(32, width, height)
    return [
        Window(0, 0, size, size),
        Window((width - size) // 2, (height - size) // 2, size, size),
        Window(width - size, height - size, size, size),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate downloaded multi-temporal S1/S2 rasters")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    actual = {path.name for path in root.iterdir() if path.is_file()}
    expected = expected_files()
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise RuntimeError(f"Missing files: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected files: {unexpected}")

    reference_grids: dict[str, tuple[int, int, object, object]] = {}
    raster_summaries: list[str] = []
    for path in sorted(root.glob("*.tif")):
        with rasterio.open(path) as dataset:
            channels = expected_channels(path.name)
            if dataset.count != channels:
                raise RuntimeError(f"{path.name}: expected {channels} bands, got {dataset.count}")
            if dataset.crs is None or dataset.crs.to_epsg() != 32651:
                raise RuntimeError(f"{path.name}: expected EPSG:32651, got {dataset.crs}")
            source = "s1" if path.name.startswith("s1_") else "s2"
            grid = (dataset.width, dataset.height, dataset.transform, dataset.crs)
            if source not in reference_grids:
                reference_grids[source] = grid
            elif grid != reference_grids[source]:
                raise RuntimeError(f"{path.name}: grid differs within source {source}")

            finite_values = 0
            sample_min = float("inf")
            sample_max = float("-inf")
            for window in sample_windows(dataset.width, dataset.height):
                values = dataset.read(window=window)
                finite_mask = np.isfinite(values)
                if dataset.nodata is not None:
                    finite_mask &= values != dataset.nodata
                finite = values[finite_mask]
                finite_values += int(finite.size)
                if finite.size:
                    sample_min = min(sample_min, float(finite.min()))
                    sample_max = max(sample_max, float(finite.max()))
            if finite_values == 0:
                raise RuntimeError(f"{path.name}: sampled windows contain no finite pixels")
            raster_summaries.append(
                f"{path.name}: {dataset.count}x{dataset.height}x{dataset.width} "
                f"{dataset.dtypes[0]} sample=[{sample_min:.6g}, {sample_max:.6g}]"
            )

    csv_summaries: list[str] = []
    for path in sorted(root.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if not fields:
                raise RuntimeError(f"{path.name}: CSV has no header")
            rows = sum(1 for _ in reader)
        csv_summaries.append(f"{path.name}: {rows} rows, fields={fields}")

    if set(reference_grids) != {"s1", "s2"}:
        raise RuntimeError(f"Expected S1 and S2 grids, got {sorted(reference_grids)}")
    total_bytes = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
    print(f"Validated {len(actual)} files ({total_bytes / 2**30:.2f} GiB)")
    for source, (width, height, transform, crs) in sorted(reference_grids.items()):
        print(f"{source.upper()} grid: {width}x{height}, {crs}, transform={transform}")
    s1_transform = reference_grids["s1"][2]
    s2_transform = reference_grids["s2"][2]
    print(
        "S2->S1 origin offset (pixels): "
        f"dx={(s2_transform.c - s1_transform.c) / s1_transform.a:.6f}, "
        f"dy={(s2_transform.f - s1_transform.f) / abs(s1_transform.e):.6f}; "
        "explicit regridding is required"
    )
    print(f"Rasters: {len(raster_summaries)}; metadata CSVs: {len(csv_summaries)}")
    for summary in raster_summaries:
        print(summary)
    for summary in csv_summaries:
        print(summary)


if __name__ == "__main__":
    main()
