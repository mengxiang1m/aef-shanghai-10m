from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Layer:
    output: str
    source: str
    valid: str
    resampling: str
    dtype: str
    nodata: float | int
    cloud: str | None = None


LAYERS = (
    Layer(
        "S2.tif",
        "s2_202408_raw_utm51_10m.tif",
        "s2_valid_flag_202408_10m.tif",
        "bilinear",
        "float32",
        np.nan,
        "s2_cloud_suspect_flag_202408_10m.tif",
    ),
    Layer(
        "S1.tif",
        "s1_202408_asc171_vv_vh_angle_utm51_10m.tif",
        "s1_valid_flag_202408_asc171_10m.tif",
        "bilinear",
        "float32",
        np.nan,
    ),
    Layer(
        "building.tif",
        "bldg_mask_20240729.tif",
        "bldg_valid_flag_20240729.tif",
        "nearest",
        "uint8",
        255,
    ),
    Layer(
        "LST.tif",
        "lst_day_mean_2024.tif",
        "lst_day_valid_flag.tif",
        "bilinear",
        "float32",
        np.nan,
    ),
    Layer(
        "DEM.tif",
        "dem_elevation_m_2024.tif",
        "dem_valid_flag_2024.tif",
        "bilinear",
        "float32",
        np.nan,
    ),
    Layer(
        "NDVI.tif",
        "ndvi_max_2022.tif",
        "ndvi_valid_flag_2022.tif",
        "bilinear",
        "float32",
        np.nan,
    ),
)


def windows(width: int, height: int, tile: int):
    from rasterio.windows import Window

    for y in range(0, height, tile):
        for x in range(0, width, tile):
            yield Window(x, y, min(tile, width - x), min(tile, height - y))


def prepare_layer(raw_root: Path, output_root: Path, reference, layer: Layer, tile: int) -> None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    source_path = raw_root / layer.source
    valid_path = raw_root / layer.valid
    cloud_path = raw_root / layer.cloud if layer.cloud else None
    for path in (source_path, valid_path, cloud_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    output_path = output_root / layer.output
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    resampling = getattr(Resampling, layer.resampling)
    print(f"prepare {layer.output}: {layer.source} ({layer.resampling})", flush=True)
    with rasterio.open(source_path) as source, rasterio.open(valid_path) as valid_source:
        cloud_source = rasterio.open(cloud_path) if cloud_path else None
        try:
            profile = reference.profile.copy()
            profile.update(
                driver="GTiff",
                count=source.count,
                dtype=layer.dtype,
                nodata=layer.nodata,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="DEFLATE",
                predictor=2 if layer.dtype == "uint8" else 3,
                BIGTIFF="YES",
            )
            with rasterio.open(temp_path, "w", **profile) as destination:
                # Let GDAL stream each source band through its warp engine once. This is
                # substantially faster than repeatedly opening a 1 m source window for
                # every 10 m destination tile.
                for band_index in range(1, source.count + 1):
                    reproject(
                        source=rasterio.band(source, band_index),
                        destination=rasterio.band(destination, band_index),
                        src_nodata=source.nodata,
                        dst_transform=reference.transform,
                        dst_crs=reference.crs,
                        dst_nodata=layer.nodata,
                        resampling=resampling,
                        num_threads=4,
                        warp_mem_limit=512,
                    )

            mask_profile = reference.profile.copy()
            mask_profile.update(
                driver="GTiff",
                count=1,
                dtype="uint8",
                nodata=255,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="DEFLATE",
                predictor=2,
                BIGTIFF="IF_SAFER",
            )
            valid_temp = output_path.with_suffix(output_path.suffix + ".valid.tmp")
            cloud_temp = output_path.with_suffix(output_path.suffix + ".cloud.tmp")
            with rasterio.open(valid_temp, "w", **mask_profile) as valid_destination:
                reproject(
                    source=rasterio.band(valid_source, 1),
                    destination=rasterio.band(valid_destination, 1),
                    src_nodata=valid_source.nodata,
                    dst_transform=reference.transform,
                    dst_crs=reference.crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                    num_threads=4,
                    warp_mem_limit=256,
                )
            if cloud_source is not None:
                with rasterio.open(cloud_temp, "w", **mask_profile) as cloud_destination:
                    reproject(
                        source=rasterio.band(cloud_source, 1),
                        destination=rasterio.band(cloud_destination, 1),
                        src_nodata=cloud_source.nodata,
                        dst_transform=reference.transform,
                        dst_crs=reference.crs,
                        dst_nodata=255,
                        resampling=Resampling.nearest,
                        num_threads=4,
                        warp_mem_limit=256,
                    )

            with (
                rasterio.open(temp_path, "r+") as destination,
                rasterio.open(valid_temp) as valid_aligned,
            ):
                cloud_aligned = rasterio.open(cloud_temp) if cloud_source is not None else None
                total = ((reference.width + tile - 1) // tile) * (
                    (reference.height + tile - 1) // tile
                )
                try:
                    for index, window in enumerate(
                        windows(reference.width, reference.height, tile), 1
                    ):
                        array = destination.read(window=window)
                        valid = valid_aligned.read(1, window=window) == 1
                        if cloud_aligned is not None:
                            valid &= cloud_aligned.read(1, window=window) == 0
                        valid &= np.all(np.isfinite(array), axis=0)
                        array[:, ~valid] = layer.nodata
                        destination.write(array, window=window)
                        if index % 100 == 0 or index == total:
                            print(f"  {index}/{total} windows", flush=True)
                finally:
                    if cloud_aligned is not None:
                        cloud_aligned.close()
            valid_temp.unlink(missing_ok=True)
            cloud_temp.unlink(missing_ok=True)
        finally:
            if cloud_source is not None:
                cloud_source.close()

    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align the supplied Shanghai rasters to the August 2024 S2 10 m grid"
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data/shanghai_10m")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import rasterio
    except ImportError as error:
        raise SystemExit('Install the geo dependency first: pip install -e ".[geo]"') from error

    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_path = raw_root / "s2_202408_raw_utm51_10m.tif"
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(reference_path) as reference:
        if reference.count != 4:
            raise ValueError(f"Expected four S2 bands, found {reference.count}")
        print(
            f"reference: {reference.width}x{reference.height}, {reference.res}, {reference.crs}",
            flush=True,
        )
        for layer in LAYERS:
            output_path = output_root / layer.output
            if output_path.exists() and not args.overwrite:
                print(f"skip existing: {output_path}", flush=True)
                continue
            prepare_layer(raw_root, output_root, reference, layer, args.tile_size)

    print(f"aligned dataset ready: {output_root}", flush=True)


if __name__ == "__main__":
    main()
