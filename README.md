# Shanghai 10 m AEF-Lite reproduction

> **Status:** the original `AEFLite` checkpoint below is a static CNN baseline, not an STP
> reproduction. The repository now also contains a separate, paper-aligned temporal path in
> `src/aef/temporal_model.py`: source projectors, L/2 precision, L/8 time, L/16 space operators,
> learned pyramid exchanges, valid-period-conditioned `TemporalSummarizer`, a dense S63 mean
> direction bottleneck, and conditional decoders. The temporal path is a resource-scaled
> reproduction of the published topology; it is not Google's unreleased 480M-parameter model.

## Temporal STP run

Put the verified 48-file temporal download at
`data/raw/AEF_STP_temporal_2024/`. It contains 12 S1 months and three S2 months plus validity,
cloud, and metadata files. Validate and train with:

```powershell
$env:PYTHONPATH="$PWD/.runtime_geo;$PWD/src"
python tools/inspect_temporal_data.py data/raw/AEF_STP_temporal_2024
python tools/compute_temporal_stats.py --windows 256
pytest -q --basetemp .pytest_tmp_stp
python train_temporal.py --config configs/shanghai_stp_temporal.yaml --device cuda:1
python train_temporal_downstream.py --config configs/shanghai_stp_temporal.yaml --device cuda:1
```

S2 and the static targets define the canonical grid. S1 is offset by 2 m in Y and is explicitly
regridded window-by-window with bilinear data resampling and nearest-neighbour validity resampling.
Training retains source-specific asynchronous time axes, removes one valid frame per source for
reconstruction, passes pixel masks through time/space attention, and conditions the temporal summary
on the 2024 valid period. The reshape contract is tested using unique `(b,t,h,w)` token identities;
no direct `view(BHW,T,C)` is used.

The downstream command loads the best temporal checkpoint, supplies all available S1/S2 frames
(no reconstruction holdout), freezes the STP backbone by default, and fits dense 1x1 linear probes
for building, LST, DEM, and NDVI.

For DM02 transfers, `tools/export_baidu_temporary_dlink.ps1` can create an eight-hour,
file-scoped URL manifest without exporting account cookies. The server-side
`tools/download_temporal_from_manifest.py` downloader uses verified 4 MiB ranges because the Baidu
PCS node rejects larger ranges. Completed ranges survive interruption; regenerate the temporary
manifest and rerun to resume.

This repository is a practical, lightweight reproduction of the representation-learning idea in
*AlphaEarth Foundations*. It is not an exact reproduction of Google's unreleased ~480M-parameter
training system or weights.

The implemented setup uses Sentinel-2 and Sentinel-1 as inputs. A spatially dense, L2-normalized
64-dimensional embedding is trained to reconstruct S2, S1, building, LST, DEM, and NDVI through
source-specific pointwise MLP decoders. A batch-uniformity regularizer follows the paper's unit-sphere
embedding idea. Downstream linear probes predict building, LST, DEM, and NDVI from the learned
embedding.

## Required data

The supplied raw files go under `data/raw/`. Convert them to six co-registered rasters on the
August 2024 S2 10 m grid with:

```powershell
python tools/prepare_shanghai_10m.py
```

This writes the following files under `data/shanghai_10m/`:

```text
S2.tif        4 bands (as stored in the supplied August 2024 composite)
S1.tif        3 bands (VV, VH, incidence angle)
building.tif  1 band, binary 0/1 mask
LST.tif       1 band
DEM.tif       1 band
NDVI.tif      1 band
```

The preparation tool uses nearest-neighbor resampling for the categorical building mask and validity
masks, and bilinear resampling for continuous sources. It applies the supplied validity masks and
excludes cloud-suspect S2 pixels. All outputs have identical CRS, affine transform, width, and height.
`.npy` files in CHW or HWC order are also supported for non-geospatial experiments.

Use a projected Shanghai CRS whose pixel size is 10 m. Do not resample or align layers inside this
training code: prepare the grid once upstream so categorical building data can use nearest-neighbor
resampling and continuous data can use bilinear/cubic resampling appropriately.

## Environment and commands

From this directory:

```powershell
python -m pip install -e ".[geo,test]"
python tools/inspect_data.py --config configs/shanghai_10m.yaml
python train_pretrain.py --config configs/shanghai_10m.yaml
python train_downstream.py --config configs/shanghai_10m.yaml
python evaluate_downstream.py --config configs/shanghai_10m.yaml --checkpoint artifacts/downstream/best.pt
python infer.py --config configs/shanghai_10m.yaml --checkpoint artifacts/downstream/best.pt --embedding
pytest -q
```

For the supplied Baidu share, Phase 1 can be downloaded directly with the signed-in official client
session, without downloading the other files:

```powershell
.\tools\download_baidupan_phase1_authenticated.ps1 `
  -ShareUrl "https://pan.baidu.com/s/1uy95FpHhUYGNg7nvd3sT6w" `
  -Password "9yj8"
```

Keep the official Baidu client signed in. The script selects the 31 Phase 1 files (about 5.03 GB),
stages only those files in `/AEF_shanghai_phase1` in the signed-in cloud account, downloads them to
`data/raw`, and verifies every file size. Large-file range pieces are retained after interruption, so
running the same command again resumes rather than restarting completed pieces. The staging folder is
left in the cloud account so it can also be inspected or downloaded with the official client.

`inspect_data.py` validates alignment, reports spatial-split sample counts, and writes normalization
statistics using training blocks only. The pretraining run writes `best.pt`, `last.pt`, and
`history.json`; downstream training writes the same files under its own output directory.

`infer.py` reads only S1/S2, runs padded tiles, and writes georeferenced prediction GeoTIFFs (or
`.npy` outputs when the inputs are NumPy arrays). Binary building output is a probability; continuous
outputs are converted back to their configured physical scale.

The split is a deterministic hash of large spatial blocks rather than a random pixel/patch split.
This limits overly optimistic validation caused by spatial autocorrelation. If any split is empty,
reduce `split_block_size` relative to the raster dimensions.

Training is resumable. To continue the latest pretraining checkpoint while increasing the total epoch
count, run:

```powershell
python train_pretrain.py --config configs/shanghai_10m.yaml --resume --epochs 100
```

For a CPU staging run, limit the number of batches and keep its artifacts separate:

```powershell
python train_pretrain.py --config configs/shanghai_10m.yaml --device cpu --epochs 1 `
  --batch-size 2 --max-train-batches 200 --max-val-batches 50 `
  --output-dir artifacts/pretrain_cpu_stage1
```

The original static-composite approximation retains the paper's spatially dense 64-dimensional
unit-sphere bottleneck, per-source implicit decoders, dense batch-rotation uniformity objective,
source-specific reconstruction losses, and a shared-parameter teacher/student consistency pass that
randomly drops S1 while retaining S2. The supplied dataset has only one composite per input sensor,
so timestamp-conditioned decoding, frame holdout/dropout, VMF sampling, text alignment, and the
full temporal STP architecture cannot be reproduced from the Phase 1 data alone. Use the temporal
configuration above for the new multi-frame implementation.

## Model checkpoints

Create an initialized model file for architecture/integration checks with:

```powershell
python tools/export_init_model.py --config configs/shanghai_10m.yaml
```

`artifacts/aef_lite_init.pt` is randomly initialized and must not be treated as a trained model.
Real `best.pt` weights require the six Shanghai rasters and actual training.

The recommended trained representation is distributed with the
[v1.0.0 GitHub release](https://github.com/mengxiang1m/aef-shanghai-10m/releases/tag/v1.0.0);
its frozen-encoder downstream probe is `artifacts/downstream_dm02/best.pt`, with full spatial test metrics in
`artifacts/downstream_dm02/test_metrics.json`. These completed 100 pretraining epochs and 50
downstream epochs on a Tesla V100S. The earlier `pretrain_cpu_stage2` and
`downstream_cpu_stage2_balanced` files are retained as diagnostic CPU-stage checkpoints. See
`MODEL_CARD.md` for scope, limitations, checksums, and final metrics.

Because this environment's HTTPS connection aborts on a single 10 MB upload, the release stores the
pretraining checkpoint as numbered 2 MB parts. Download every
`aef_shanghai_pretrain_best.pt.partNNN` asset into one directory, then reconstruct and verify it with:

```powershell
.\tools\reassemble_pretrain_checkpoint.ps1 -PartsDirectory .
```

## Important interpretation notes

- The paper uses long multi-temporal sequences, timestamps, teacher/student consistency, and many
  additional global sources. This first Shanghai pipeline treats S1/S2 as aligned annual/composite
  rasters because no temporal data schema was supplied.
- Building is configured as a binary target. Change it to a continuous loss if it is building height
  or density.
- S2 uses `log1p` before standardization, mirroring the paper at a high level. Confirm whether your S2
  values are DN, reflectance, or already normalized. S1 is left unchanged because it may already be dB.
- Statistics are computed after the configured transform and clipped to +/-6 standard deviations.
- The initialized checkpoint only proves that the model can be serialized. A meaningful reproduction
  needs the actual rasters, their acquisition dates/compositing rule, nodata convention, and enough
  compute for training.
