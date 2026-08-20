# AEF-Lite Shanghai 10 m model card

## Model-family clarification

There are now two deliberately separate model families in this repository:

- `AEFLite` is the completed static-composite CNN baseline whose results and checksums are recorded
  below. Its weights must not be described as an AEF/STP reproduction.
- `AEFTemporal` is the new resource-scaled implementation of the paper's STP topology and temporal
  training method. It consumes 12 monthly S1 frames and three S2 frames on independent time axes,
  uses masked temporal summarization, and reconstructs truly held-out frames plus the four static
  targets. Its checkpoint is written to `artifacts/stp_temporal/` once training completes.

Google's production architecture, data mixture, exact VMF sampler, text model, and weights are not
public. Accordingly, `AEFTemporal` is a method/topology reproduction on Shanghai data, not an exact
production-model reproduction.

## Scope

This project is a data-constrained, static-composite reproduction of the representation-learning
components in *AlphaEarth Foundations*. Sentinel-2 (four bands) and Sentinel-1 (VV, VH, incidence
angle) are inputs. S2, S1, building presence, daytime LST, DEM, and NDVI are reconstruction targets.
The encoder emits a spatially dense, L2-normalized 64-dimensional field.

It is not the official AEF model or an exact reproduction of the unpublished 480M-parameter system.
The Phase 1 dataset contains only one input composite per sensor, so it cannot reproduce temporal STP
attention, timestamp-conditioned decoding, VMF sampling, held-out temporal frames, or text alignment.

## Data and split

- Grid: 13,661 x 13,734 pixels, EPSG:32651, 10 m.
- Inputs: August 2024 S1/S2 composites.
- Targets: building/DEM 2024, daytime LST 2024, NDVI 2022, plus S1/S2.
- Deterministic spatial blocks: 7,964 train, 1,800 validation, 1,792 test patches (128 x 128).
- Coarse targets retain their native information scale in the loss: LST is re-gridded to 1 km and
  NDVI to approximately 50 m before loss evaluation.

## Implemented AEF components

- Source-specific S1/S2 stems and multi-resolution precision/context blocks.
- Dense 64D unit-sphere bottleneck.
- Two-hidden-layer pointwise implicit decoder per source.
- Source-specific masked reconstruction objectives, 20 m shift invariance for S1/S2.
- Dense batch-rotation orthogonality objective with weight 0.05.
- Shared-parameter teacher/student consistency with random S1 source dropout, weight 0.02.
- Frozen-encoder pointwise linear probes for building, LST, DEM, and NDVI.

## Temporal STP checkpoints

The resource-scaled temporal model was trained on DM02 using a Tesla V100S 32 GB. Pretraining ran
for 50 epochs over 2,714 spatial training patches; the selected representation checkpoint is epoch
27 with validation reconstruction loss 0.376224. Frozen dense linear probes ran for 30 epochs and
selected epoch 11 with validation loss 0.349505. The spatial split contains 733 validation and 733
test patches, and all available input frames are supplied to the probes without reconstruction
holdout.

Full independent test-split results:

| Target | MAE | RMSE | R2 | Classification metrics |
|---|---:|---:|---:|---|
| Building | - | - | - | balanced accuracy 0.8301; IoU 0.2826; F1 0.4407 |
| LST | 0.5412 | 0.6972 | 0.1779 | - |
| DEM | 0.7628 m | 1.0826 m | 0.6176 | - |
| NDVI | 0.0678 | 0.0940 | 0.7959 | - |

Inference-only weights and the metric/history files are published in the
[`stp-temporal-shanghai-v1` release](https://github.com/mengxiang1m/aef-shanghai-10m/releases/tag/stp-temporal-shanghai-v1).
The full optimizer checkpoints remain on DM02 for resuming training.

SHA-256 checksums:

- `stp_temporal_best_inference.pt`: `c9f1f7534cf1aeb2cc181bfce7d8a797fd77f88066942bb6cfd99f69dc77d02f`
- `stp_temporal_downstream_best_inference.pt`: `71ab2b1c6a29815d1e70698233861536132932bd9b03aef07c2bb007d274668c`
- `stp_temporal_test_metrics.json`: `f56ff0267d4ff9c22fe8809c4505e1dbf9774baacb8870aeb3d3fc73a5c34099`

These results validate the implemented STP/TemporalSummarizer representation-learning path on the
available Shanghai data. They are not benchmark numbers for Google's unreleased production model.

## DM02 final checkpoints

The recommended checkpoints were trained on a Tesla V100S 32 GB. Pretraining resumed from the CPU
Stage-2 checkpoint at epoch 2 and completed epoch 100 with batch size 32. The lowest validation
reconstruction loss was 0.242062 at epoch 58. Frozen-encoder downstream probes then trained for 50
epochs; the selected downstream checkpoint is epoch 43.

- Pretrained representation: `aef_shanghai_pretrain_best.pt` from the
  [v1.0.0 release](https://github.com/mengxiang1m/aef-shanghai-10m/releases/tag/v1.0.0)
- Downstream probes: `artifacts/downstream_dm02/best.pt`
- Full test metrics: `artifacts/downstream_dm02/test_metrics.json`

Full 1,792-patch spatial test split results:

| Target | Metric | Value |
|---|---:|---:|
| Building | Balanced accuracy | 0.8949 |
| Building | IoU | 0.2016 |
| LST | MAE | 0.5575 |
| LST | R2 | 0.2374 |
| DEM | MAE | 1.6064 m |
| DEM | R2 | 0.7966 |
| NDVI | MAE | 0.0770 |
| NDVI | R2 | 0.8891 |

SHA-256 checksums:

- `pretrain_dm02/best.pt`: `c0746d3c2ea71d5d5b34f19dcbf56780055db8719d4fa121cf23e4c34b13578f`
- `downstream_dm02/best.pt`: `5ccc95346698accae31ea5f6603f6f50cd6a87c8b18215c7129e59b23ea49984`
- `test_metrics.json`: `40d7ddfcbb9b4851e76c098d975f8a4602aab1267a933a500d9910be73c06421`

## Stage-2 checkpoints

`artifacts/pretrain_cpu_stage2/best.pt` continues the Stage-1 checkpoint for another 300 mini-batches
(batch size 2), followed by validation over a fixed, spatially distributed 200-sample subset. Across
both stages, the encoder has seen 500 training mini-batches. It is a genuine trained checkpoint but is
not converged. `artifacts/downstream_cpu_stage2_balanced/best.pt` contains frozen-encoder downstream
heads trained for 300 mini-batches using balanced BCE for rare building-positive pixels.

Full 1,792-patch spatial test split results:

| Target | Metric | Value |
|---|---:|---:|
| Building | Balanced accuracy | 0.8482 |
| Building | IoU | 0.1404 |
| LST | MAE | 0.6095 |
| LST | R2 | 0.0883 |
| DEM | MAE | 1.9500 m |
| DEM | R2 | 0.7278 |
| NDVI | MAE | 0.1367 |
| NDVI | R2 | 0.6943 |

These are early-stage diagnostic results, not final benchmark claims. In particular, the 2022 NDVI
target is temporally mismatched with the 2024 input composites, and upsampled LST/NDVI cannot gain
new spatial information beyond their original resolutions.

## Continue training

On the current CPU-only installation, a full epoch takes roughly 3.5 hours. Resume the Stage-1 model
for another limited stage with:

```powershell
$env:PYTHONPATH="D:\AEF\aef_shanghai\.deps;D:\AEF\aef_shanghai\src"
python train_pretrain.py --config configs/shanghai_10m.yaml --device cpu `
  --resume artifacts/pretrain_cpu_stage2/last.pt --epochs 3 --batch-size 2 `
  --max-train-batches 500 --max-val-batches 100 `
  --output-dir artifacts/pretrain_cpu_stage2
```

With a CUDA-enabled PyTorch environment, omit the batch limits and use `--device cuda`.
