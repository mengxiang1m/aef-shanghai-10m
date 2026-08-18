# AEF-Lite Shanghai 10 m model card

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
