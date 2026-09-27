# Experiment report: `uav2uav_yolo26n_baseline`

- Generated (UTC): `2026-09-27T18:36:39.529274+00:00`
- Weights: `runs/uav2uav_yolo26n_baseline/weights/best.pt`
- imgsz: `50`

## Notes
YOLO26n baseline on ThermalUAV2UAV (epochs=2, imgsz=50, batch=16).

## Final accuracy

| Metric | Value |
|---|---|
| mAP@0.5 | 0.1564 |
| mAP@0.5:0.95 | 0.0595 |
| Test P / R / F1 | 0.3223 / 0.2066 / 0.2518 |
| Train P / R / F1 | 0.3393 / 0.1940 / 0.2468 |
| Small-object score | 0.0000 |

## Latency (batch=1)

| Stage | mean ms |
|---|---|
| preprocess | 0.14 |
| infer | 3.81 |
| post | 0.15 |
| e2e | 6.54 |
| FPS | 152.90 |

## Model cost

| Metric | Value |
|---|---|
| Params | 2504190 |
| GFLOPs | 0.000 |
| Weight MB | 5.08 |
| Peak RSS MB | 318.9 |
| Peak VRAM MB | — |

## Efficiency

| Metric | Value |
|---|---|
| mAP / MB | 0.0308 |
| mAP / GFLOP | — |
| mAP × FPS | 23.9194 |

## Latency–accuracy curve

| imgsz | mAP50 | mAP50-95 | e2e ms | FPS |
|---|---|---|---|---|
| 320 | 0.0802 | 0.0225 | 14.25 | 70.18 |
| 480 | 0.0065 | 0.0019 | 22.60 | 44.24 |
| 640 | 0.0004 | 0.0001 | 43.34 | 23.07 |
| 800 | 0.0000 | 0.0000 | 50.43 | 19.83 |

## Epoch log

| epoch | mAP50-95 | P | R | F1 | grad% bb/neck/head |
|---|---|---|---|---|---|
| 1 | 0.0094 | 0.1038 | 0.0861 | 0.0941 | —/—/— |
| 2 | 0.0500 | 0.3455 | 0.1978 | 0.2516 | 90.8/84.4/40.9 |

## Artifacts in this folder

- `report.md` / `report.json`
- `plot_data.json`, `epochs.jsonl`, `latency.json`
- Compare index: `reports/compare/latency_accuracy.jsonl`

## Local artifacts (gitignored)

- Metrics: `runs/uav2uav_yolo26n_baseline/metrics`
- Plots: `runs/uav2uav_yolo26n_baseline/metrics/plots`
- Summary JSON: `runs/uav2uav_yolo26n_baseline/metrics/summary.json`
