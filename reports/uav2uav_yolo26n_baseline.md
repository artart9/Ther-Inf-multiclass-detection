# Experiment report: `uav2uav_yolo26n_baseline`

- Generated (UTC): `2026-09-27T14:46:29.939276+00:00`
- Weights: `runs/uav2uav_yolo26n_baseline/weights/best.pt`
- imgsz: `640`

## Notes
YOLO26n smoke baseline on ThermalUAV2UAV (short local run).

## Final accuracy

| Metric | Value |
|---|---|
| mAP@0.5 | 0.7921 |
| mAP@0.5:0.95 | 0.4876 |
| Test P / R / F1 | 0.8311 / 0.7025 / 0.7614 |
| Train P / R / F1 | 0.8375 / 0.7278 / 0.7788 |
| Small-object score | 0.6420 |

## Latency (batch=1)

| Stage | mean ms |
|---|---|
| preprocess | 0.69 |
| infer | 34.41 |
| post | 0.33 |
| e2e | 38.03 |
| FPS | 26.30 |

## Model cost

| Metric | Value |
|---|---|
| Params | 2504190 |
| GFLOPs | 5.892 |
| Weight MB | 5.09 |
| Peak RSS MB | 283.3 |
| Peak VRAM MB | — |

## Efficiency

| Metric | Value |
|---|---|
| mAP / MB | 0.1556 |
| mAP / GFLOP | 0.1344 |
| mAP × FPS | 20.8288 |

## Latency–accuracy curve

| imgsz | mAP50 | mAP50-95 | e2e ms | FPS |
|---|---|---|---|---|
| 320 | 0.8917 | 0.5327 | 14.87 | 67.23 |
| 480 | 0.8601 | 0.5387 | 31.55 | 31.70 |
| 640 | 0.7921 | 0.4876 | 44.09 | 22.68 |
| 800 | 0.6948 | 0.4052 | 53.21 | 18.79 |

## Epoch log

| epoch | mAP50-95 | P | R | F1 | grad% bb/neck/head |
|---|---|---|---|---|---|
| 1 | 0.3532 | 0.8172 | 0.6432 | 0.7199 | —/—/— |
| 2 | 0.5120 | 0.8631 | 0.8113 | 0.8364 | 90.3/84.6/76.3 |

## Local artifacts (gitignored)

- Metrics: `runs/uav2uav_yolo26n_baseline/metrics`
- Plots: `runs/uav2uav_yolo26n_baseline/metrics/plots`
- Summary JSON: `runs/uav2uav_yolo26n_baseline/metrics/summary.json`

_Optional: copy selected plots into `reports/figures/` for write-ups._
