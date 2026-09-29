# Experiment report: `test`

- Generated (UTC): `2026-09-29T11:32:31.575912+00:00`
- Weights: `runs/test/weights/best.pt`
- imgsz: `256`
- quantize: `int8`

## Notes
YOLO26n-SPD (SPD-Conv stem, factor=4) on ThermalUAV2UAV (epochs=2, imgsz=256, batch=8, quantize=int8).

## Final accuracy

| Metric | Value |
|---|---|
| mAP@0.5 | 0.1097 |
| mAP@0.5:0.95 | 0.0261 |
| Test P / R / F1 | 0.2928 / 0.1488 / 0.1973 |
| Train P / R / F1 | 0.2150 / 0.1246 / 0.1578 |
| Small-object score | 0.0000 |

## Latency (Ultralytics inference-only, batch=1)

- Method: `ultralytics_speed_inference`
- Quantize: `int8`
- Backend: `pytorch`
- Device: `mps`
- Timed imgsz: `256`

| Stage | mean ms |
|---|---|
| preprocess | 0.1866 |
| infer (primary) | 4.4605 |
| post | 0.0849 |
| e2e (= infer) | 4.4605 |
| FPS | 224.1903 |

## Model cost

| Metric | Value |
|---|---|
| Params | 2510670 |
| GFLOPs | 0.284 |
| Weight MB | 5.09 |
| Peak RSS MB | 656.7 |
| Peak VRAM MB | — |

## Efficiency

| Metric | Value |
|---|---|
| mAP / MB | 0.0215 |
| mAP / GFLOP | 0.3864 |
| mAP × FPS | 24.6038 |

## Latency–accuracy curve

| imgsz | mAP50 | mAP50-95 | infer ms | FPS | backend |
|---|---|---|---|---|---|
| 320 | 0.0985 | 0.0265 | 5.6589 | 176.7134 | pytorch |
| 512 | 0.0541 | 0.0128 | 10.6853 | 93.5866 | pytorch |
| 640 | 0.0250 | 0.0060 | 14.6902 | 68.0727 | pytorch |
| 832 | 0.0172 | 0.0037 | 24.7665 | 40.3771 | pytorch |

## Epoch log

| epoch | mAP50-95 | P | R | F1 | grad% bb/neck/head |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | —/—/— |
| 2 | 0.0131 | 0.2313 | 0.1245 | 0.1619 | 96.0637/92.1345/156.8616 |

## Artifacts in this folder

- `report.md` / `report.json`
- `plot_data.json`, `epochs.jsonl`, `latency.json`
- Compare index: `reports/compare/latency_accuracy.jsonl`

## Local artifacts (gitignored)

- Metrics: `runs/test/metrics`
- Plots: `runs/test/metrics/plots`
- Summary JSON: `runs/test/metrics/summary.json`
