# Metrics schema

Definitions used by the training callback and final evaluation. Keep these stable across comparable runs.

## Splits
- Epoch metrics: Ultralytics **val** split each epoch
- Final accuracy: **test** (primary) plus **train** P/R/F1
- Data config: `uav2uav.yaml`

## Detection
- Confidence / IoU / NMS: Ultralytics defaults unless overridden
- Image size (baseline): **640**
- F1: `2 * P * R / (P + R)` (0 if P = R = 0)
- Small object: box area `< 32 * 32` pixels on the letterboxed model input
- Small AP: AP@0.5 on GT/pred pairs where the GT box is small

## Gradient % (per epoch)
For each group `g ∈ {backbone, neck, head}`:
1. Mean grad L2 norm over that epoch’s training batches → `grad_norm[g]`
2. `grad_pct[g] = 100 * |G_t - G_{t-1}| / (G_{t-1} + 1e-12)` (null on epoch 1)
3. Groups on YOLO26 `model.model`:
   - **backbone**: layers before the first `Concat`
   - **head**: final `Detect` layer
   - **neck**: layers in between

## Latency (final only)
- batch = 1, warmup = 30, sample = 200 test images
- Stages: preprocess / infer / post / e2e (ms)
- Full e2e samples stored in `summary.json` for distribution plots
- FPS = `1000 / mean(e2e_ms)`

## Efficiency (final only)
- `map_per_mb = map50 / weight_mb`
- `map_per_gflop = map50 / gflops`
- `map_times_fps = map50 * fps`
- Latency–accuracy curve: imgsz in `[320, 480, 640, 800]`

## Outputs
```text
runs/<exp>/metrics/
  epochs.jsonl
  summary.json
  plots/
reports/<exp>.md
reports/<exp>.json
```
