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

## Gradient metrics (per epoch)
For each group `g ∈ {backbone, neck, head}`:
1. Mean grad L2 norm over that epoch’s **optimizer steps** → `grad_norm[g]`
   - Sampled immediately before Ultralytics `optimizer_step()` (which zeros grads)
   - **Unit:** none (raw Euclidean magnitude of ∂L/∂w over the group’s parameters)
2. `grad_pct[g] = 100 * |G_t - G_{t-1}| / (G_{t-1} + 1e-12)`
   - **Unit:** **percent (%)** change vs the previous epoch
   - **null on epoch 1** (no previous epoch); plots start at **epoch 2**
3. Groups on YOLO26 `model.model`:
   - **backbone**: layers before the first `Concat`
   - **head**: final `Detect` layer
   - **neck**: layers in between

## Latency (final only)
- Methodology matches Ultralytics detect **Speed (T4 TensorRT10)** column:
  - Prefer **TensorRT FP16** engine on CUDA; else PyTorch on the active device
  - Dummy `uint8` image, **batch = 1**, square `imgsz`
  - Warmup **10**, timed runs **100** (curve uses 50)
  - Read `results[0].speed` stages; primary metric = **`inference`** (pre/post excluded)
  - Iterative σ-clipping (σ=2, max 3 iters) on samples
- Stages still recorded: preprocess / infer / post; **`e2e` is aliased to infer**
- Stats: mean, std, p50, p95 + full `samples_ms` list
- FPS = `1000 / mean(infer_ms)`
- Metadata: `method`, `backend` (`tensorrt_fp16` | `pytorch`), `device`, optional `engine_path`
- Engines cached under `runs/.../metrics/engines/`
- Saved per run to `reports/<exp>/latency.json` (+ local `runs/.../metrics/`)
- Cross-run index: `reports/compare/latency_accuracy.jsonl`

## Efficiency (final only)
- `map_per_mb = map50 / weight_mb`
- `map_per_gflop = map50 / gflops`
- `map_times_fps = map50 * fps`
- Latency–accuracy curve: imgsz in `[320, 480, 640, 800]`

## Outputs
```text
runs/<exp>/metrics/          # local only (gitignored)
  epochs.jsonl, summary.json, plot_data.json, plots/

reports/<exp>/               # one folder per run (committed)
  report.md
  report.json
  plot_data.json
  epochs.jsonl
  latency.json

reports/compare/             # shared cross-run indexes
  latency_accuracy.jsonl
```
