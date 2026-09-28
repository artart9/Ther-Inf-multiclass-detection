# Thermal UAV detection

Baseline training and evaluation for YOLO26 on the [ThermalUAV2UAV](https://github.com/GabryV00/ThermalUAV2UAV_Dataset) dataset, with custom epoch/final metrics for edge-oriented comparisons.

## Setup (local Mac/Linux)

Do **not** run this block on Google Colab — Colab has no project `.venv`. Use `notebooks/colab_baseline.ipynb` instead.

```bash
cd /path/to/Ther-Inf-multiclass-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data
git clone https://github.com/GabryV00/ThermalUAV2UAV_Dataset.git data/ThermalUAV2UAV_Dataset
```

## Train

```bash
python models/baseline.py                          # defaults: 50 epochs, imgsz 640, batch 16, quantize fp16
python models/baseline.py --epochs 20 --imgsz 640 --batch 16 --quantize int8 --name uav2uav_yolo26n_baseline_int8
```

`--quantize` selects TensorRT latency export precision only (`fp32` | `fp16` | `int8`); training stays float. Use a **different `--name`** for each experiment so reports stay side-by-side.

After training, the script writes:

| Output | Purpose |
|---|---|
| `runs/<name>/metrics/` | Epoch log, summary, plots (local only) |
| `reports/<name>/` | Per-run folder: report + plot/latency/epoch data (commit) |
| `reports/compare/` | Shared index + meta plots (e.g. imgsz vs latency/GFLOPs) |
| `weights/<name>_best_*.pt` + `latest.pt` | Demo checkpoints (gitignored) |
| `weights/manifest.json` | Index of saved demo weights (committed) |

## Quick checks

```bash
python show_random_sample.py   # random image + boxes → random_sample.png
python test.py                 # dataset paths + Ultralytics import smoke test
```

## Colab

See `notebooks/colab_baseline.ipynb`. Commit `reports/<run>/` and `reports/compare/` after a GPU run; keep full `runs/` on Drive.

## Layout

```text
models/baseline.py    training entrypoint
uav2uav.yaml          dataset paths / class names
metrics/              epoch callback, final eval, plots, report, meta store
weights/              demo checkpoint shelf + manifest
reports/<run>/        one folder per experiment
reports/compare/      cross-run indexes
notebooks/            Colab workflow
```
