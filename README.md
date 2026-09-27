# Thermal UAV detection

Baseline training and evaluation for YOLO26 on the [ThermalUAV2UAV](https://github.com/GabryV00/ThermalUAV2UAV_Dataset) dataset, with custom epoch/final metrics for edge-oriented comparisons.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data
git clone https://github.com/GabryV00/ThermalUAV2UAV_Dataset.git data/ThermalUAV2UAV_Dataset
```

## Train

```bash
python train.py                          # defaults: 50 epochs, imgsz 640, batch 16
python train.py --epochs 20 --imgsz 640 --batch 16 --name uav2uav_yolo26n_baseline
```

After training, the script writes:

| Output | Purpose |
|---|---|
| `runs/<name>/metrics/` | Epoch log, summary JSON, plots (local only) |
| `reports/<name>.md` / `.json` | Compact experiment log (safe to commit) |
| `weights/<name>_best_*.pt` + `latest.pt` | Demo checkpoints (gitignored) |
| `weights/manifest.json` | Index of saved demo weights (committed) |

## Quick checks

```bash
python show_random_sample.py   # random image + boxes → random_sample.png
python test.py                 # dataset paths + Ultralytics import smoke test
```

## Colab

See `notebooks/colab_baseline.ipynb`. Commit updated `reports/*` after a GPU run; keep full `runs/` on Drive.

## Layout

```text
train.py              training entrypoint
uav2uav.yaml          dataset paths / class names
metrics/              epoch callback, final eval, plots, report builder
weights/              demo checkpoint shelf + manifest
reports/              experiment write-ups
notebooks/            Colab workflow
```
