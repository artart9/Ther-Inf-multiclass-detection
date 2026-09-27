"""Smoke test: dataset paths and Ultralytics load."""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
data = ROOT / "uav2uav.yaml"
images = ROOT / "data" / "ThermalUAV2UAV_Dataset" / "train" / "images"

print("data yaml:", data.exists(), data)
print("train images:", images.exists(), images)
print("image count:", len(list(images.glob("*.png"))) if images.exists() else 0)

model = YOLO("yolo26n.pt")
print("model loaded:", getattr(model, "model_name", "yolo26n"))
print("OK — next: python models/baseline.py")
