"""Show a random ThermalUAV2UAV image with its YOLO label boxes."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

DATASET_ROOT = Path("data/ThermalUAV2UAV_Dataset")
NAMES = {0: "drone"}
CLASS_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8"]


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    bw, bh = w * img_w, h * img_h
    x1 = (cx * img_w) - bw / 2
    y1 = (cy * img_h) - bh / 2
    return x1, y1, bw, bh


def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(
            f"Dataset not found at {DATASET_ROOT.resolve()}. "
            "Clone it with:\n"
            "  git clone https://github.com/GabryV00/ThermalUAV2UAV_Dataset.git "
            "data/ThermalUAV2UAV_Dataset"
        )

    split = random.choice(["train", "val", "test"])
    images = list((DATASET_ROOT / split / "images").glob("*.png"))
    images += list((DATASET_ROOT / split / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {DATASET_ROOT / split / 'images'}")

    labeled = []
    for image_path in images:
        label_path = DATASET_ROOT / split / "labels" / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text().strip():
            labeled.append(image_path)
    image_path = random.choice(labeled or images)
    label_path = DATASET_ROOT / split / "labels" / f"{image_path.stem}.txt"

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"{split} / {image_path.name}")
    ax.axis("off")

    box_count = 0
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            x, y, bw, bh = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            ax.add_patch(
                patches.Rectangle(
                    (x, y), bw, bh, linewidth=2, edgecolor=color, facecolor="none"
                )
            )
            ax.text(
                x,
                max(y - 4, 0),
                NAMES.get(cls_id, str(cls_id)),
                color="white",
                fontsize=9,
                bbox=dict(facecolor=color, edgecolor="none", pad=2),
            )
            box_count += 1

    print(f"Image: {image_path}")
    print(f"Label: {label_path} ({box_count} boxes)")

    out = Path("random_sample.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved preview to: {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
