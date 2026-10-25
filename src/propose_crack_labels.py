
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import math

DATASET = Path("datasets/machine-defects-4class")
REVIEW = Path("outputs/empty_label_review/proposals")
LABEL_OUTPUT = REVIEW / "labels"
REVIEW.mkdir(parents=True, exist_ok=True)
LABEL_OUTPUT.mkdir(parents=True, exist_ok=True)

# Approximate normalized boxes (x_center, y_center, width, height).
# These are initial estimates from the contact sheet and require visual review.
PROPOSALS = {
    "crack-1500-": (0.20, 0.48, 0.40, 0.68),
    "istockphoto-666467174-": (0.42, 0.55, 0.70, 0.45),
    "49_jpg": (0.48, 0.67, 0.94, 0.28),
    "crack-1392-": (0.32, 0.50, 0.40, 0.98),
    "1-876-": (0.50, 0.50, 0.95, 0.95),
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
found = {}
for split in ("train", "valid", "test"):
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    for label_path in label_dir.glob("*.txt"):
        if label_path.read_text(encoding="utf-8").strip():
            continue

        matches = [
            key for key in PROPOSALS
            if key in label_path.name
        ]
        if not matches:
            continue

        key = matches[0]
        image_path = next(
            (p for p in image_dir.glob(label_path.stem + ".*")
             if p.suffix.lower() in EXTENSIONS),
            None,
        )
        if image_path:
            found[key] = (split, image_path, label_path)

missing = set(PROPOSALS) - set(found)
if missing:
    raise SystemExit(
        "Could not locate these empty-label images: "
        + ", ".join(sorted(missing))
    )

records = []
for key, (split, image_path, original_label) in found.items():
    x, y, w, h = PROPOSALS[key]

    # Keep the proposed box within normalized image bounds.
    x1 = max(0.0, x - w / 2)
    y1 = max(0.0, y - h / 2)
    x2 = min(1.0, x + w / 2)
    y2 = min(1.0, y + h / 2)

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read {image_path}")

    height, width = image.shape[:2]
    px1, py1 = int(x1 * width), int(y1 * height)
    px2, py2 = int(x2 * width), int(y2 * height)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(preview, (px1, py1), (px2, py2), (255, 0, 0), 3)

    # Store proposal separately. Class ID 3 = crack.
    proposal_path = LABEL_OUTPUT / f"{split}_{image_path.stem}.txt"
    proposal_path.write_text(
        f"3 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n",
        encoding="utf-8",
    )

    records.append((split, image_path.name, preview, proposal_path))

cols = 2
rows = math.ceil(len(records) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

for ax, (split, name, preview, proposal_path) in zip(axes, records):
    ax.imshow(preview)
    ax.set_title(f"{split} | {name}\nProposed crack box in red")
    ax.axis("off")

for ax in axes[len(records):]:
    ax.axis("off")

plt.tight_layout()
preview_path = REVIEW / "proposed_crack_boxes.png"
plt.savefig(preview_path, dpi=160, bbox_inches="tight")
plt.close()

print(f"Proposed labels: {len(records)}")
print(f"Preview: {preview_path}")
print(f"Label proposals: {LABEL_OUTPUT}")
print("No dataset labels were modified.")
print("Review every box before applying these proposals.")
