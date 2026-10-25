
from pathlib import Path
import math
import cv2
import matplotlib.pyplot as plt

DATASET = Path("datasets/machine-defects-4class")
OUTPUT = Path("outputs/empty_label_review")
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
items = []

for split in ("train", "valid", "test"):
    label_dir = DATASET / split / "labels"
    image_dir = DATASET / split / "images"

    for label_path in sorted(label_dir.glob("*.txt")):
        if label_path.read_text(encoding="utf-8").strip():
            continue

        image_path = next(
            (
                p for p in image_dir.glob(label_path.stem + ".*")
                if p.suffix.lower() in IMAGE_EXTS
            ),
            None,
        )

        if image_path:
            items.append((split, image_path))

if not items:
    raise SystemExit("No empty-label images found.")

cols = 2
rows = math.ceil(len(items) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

for ax, (split, image_path) in zip(axes, items):
    image = cv2.imread(str(image_path))
    if image is None:
        ax.set_title(f"Could not read: {image_path.name}")
        ax.axis("off")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.set_title(f"{split} | {image_path.name}")
    ax.axis("off")

for ax in axes[len(items):]:
    ax.axis("off")

plt.tight_layout()
output_path = OUTPUT / "empty_labels_contact_sheet.png"
plt.savefig(output_path, dpi=160, bbox_inches="tight")
plt.close()

print(f"Images included: {len(items)}")
print(f"Saved: {output_path}")
