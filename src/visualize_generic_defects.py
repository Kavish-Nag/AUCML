
from pathlib import Path
import math
import cv2
import matplotlib.pyplot as plt

DATASET = Path("datasets/machine-defects-5-clean")
OUTPUT = Path("outputs/defect_review")
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
items = []

for split in ("train", "valid", "test"):
    label_dir = DATASET / split / "labels"
    image_dir = DATASET / split / "images"

    if not label_dir.exists():
        continue

    for label_path in sorted(label_dir.glob("*.txt")):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        generic_boxes = []

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            try:
                if int(parts[0]) != 2:
                    continue
                values = list(map(float, parts[1:]))
            except ValueError:
                continue

            # This review script expects standard YOLO bounding boxes.
            if len(values) == 4:
                generic_boxes.append(values)

        if not generic_boxes:
            continue

        image_path = next(
            (p for p in image_dir.glob(label_path.stem + ".*")
             if p.suffix.lower() in IMAGE_EXTENSIONS),
            None,
        )

        if image_path:
            items.append((split, image_path, generic_boxes))

if not items:
    raise SystemExit("No generic-defect bounding boxes found.")

cols = 3
rows = math.ceil(len(items) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5))
axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

for ax, (split, image_path, boxes) in zip(axes, items):
    image = cv2.imread(str(image_path))
    if image is None:
        ax.set_title(f"Could not read: {image_path.name}")
        ax.axis("off")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    for x, y, w, h in boxes:
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    ax.imshow(image)
    ax.set_title(f"{split} | {image_path.name}\nGeneric boxes: {len(boxes)}")
    ax.axis("off")

for ax in axes[len(items):]:
    ax.axis("off")

plt.tight_layout()
output_path = OUTPUT / "generic_defect_contact_sheet.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Images reviewed: {len(items)}")
print(f"Saved: {output_path}")
