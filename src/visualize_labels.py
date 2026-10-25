
from pathlib import Path
import random
import yaml
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "machine-defects-5"
OUTPUT = ROOT / "outputs" / "class_samples"
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

config = yaml.safe_load(
    (DATASET / "data.yaml").read_text(encoding="utf-8")
)
names = config["names"]

if isinstance(names, list):
    names = dict(enumerate(names))
else:
    names = {int(k): v for k, v in names.items()}

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
samples = {class_id: [] for class_id in names}

# Search all three splits and collect images containing each class.
for split in ["train", "valid", "test"]:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        continue

    for label_path in label_dir.glob("*.txt"):
        image_path = next(
            (
                image_dir / (label_path.stem + ext)
                for ext in image_extensions
                if (image_dir / (label_path.stem + ext)).exists()
            ),
            None,
        )

        if image_path is None:
            continue

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
            class_ids = {
                int(line.split()[0])
                for line in lines
                if line.strip()
                and len(line.split()) == 5
                and line.split()[0].isdigit()
            }
        except OSError:
            continue

        for class_id in class_ids:
            if class_id in samples:
                samples[class_id].append((image_path, label_path))

for class_id, class_name in names.items():
    candidates = samples[class_id]
    random.shuffle(candidates)
    candidates = candidates[:12]

    if not candidates:
        print(f"No samples found for {class_name}")
        continue

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle(
        f"Class {class_id}: {class_name}",
        fontsize=15,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (image_path, label_path) in zip(axes.flat, candidates):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        ax.imshow(image)
        ax.set_title(image_path.name[:22], fontsize=7)

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            if cid != class_id:
                continue

            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            rect = plt.Rectangle(
                (x1, y1),
                bw * width,
                bh * height,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
            ax.add_patch(rect)

    fig.tight_layout()
    output_file = OUTPUT / f"class_{class_id}_{class_name.replace(' ', '_')}.png"
    fig.savefig(output_file, dpi=130)
    plt.close(fig)
    print(f"Saved: {output_file}")

print("\nOpen outputs/class_samples to inspect the class images.")
