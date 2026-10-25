
from pathlib import Path
from collections import Counter
import yaml

DATASET = Path("datasets/machine-defects-4class")
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_NAMES = {
    0: "acceptable_weld",
    1: "bad_weld",
    2: "corrosion",
    3: "crack",
}

errors = []
counts = Counter()
split_stats = {}
empty_labels = []

# 1. Validate YAML configuration.
try:
    with (DATASET / "data.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})
    names = {int(k): v for k, v in names.items()}

    if config.get("nc") != 4:
        errors.append("YAML nc must be 4")

    if names != EXPECTED_NAMES:
        errors.append(f"Unexpected YAML names: {names}")

    for key, expected_path in (
        ("train", "train/images"),
        ("val", "valid/images"),
        ("test", "test/images"),
    ):
        if config.get(key) != expected_path:
            errors.append(f"Unexpected YAML {key} path: {config.get(key)}")

except Exception as exc:
    errors.append(f"Could not read data.yaml: {exc}")

# 2. Validate image/label pairs and annotations.
for split in SPLITS:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        errors.append(f"{split}: missing images or labels directory")
        continue

    images = {
        p.stem: p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    labels = {p.stem: p for p in label_dir.glob("*.txt")}

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))

    if missing_labels:
        errors.append(
            f"{split}: {len(missing_labels)} images have no label file"
        )
    if missing_images:
        errors.append(
            f"{split}: {len(missing_images)} label files have no image"
        )

    split_stats[split] = {
        "images": len(images),
        "label_files": len(labels),
        "objects": 0,
        "empty_labels": 0,
    }

    for stem, label_path in labels.items():
        lines = label_path.read_text(encoding="utf-8").splitlines()
        valid_lines = [line for line in lines if line.strip()]

        if not valid_lines:
            split_stats[split]["empty_labels"] += 1
            empty_labels.append(f"{split}/{stem}")
            continue

        for line_no, line in enumerate(valid_lines, start=1):
            parts = line.split()

            try:
                class_id = int(parts[0])
                values = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                errors.append(f"{label_path}:{line_no}: invalid values")
                continue

            if class_id not in EXPECTED_NAMES:
                errors.append(
                    f"{label_path}:{line_no}: invalid class ID {class_id}"
                )
                continue

            if not all(0 <= v <= 1 for v in values):
                errors.append(
                    f"{label_path}:{line_no}: coordinate outside [0, 1]"
                )
                continue

            if len(values) == 4:
                x, y, w, h = values

                if w <= 0 or h <= 0:
                    errors.append(
                        f"{label_path}:{line_no}: invalid box dimensions"
                    )
                    continue

                if (
                    x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6
                    or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6
                ):
                    errors.append(
                        f"{label_path}:{line_no}: box outside image"
                    )
                    continue

            elif len(values) >= 6 and len(values) % 2 == 0:
                # Polygon coordinates: x1, y1, x2, y2, ...
                pass
            else:
                errors.append(
                    f"{label_path}:{line_no}: unsupported annotation format"
                )
                continue

            counts[class_id] += 1
            split_stats[split]["objects"] += 1

# 3. Print report.
print("--- SPLIT SUMMARY ---")
for split, stats in split_stats.items():
    print(
        f"{split}: images={stats['images']}, "
        f"label_files={stats['label_files']}, "
        f"objects={stats['objects']}, "
        f"empty_labels={stats['empty_labels']}"
    )

print("\n--- CLASS COUNTS ---")
for class_id, name in EXPECTED_NAMES.items():
    print(f"{class_id}: {name}: {counts[class_id]}")

print(f"\nImages with empty labels: {len(empty_labels)}")
for item in empty_labels:
    print(f"  {item}")

print(f"\nValidation errors: {len(errors)}")
for error in errors[:50]:
    print(f"ERROR: {error}")

if not errors:
    print("\nPASS: structural validation completed.")
    print("Manually review the empty-label images before training.")
else:
    print("\nFAIL: resolve the errors before training.")
