
from pathlib import Path
from collections import Counter
import yaml

SOURCE = Path("datasets/machine-defects-5-clean")
OUTPUT = Path("datasets/machine-defects-5-remapped")

OLD_NAMES = {
    0: "Bad_Welding",
    1: "Corrosions",
    2: "Defect",
    3: "Good Weld",
    4: "Good_Welding",
    5: "crack",
}

CLASS_MAP = {0: 1, 1: 2, 2: 4, 3: 0, 4: 0, 5: 3}
NEW_NAMES = {
    0: "acceptable_weld",
    1: "bad_weld",
    2: "corrosion",
    3: "crack",
    4: "generic_defect_review",
}

def collect_counts(dataset):
    counts = Counter()
    annotation_lines = 0
    errors = []

    for split in ("train", "valid", "test"):
        label_dir = dataset / split / "labels"
        image_dir = dataset / split / "images"

        if not label_dir.exists() or not image_dir.exists():
            errors.append(f"{split}: missing images or labels directory")
            continue

        image_stems = {p.stem for p in image_dir.iterdir()
                       if p.is_file()}
        label_files = list(label_dir.glob("*.txt"))
        label_stems = {p.stem for p in label_files}

        if image_stems != label_stems:
            errors.append(
                f"{split}: image/label filename mismatch "
                f"({len(image_stems)} images, {len(label_stems)} labels)"
            )

        for path in label_files:
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue

                parts = line.split()

                try:
                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                except (ValueError, IndexError):
                    errors.append(f"{path}:{line_no}: invalid values")
                    continue

                if class_id not in NEW_NAMES and dataset == OUTPUT:
                    errors.append(f"{path}:{line_no}: invalid remapped ID")
                    continue
                if dataset == SOURCE and class_id not in OLD_NAMES:
                    errors.append(f"{path}:{line_no}: invalid original ID")
                    continue

                # Detection format: 4 values; segmentation: even number >= 6
                if len(values) != 4 and not (
                    len(values) >= 6 and len(values) % 2 == 0
                ):
                    errors.append(f"{path}:{line_no}: invalid annotation format")
                    continue

                counts[class_id] += 1
                annotation_lines += 1

    return counts, annotation_lines, errors


source_counts, source_lines, source_errors = collect_counts(SOURCE)
output_counts, output_lines, output_errors = collect_counts(OUTPUT)

expected = Counter()
for old_id, count in source_counts.items():
    expected[CLASS_MAP[old_id]] += count

print("--- ORIGINAL OBJECT COUNTS ---")
for class_id, name in OLD_NAMES.items():
    print(f"{class_id}: {name}: {source_counts[class_id]}")

print("\n--- REMAPPED OBJECT COUNTS ---")
for class_id, name in NEW_NAMES.items():
    print(f"{class_id}: {name}: {output_counts[class_id]}")

print("\n--- CHECKS ---")
print(f"Original annotation lines: {source_lines}")
print(f"Remapped annotation lines: {output_lines}")
print(f"Expected remapped counts: {expected == output_counts}")
print(f"Annotation line totals match: {source_lines == output_lines}")

try:
    with (OUTPUT / "data.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"YAML class count correct: {config.get('nc') == 5}")
    print(f"YAML names correct: {config.get('names') == NEW_NAMES}")
except Exception as exc:
    print(f"YAML check failed: {exc}")

all_errors = source_errors + output_errors
print(f"Validation errors: {len(all_errors)}")
for error in all_errors[:30]:
    print(error)

if not all_errors and expected == output_counts and source_lines == output_lines:
    print("\nPASS: counts and annotation structure are consistent.")
else:
    print("\nREVIEW REQUIRED: do not train yet.")
