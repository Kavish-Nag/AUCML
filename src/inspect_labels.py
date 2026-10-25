
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "machine-defects-5"

config = yaml.safe_load(
    (DATASET / "data.yaml").read_text(encoding="utf-8")
)
names = config["names"]

if isinstance(names, list):
    names = dict(enumerate(names))

counts = Counter()
empty_files = 0
invalid_lines = 0
label_files = 0

for split in ["train", "valid", "test"]:
    label_dir = DATASET / split / "labels"

    if not label_dir.exists():
        print(f"Missing labels directory: {label_dir}")
        continue

    for path in label_dir.glob("*.txt"):
        label_files += 1
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.split()

            try:
                class_id = int(parts[0])
                if class_id not in names:
                    raise ValueError("Unknown class ID")
                if len(parts) != 5:
                    raise ValueError("Expected 5 YOLO values")
                values = [float(v) for v in parts[1:]]
                if not all(0 <= v <= 1 for v in values):
                    raise ValueError("Coordinates outside [0, 1]")
                counts[class_id] += 1
            except (ValueError, IndexError):
                invalid_lines += 1

print(f"Label files checked: {label_files}")
print(f"Empty label files: {empty_files}")
print(f"Invalid annotation lines: {invalid_lines}")
print("\nAnnotated object counts:")

for class_id, name in sorted(names.items(), key=lambda item: int(item[0])):
    print(f"ID {class_id}: {name}: {counts[int(class_id)]} objects")
