
from pathlib import Path

DATASET = Path("datasets/machine-defects-5-clean")
CLASS_COUNT = 6

LABEL_DIRS = [
    DATASET / "train" / "labels",
    DATASET / "valid" / "labels",
    DATASET / "test" / "labels",
]

def audit_line(line):
    parts = line.split()

    if len(parts) < 5:
        return "Too few values"

    try:
        class_id = int(parts[0])
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return "Non-numeric value"

    if not 0 <= class_id < CLASS_COUNT:
        return f"Invalid class ID: {class_id}"

    if not all(0 <= v <= 1 for v in values):
        return "Coordinate outside [0, 1]"

    # YOLO detection: class + x_center, y_center, width, height
    if len(values) == 4:
        _, _, width, height = values

        if width <= 0 or height <= 0:
            return "Non-positive bounding-box dimensions"

        x, y = values[:2]
        if x - width / 2 < 0 or x + width / 2 > 1:
            return "Bounding box exceeds image horizontally"
        if y - height / 2 < 0 or y + height / 2 > 1:
            return "Bounding box exceeds image vertically"

        return None

    # YOLO segmentation: class + at least 3 (x, y) points
    if len(values) >= 6 and len(values) % 2 == 0:
        return None

    return (
        f"Unrecognized annotation structure: "
        f"{len(values)} coordinate values"
    )


checked = 0
empty = 0
issues = 0

for label_dir in LABEL_DIRS:
    if not label_dir.exists():
        print(f"MISSING DIRECTORY: {label_dir}")
        continue

    for path in sorted(label_dir.glob("*.txt")):
        checked += 1
        lines = path.read_text(encoding="utf-8").splitlines()
        nonempty = [(i, line) for i, line in enumerate(lines, 1)
                    if line.strip()]

        if not nonempty:
            empty += 1
            continue

        for line_number, line in nonempty:
            problem = audit_line(line)

            if problem:
                issues += 1
                print(f"\nFile: {path}")
                print(f"Line {line_number}: {line}")
                print(f"Issue: {problem}")

print("\n--- SEGMENTATION-AWARE AUDIT ---")
print(f"Label files checked: {checked}")
print(f"Empty label files: {empty}")
print(f"Lines requiring review: {issues}")
print("No files were modified.")
