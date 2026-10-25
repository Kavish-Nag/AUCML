
from pathlib import Path
import shutil
import sys

SOURCE = Path("datasets/machine-defects-5-clean")
OUTPUT = Path("datasets/machine-defects-5-remapped")

# Original class ID -> new class ID
CLASS_MAP = {
    0: 1,  # Bad_Welding -> bad_weld
    1: 2,  # Corrosions -> corrosion
    2: 4,  # Defect -> generic_defect_review
    3: 0,  # Good Weld -> acceptable_weld
    4: 0,  # Good_Welding -> acceptable_weld
    5: 3,  # crack -> crack
}

CLASS_NAMES = [
    "acceptable_weld",
    "bad_weld",
    "corrosion",
    "crack",
    "generic_defect_review",
]

if not (SOURCE / "data.yaml").exists():
    sys.exit(f"Source dataset not found: {SOURCE}")

if OUTPUT.exists():
    sys.exit(
        f"Output already exists: {OUTPUT}\n"
        "Inspect it first. Rename or remove it manually before rerunning."
    )

# Copy the dataset structure first; only label files in OUTPUT are changed.
shutil.copytree(SOURCE, OUTPUT)

changes = []
errors = []

for split in ("train", "valid", "test"):
    label_dir = OUTPUT / split / "labels"

    if not label_dir.exists():
        errors.append(f"Missing label directory: {label_dir}")
        continue

    for path in sorted(label_dir.glob("*.txt")):
        original_lines = path.read_text(encoding="utf-8").splitlines()
        new_lines = []

        for line_number, line in enumerate(original_lines, start=1):
            if not line.strip():
                continue

            parts = line.split()

            try:
                old_id = int(parts[0])
            except (ValueError, IndexError):
                errors.append(f"{path}:{line_number}: invalid class ID")
                continue

            if old_id not in CLASS_MAP:
                errors.append(
                    f"{path}:{line_number}: unknown class ID {old_id}"
                )
                continue

            new_id = CLASS_MAP[old_id]
            new_lines.append(" ".join([str(new_id), *parts[1:]]))

            if new_id != old_id:
                changes.append(
                    f"{split}/{path.name}: line {line_number}, "
                    f"class {old_id} -> {new_id}"
                )

        path.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
        )

# Write a new YAML file with relative paths and the remapped class names.
yaml_text = """\
train: train/images
val: valid/images
test: test/images

nc: 5
names:
  0: acceptable_weld
  1: bad_weld
  2: corrosion
  3: crack
  4: generic_defect_review
"""

(OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

(OUTPUT / "remap_log.txt").write_text(
    "\n".join(changes) if changes else "No class IDs changed.",
    encoding="utf-8",
)

if errors:
    (OUTPUT / "remap_errors.txt").write_text(
        "\n".join(errors), encoding="utf-8"
    )
    print(f"Remapping completed with {len(errors)} errors.")
    print(f"See {OUTPUT / 'remap_errors.txt'}")
else:
    print("Remapping completed without structural errors.")

print(f"Output dataset: {OUTPUT}")
print(f"Class ID changes: {len(changes)}")
print(f"Remap log: {OUTPUT / 'remap_log.txt'}")
print("Source and original backup were not modified.")
