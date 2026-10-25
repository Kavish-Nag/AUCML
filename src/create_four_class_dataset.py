
from pathlib import Path
from collections import Counter
import shutil
import sys
import yaml

SOURCE = Path("datasets/machine-defects-5-remapped")
OUTPUT = Path("datasets/machine-defects-4class")

SPLITS = ("train", "valid", "test")
EXCLUDED_CLASS_ID = 4

CLASS_NAMES = {
    0: "acceptable_weld",
    1: "bad_weld",
    2: "corrosion",
    3: "crack",
}

if not (SOURCE / "data.yaml").exists():
    sys.exit(f"Source dataset not found: {SOURCE}")

if OUTPUT.exists():
    sys.exit(
        f"Output already exists: {OUTPUT}\n"
        "Nothing was changed. Inspect the folder before rerunning."
    )

# Copy first. All edits happen inside the new output directory.
shutil.copytree(SOURCE, OUTPUT)

excluded_log = []
counts = Counter()
images_without_remaining_labels = []
errors = []

for split in SPLITS:
    label_dir = OUTPUT / split / "labels"
    image_dir = OUTPUT / split / "images"

    if not label_dir.exists() or not image_dir.exists():
        errors.append(f"{split}: missing images or labels directory")
        continue

    for label_path in sorted(label_dir.glob("*.txt")):
        original_lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        kept_lines = []

        for line_number, line in enumerate(original_lines, start=1):
            if not line.strip():
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
            except (ValueError, IndexError):
                errors.append(
                    f"{label_path}:{line_number}: invalid class ID"
                )
                kept_lines.append(line)
                continue

            if class_id == EXCLUDED_CLASS_ID:
                excluded_log.append(
                    f"{split}/{label_path.name}: line {line_number}: {line}"
                )
                continue

            if class_id not in CLASS_NAMES:
                errors.append(
                    f"{label_path}:{line_number}: unexpected class {class_id}"
                )
                kept_lines.append(line)
                continue

            kept_lines.append(line)
            counts[class_id] += 1

        # Preserve empty files as valid background-image labels.
        label_path.write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )

        if not kept_lines:
            matching_images = list(image_dir.glob(label_path.stem + ".*"))
            if matching_images:
                images_without_remaining_labels.append(
                    f"{split}/{matching_images[0].name}"
                )

# Write the four-class configuration.
config = {
    "path": ".",
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
}

with (OUTPUT / "data.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

(OUTPUT / "excluded_generic_annotations.txt").write_text(
    "\n".join(excluded_log) if excluded_log else "No generic annotations found.",
    encoding="utf-8",
)

(OUTPUT / "four_class_report.txt").write_text(
    "\n".join([
        "FOUR-CLASS DATASET REPORT",
        f"Source: {SOURCE}",
        f"Output: {OUTPUT}",
        "",
        "Remaining annotated object counts:",
        *[
            f"{class_id}: {CLASS_NAMES[class_id]} = {counts[class_id]}"
            for class_id in CLASS_NAMES
        ],
        "",
        f"Generic annotations excluded: {len(excluded_log)}",
        f"Images with no remaining annotations: {len(images_without_remaining_labels)}",
        "",
        "Images with no remaining annotations:",
        *images_without_remaining_labels,
        "",
        f"Errors: {len(errors)}",
        *errors,
    ]),
    encoding="utf-8",
)

print("Four-class dataset creation finished.")
print(f"Output: {OUTPUT}")
print(f"Excluded generic annotations: {len(excluded_log)}")

print("\nRemaining annotated objects:")
for class_id, name in CLASS_NAMES.items():
    print(f"  {class_id}: {name}: {counts[class_id]}")

print(
    "\nImages with no remaining annotations:",
    len(images_without_remaining_labels),
)
print(f"Errors: {len(errors)}")
print(f"Report: {OUTPUT / 'four_class_report.txt'}")
print(f"Exclusion log: {OUTPUT / 'excluded_generic_annotations.txt'}")

if errors:
    print("\nREVIEW REQUIRED. Do not train yet.")
else:
    print("\nNo unexpected class IDs or malformed class-ID fields found.")
    print("Run the next validation step before training.")
