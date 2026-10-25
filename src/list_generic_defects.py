
from pathlib import Path

DATASET = Path("datasets/machine-defects-5-clean")
SPLITS = ("train", "valid", "test")

for split in SPLITS:
    label_dir = DATASET / split / "labels"
    image_dir = DATASET / split / "images"

    if not label_dir.exists():
        continue

    for label_path in sorted(label_dir.glob("*.txt")):
        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
            except (ValueError, IndexError):
                continue

            if class_id == 2:
                image_path = next(
                    (
                        p for p in image_dir.glob(label_path.stem + ".*")
                        if p.is_file()
                    ),
                    None,
                )

                print(
                    f"Split: {split} | "
                    f"Image: {image_path} | "
                    f"Label: {label_path} | "
                    f"Line: {line_no} | "
                    f"Annotation: {line}"
                )
