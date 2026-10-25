
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "datasets" / "machine-defects-4class"

# Exact filename prefixes of the five images with missing annotations.
TARGETS = {
    "crack-1500-",
    "istockphoto-666467174-",
    "49_jpg",
    "crack-1392-",
    "1-876-",
}

SPLITS = ("train", "valid", "test")


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    # Find all target images before making any changes.
    matches = []

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        for image_path in image_dir.iterdir():
            if not image_path.is_file():
                continue

            if any(image_path.stem.startswith(prefix) for prefix in TARGETS):
                label_path = label_dir / f"{image_path.stem}.txt"
                matches.append((image_path, label_path))

    # Safety check: don't modify anything unless all five are found.
    found_names = {image.stem for image, _ in matches}

    if len(matches) != 5 or len(found_names) != 5:
        print(f"Expected 5 unique images, found {len(found_names)}.")
        for image, _ in matches:
            print("Found:", image)
        print("No files were deleted.")
        return

    # Confirm each target has an empty label file.
    for image_path, label_path in matches:
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        if label_path.read_text(encoding="utf-8").strip():
            raise ValueError(
                f"Label is not empty; refusing to delete: {label_path}"
            )

    print("The following image/label pairs will be deleted:")
    for image_path, label_path in matches:
        print(f"  Image: {image_path.relative_to(DATASET)}")
        print(f"  Label: {label_path.relative_to(DATASET)}")

    confirmation = input(
        "\nType DELETE FIVE to continue: "
    ).strip()

    if confirmation != "DELETE FIVE":
        print("Cancelled. No files were deleted.")
        return

    for image_path, label_path in matches:
        image_path.unlink()
        label_path.unlink()

    print("\nRemoved exactly five images and their empty label files.")
    print("Original dataset and backup were not modified.")
    print("Next: run the four-class validation script.")


if __name__ == "__main__":
    main()
