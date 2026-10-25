
from pathlib import Path
import argparse
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_path(value, yaml_dir, dataset_root):
    path = Path(value)

    if path.is_absolute():
        return path

    # Prefer the dataset root, then data.yaml's directory.
    candidate = dataset_root / path
    if candidate.exists():
        return candidate

    return yaml_dir / path


def count_images(entry, yaml_dir, dataset_root):
    path = resolve_path(entry, yaml_dir, dataset_root)

    if path.is_dir():
        return [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            listed_path = Path(line)
            if not listed_path.is_absolute():
                listed_path = path.parent / listed_path

            if listed_path.is_file():
                images.append(listed_path)

        return images

    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]

    print(f"  Could not resolve: {path}")
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    yaml_path = Path(args.data).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    dataset_root = Path(config.get("path") or yaml_path.parent)
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()

    names = config.get("names", [])
    print(f"Config: {yaml_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {names}")

    for split in ("train", "val", "test"):
        entries = config.get(split)
        if not entries:
            print(f"{split}: not configured")
            continue

        if isinstance(entries, str):
            entries = [entries]

        found = []
        for entry in entries:
            found.extend(count_images(entry, yaml_path.parent, dataset_root))

        print(f"{split} images found: {len(set(found))}")
        if found:
            print(f"  Example: {found[0]}")

    print("\nInspect class semantics before mapping or training.")


if __name__ == "__main__":
    main()
