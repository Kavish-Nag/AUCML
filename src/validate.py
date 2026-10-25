from pathlib import Path
import argparse
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validate a trained YOLO detector.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if args.split == "test" and not config.get("test"):
        raise ValueError("data.yaml has no test split. Use --split val or add a held-out test split.")

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_path), split=args.split)
    print("\nValidation complete.")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
