
from pathlib import Path
import argparse
import yaml
import torch
from ultralytics import YOLO

EXPECTED_CLASSES = [
    "acceptable_weld",
    "bad_weld",
    "corrosion",
    "crack",
]


def validate_dataset(data_yaml):
    config = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = config.get("names")

    if isinstance(names, dict):
        actual = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        actual = names
    else:
        raise ValueError("data.yaml must contain a valid `names` list or mapping.")

    if actual != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected classes: {EXPECTED_CLASSES}\n"
            f"Found classes: {actual}"
        )

    for key in ("train", "val", "test"):
        if not config.get(key):
            raise ValueError(f"data.yaml is missing the '{key}' path.")

    print("Verified classes:", actual)


def main():
    parser = argparse.ArgumentParser(
        description="Train a four-class industrial defect detector."
    )
    parser.add_argument(
        "--data",
        default="datasets/machine-defects-4class/data.yaml",
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="four_class_v1")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = (project_root / args.data).resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    validate_dataset(data_path)

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {data_path}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(project_root / "runs" / "industrial_defect"),
        name=args.name,
        exist_ok=False,
        patience=10,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
        val=True,
        workers=2,
    )

    print("\nTraining finished.")
    print(f"Results directory: {results.save_dir}")
    print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
