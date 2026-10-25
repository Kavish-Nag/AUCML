from pathlib import Path
import os
import getpass

from roboflow import Roboflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "datasets"
WORKSPACE = "j-component-ml"
PROJECT = "machine-defects"
VERSION = 5


def main():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = getpass.getpass(
            "Roboflow API key (input hidden; do not share it): "
        ).strip()
    if not api_key:
        raise ValueError("A Roboflow API key is required for this SDK workflow.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    DESTINATION.mkdir(parents=True, exist_ok=True)
    dataset = version.download(
        model_format="yolov8",
        location=str(DESTINATION / f"{PROJECT}-{VERSION}"),
    )

    print("\nDataset download completed.")
    print(f"Location: {dataset.location}")
    print("Next: run src/inspect_dataset.py and verify the class semantics.")
    print("Do not train until the dataset has the four intended target classes.")


if __name__ == "__main__":
    main()
