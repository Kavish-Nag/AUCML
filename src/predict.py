from pathlib import Path
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image, video, or webcam.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--source", required=True, help="Image/video path or webcam index, e.g. 0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", help="Save annotated prediction images/video")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(str(weights))
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        show=True,
        save=args.save,
        stream=(source == 0),
        verbose=True,
    )

    if source == 0:
        try:
            for result in results:
                # Ultralytics renders the live preview with show=True.
                # Detection details are printed here for later Unity/API integration.
                boxes = result.boxes
                if boxes is None:
                    continue
                names = result.names
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                    print({
                        "class": names[cls_id],
                        "confidence": round(confidence, 4),
                        "bbox_xyxy": xyxy,
                    })
        except KeyboardInterrupt:
            print("\nCamera inference stopped.")
    else:
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                print({
                    "class": result.names[cls_id],
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": xyxy,
                })


if __name__ == "__main__":
    main()
