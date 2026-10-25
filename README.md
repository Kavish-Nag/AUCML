# Industrial Defect Detection with YOLO

This project provides a starter YOLO object-detection workflow for:
1. Normal
2. Damaged/Rusted
3. Cracked
4. Missing/Abnormal Component

It uses Ultralytics YOLO and a Roboflow dataset export. **The starter dataset linked in the project has not been verified to contain all four target classes.** You must inspect and map the actual classes, and add annotated images for missing/abnormal components and normal examples as needed. The code validates class names and stops if the dataset's `data.yaml` does not match the configured target classes. Do not relabel `Good_Welding` as general Normal without confirming that the images represent the components and operating conditions you intend to inspect.

This is object detection: predictions include bounding boxes. It does not implement Unity AR integration.

## 1. Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- Internet access for package/model/dataset downloads
- Roboflow account/API key if using the Roboflow download script

## 2. Setup in PowerShell
Extract the ZIP, open the extracted folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
Select the `.venv` interpreter in VS Code (`Ctrl+Shift+P` -> `Python: Select Interpreter`).

## 3. Download the Roboflow dataset
The default downloader points to the dataset URL supplied for Machine Defects, version 5. Roboflow project IDs, versions, and export formats can change. If the download fails, open the dataset page and use its Download Dataset option to export in **YOLOv8** format, then extract it into `datasets/machine-defects-5/`.

Recommended script:
```powershell
python src/download_dataset.py
```
It prompts for a Roboflow API key, or reads `ROBOFLOW_API_KEY` from the environment. Do not commit or share your API key. If the dataset is public and the API supports unauthenticated downloads, Roboflow may still require a key for this SDK workflow.

Set the API key for this terminal only:
```powershell
$env:ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
python src/download_dataset.py
```

## 4. Inspect class names before training
```powershell
python src/inspect_dataset.py --data datasets/machine-defects-5/data.yaml
```
This prints class names and image/label counts. It does not automatically reinterpret labels. Open the resulting `data.yaml` and confirm that classes are correctly mapped to:
- `normal`
- `damaged_rusted`
- `cracked`
- `missing_abnormal_component`

If the dataset uses labels such as `Good_Welding`, `Corrosions`, `crack`, and `Bad_Welding`, do not automatically map `Good_Welding` to general `normal` or `Bad_Welding` to `missing_abnormal_component`. Those are not semantically equivalent. Curate/relabel data in Roboflow and export a new dataset version whose class names match the four target categories. The training script intentionally refuses mismatched classes.

## 5. Train
Once `data.yaml` matches the four target classes and contains appropriate annotated examples:
```powershell
python src/train.py --data datasets/machine-defects-5/data.yaml --epochs 50 --imgsz 640
```
For a first smoke test:
```powershell
python src/train.py --data datasets/machine-defects-5/data.yaml --epochs 3 --imgsz 640
```
Training uses a small pretrained YOLO model (`yolo11n.pt`) and saves results under `runs/industrial_defect/`. Model weights are downloaded by Ultralytics on first use.

## 6. Validate and predict
```powershell
python src/validate.py --data datasets/machine-defects-5/data.yaml --weights runs/industrial_defect/train/weights/best.pt
```
The exact run folder can differ if the output directory already exists. Use the path printed by training.

Predict on an image:
```powershell
python src/predict.py --weights runs/industrial_defect/train/weights/best.pt --source "C:\path\to\image.jpg"
```
Predict using a webcam:
```powershell
python src/predict.py --weights runs/industrial_defect/train/weights/best.pt --source 0
```
Press `q` to exit the live camera preview. Camera performance depends on hardware and model speed.

## 7. Outputs
- `runs/industrial_defect/.../weights/best.pt`: trained model
- `runs/industrial_defect/.../results.png`: training curves, if produced by the installed Ultralytics version
- Prediction windows/files with class labels, confidence scores and bounding boxes

## 8. Important limitations
- The supplied Roboflow project is only a candidate starting dataset; it is not guaranteed to provide the required four categories.
- A detector cannot reliably detect a missing component unless training examples and annotations teach it what is missing, or a reference/expected-component strategy is implemented.
- An image with no boxes can be treated as a background/normal image during training, but it does not itself teach a semantic "normal" bounding box. Normal is often better represented by no detected faults plus a separate inspection rule.
- A four-class detector needs representative annotated data from the intended camera, objects, lighting, and viewpoint.
- This code does not yet send detections to Unity. It provides detection classes, confidence scores and bounding boxes as the ML side of the pipeline.
- Check the dataset license and Roboflow terms before redistribution or non-academic use.
