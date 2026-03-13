# 🚗 Object Detection for Self-Driving using YOLO
### Project Code: SIC/AI/003 | SIC Student Innovation Challenge 2025–26

| Member | Role | SIC ID |
|---|---|---|
| Mr. Aryan Kaushik | Team Leader | SIC202500606 |
| Mr. Naman Kumar | Model Builder | SIC202500602 |
| Ms. Muskan | Research & Testing | SIC202500809 |
| Mr. Sukesh Bhardwaj | Data Lead | SIC202500608 |
| Mr. Adhisha Pahuja | Presentation & Demo | SIC202500609 |

---

## 📋 Project Overview

Real-time object detection system for autonomous vehicles using **YOLOv8 (You Only Look Once)** — detecting 13 object categories including pedestrians, vehicles, traffic signs, and road hazards at 30+ FPS.

**Detects:** Person · Bicycle · Car · Motorcycle · Auto-rickshaw · Bus · Truck · Traffic Light · Stop Sign · Speed Breaker · Pothole · Animal · Construction Zone

---

## 🗂️ Project Structure

```
yolo_selfdriving/
├── src/
│   ├── dataset.py      # Data preprocessing, format conversion, augmentation
│   ├── train.py        # YOLOv8 training pipeline
│   ├── detect.py       # Real-time inference & demo
│   └── evaluate.py     # Metrics, evaluation, benchmarking
├── configs/
│   ├── dataset.yaml    # Dataset configuration & class definitions
│   └── train_config.yaml  # Training hyperparameters
├── notebooks/
│   └── YOLO_SelfDriving_Pipeline.ipynb  # End-to-end Google Colab notebook
├── scripts/
│   └── run_pipeline.sh # One-click full pipeline script
├── data/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── runs/               # Training outputs (auto-generated)
├── results/            # Evaluation charts & reports
└── requirements.txt
```

---

## ⚡ Quick Start

### Option 1: Google Colab (Recommended — Free GPU)
1. Open `notebooks/YOLO_SelfDriving_Pipeline.ipynb` in Google Colab
2. Set Runtime → Change runtime type → **GPU (T4)**
3. Run all cells top to bottom

### Option 2: Local Setup

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/yolo_selfdriving.git
cd yolo_selfdriving

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
bash scripts/run_pipeline.sh
```

---

## 📖 Step-by-Step Usage

### Step 1 — Dataset Preparation
```bash
# Setup directories & convert annotations
python src/dataset.py

# The script handles:
# - COCO → YOLO format conversion
# - Train/Val/Test split (70/20/10)
# - Data augmentation (2× factor)
# - Dataset statistics & visualization
```

### Step 2 — Training
```bash
# Train YOLOv8-Medium (recommended)
python src/train.py \
    --data   configs/dataset.yaml \
    --model  medium \
    --epochs 100 \
    --batch  16

# Quick test (50 epochs, nano model)
python src/train.py --model nano --epochs 50 --batch 32

# Resume interrupted training
python src/train.py --resume
```

**Model Size Options:**
| Flag | Model | Params | Speed | Best For |
|------|-------|--------|-------|---------|
| `nano` | yolov8n | 3.2M | 300+ FPS | Edge deployment |
| `small` | yolov8s | 11.2M | 120+ FPS | Speed-accuracy balance |
| `medium` | yolov8m | 25.9M | 80+ FPS | **Recommended** |
| `large` | yolov8l | 43.7M | 60+ FPS | High accuracy |

### Step 3 — Evaluation
```bash
# Full evaluation with plots
python src/evaluate.py \
    --model  runs/train/yolo_selfdriving/weights/best.pt \
    --data   configs/dataset.yaml \
    --benchmark \
    --report

# Generates:
# - Per-class metrics chart
# - Confusion matrix
# - Speed benchmark
# - JSON report
```

### Step 4 — Real-Time Detection
```bash
# Webcam (live demo)
python src/detect.py \
    --model runs/train/yolo_selfdriving/weights/best.pt \
    --source 0 \
    --mode realtime

# Video file
python src/detect.py --model best.pt --source video.mp4 --save

# Single image
python src/detect.py --model best.pt --source image.jpg --mode image

# Batch images
python src/detect.py --model best.pt --source ./test_images/ --mode batch
```

**Keyboard Controls (real-time mode):**
- `Q` — Quit
- `S` — Save screenshot
- `P` — Pause/Resume

---

## 📊 Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| mAP@0.5 | ≥ 0.75 | Mean Average Precision |
| FPS | ≥ 30 | Real-time performance |
| False Positive Rate | < 5% | Detection reliability |
| Detection Classes | 13 | Object categories |

---

## 🗂️ Datasets Used

| Dataset | Size | Source |
|---------|------|--------|
| COCO | 330K+ images | [cocodataset.org](https://cocodataset.org) |
| BDD100K | 100K videos | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) |
| KITTI Vision | 15K images | [cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti) |
| Custom Annotated | 2000+ images | Indian road conditions |

---

## 🔧 Model Export

```bash
# Export to ONNX (cross-platform deployment)
python -c "
from ultralytics import YOLO
model = YOLO('./runs/train/yolo_selfdriving/weights/best.pt')
model.export(format='onnx', imgsz=640, optimize=True)
"

# Export to TensorRT (NVIDIA edge devices)
# model.export(format='engine', imgsz=640)

# Export to TFLite (mobile/Raspberry Pi)
# model.export(format='tflite', imgsz=640)
```

---

## 🧰 Tech Stack

- **Framework:** PyTorch + Ultralytics YOLOv8
- **Annotation:** LabelImg, Roboflow
- **Augmentation:** Albumentations
- **Tracking:** Weights & Biases (W&B)
- **Visualization:** OpenCV, Matplotlib, Seaborn
- **Export:** ONNX, TensorRT

---

## 📞 Team Contact

| Name | Email | Role |
|------|-------|------|
| Aryan Kaushik | aryan29.kaushik@gmail.com | Team Leader |
| Naman Kumar | namanphogat0003@gmail.com | Model Builder |
| Muskan | lakramuskann@gmail.com | Research & Testing |
| Sukesh Bhardwaj | sukesh31104@gmail.com | Data Lead |
| Adhisha Pahuja | adhishapahuja25@gmail.com | Presentation & Demo |

---

*SIC Student Innovation Challenge 2025–26 | Project SIC/AI/003*
