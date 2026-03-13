# -*- coding: utf-8 -*-
"""
============================================================
CPU Quickstart -- One Script to Run Everything
Project Code: SIC/AI/003

Run this directly:   python quickstart_cpu.py
============================================================
"""

import os
import sys
import subprocess


def run(cmd):
    print(f"\n  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def header(title):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")


def main():
    header("SIC/AI/003 -- CPU Quickstart Setup")
    print("""
  This script will guide you through:
    1. Install all dependencies
    2. Check your environment
    3. Run a quick demo detection (pretrained COCO weights)
    4. (Optional) Full training on your dataset

  No GPU required -- everything runs on CPU!
  For free GPU: open the Google Colab notebook instead.
""")
    input("  Press ENTER to start > ")

    # STEP 1: Install
    header("Step 1/4: Installing Dependencies")
    ok = run(
        f"{sys.executable} -m pip install "
        "ultralytics opencv-python albumentations "
        "seaborn tqdm pandas matplotlib -q"
    )
    if not ok:
        print("  [FAILED] Try running: pip install ultralytics manually")
        sys.exit(1)
    print("  [OK] All dependencies installed!")

    # STEP 2: Check environment
    header("Step 2/4: Checking Environment")
    run(f'{sys.executable} -c "import torch; print(\'PyTorch:\', torch.__version__); print(\'GPU:\', torch.cuda.is_available())"')
    run(f'{sys.executable} -c "import ultralytics; print(\'Ultralytics:\', ultralytics.__version__)"')
    run(f'{sys.executable} -c "import cv2; print(\'OpenCV:\', cv2.__version__)"')

    # STEP 3: Quick demo
    header("Step 3/4: Quick Demo Detection (no training needed)")
    print("""
  Running YOLOv8-Nano on a sample image.
  Uses pretrained weights -- no training required!
""")

    demo_lines = [
        "import cv2, urllib.request, numpy as np\n",
        "from ultralytics import YOLO\n",
        "\n",
        "print('Downloading YOLOv8-Nano weights (6MB)...')\n",
        "model = YOLO('yolov8n.pt')\n",
        "\n",
        "url = 'https://ultralytics.com/images/bus.jpg'\n",
        "print('Downloading sample image...')\n",
        "try:\n",
        "    urllib.request.urlretrieve(url, 'sample_demo.jpg')\n",
        "    results = model('sample_demo.jpg', conf=0.30, save=True, project='./results', name='demo')\n",
        "    print(f'Found {len(results[0].boxes)} objects! Saved to ./results/demo/')\n",
        "except Exception as e:\n",
        "    print(f'Network issue: {e} -- using blank test image instead')\n",
        "    dummy = np.zeros((480, 640, 3), dtype=np.uint8)\n",
        "    cv2.rectangle(dummy, (100,100),(300,350),(200,200,200),-1)\n",
        "    cv2.imwrite('sample_demo.jpg', dummy)\n",
        "    model('sample_demo.jpg', conf=0.10, save=True, project='./results', name='demo')\n",
        "\n",
        "print('[DONE] Demo complete!')\n",
    ]

    with open("_demo_temp.py", "w", encoding="utf-8") as f:
        f.writelines(demo_lines)

    run(f"{sys.executable} _demo_temp.py")
    try:
        os.remove("_demo_temp.py")
    except Exception:
        pass

    # STEP 4: Training
    header("Step 4/4: Training (Optional)")
    print("""
  To train on your own dataset:
    - Images in: data/images/train/  and  data/images/val/
    - Labels in: data/labels/train/  and  data/labels/val/
    - Edit:      configs/dataset.yaml  with your class names

  Training on CPU takes about 30-90 min for 30 epochs.
  For free GPU: open notebooks/YOLO_SelfDriving_Colab.ipynb on Colab
""")

    choice = input("  Start CPU training now? (y/N): ").strip().lower()
    if choice == "y":
        epochs = input("  How many epochs? (press Enter for 20): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 20
        print(f"\n  Starting {epochs}-epoch training on CPU...")
        run(
            f"{sys.executable} src/train_cpu.py "
            f"--data configs/dataset.yaml "
            f"--epochs {epochs} --batch 4 --imgsz 416"
        )
    else:
        print("\n  Skipping. Run it later with:")
        print("    python src/train_cpu.py --epochs 30")

    # Summary
    header("All Done! -- Quick Command Reference")
    print("""
  WEBCAM DEMO (works right now, no training needed):
    python src/detect_cpu.py --model yolov8n.pt --source 0

  TRAIN the model on your dataset:
    python src/train_cpu.py --epochs 30 --batch 4 --imgsz 416

  DETECT on a video:
    python src/detect_cpu.py --model yolov8n.pt --source your_video.mp4

  DETECT on an image:
    python src/detect_cpu.py --model yolov8n.pt --source image.jpg --mode image

  EVALUATE trained model:
    python src/evaluate_cpu.py --model runs/train/yolo_selfdriving_cpu/weights/best.pt --data configs/dataset.yaml

  FOR FREE GPU:
    Open notebooks/YOLO_SelfDriving_Colab.ipynb on Google Colab
    Runtime > Change runtime type > T4 GPU > Run All

  Project Code: SIC/AI/003
""")


if __name__ == "__main__":
    main()
