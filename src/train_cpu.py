"""
============================================================
CPU-Optimized Training Pipeline — YOLOv8 Self-Driving
Project Code: SIC/AI/003
Team: Naman Kumar, Aryan Kaushik, Muskan, Adhisha Pahuja, Sukesh Bhardwaj
============================================================

OPTIMIZED FOR: CPU-only machines (no GPU required)
- Uses YOLOv8-Nano (fastest, smallest model — 3.2M params)
- Reduced image size (416px instead of 640px)
- Smaller batch size
- Fewer workers to avoid memory issues
- Transfer learning from COCO pretrained weights (saves huge training time)

TIP: If you have access to Google Colab, use the .ipynb notebook instead
     for ~10× faster training on free T4 GPU!
"""

import os
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ Run:  pip install ultralytics torch")
    sys.exit(1)


CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "autorickshaw",
    "bus", "truck", "traffic_light", "stop_sign",
    "speed_breaker", "pothole", "animal", "construction"
]


# ─────────────────────────────────────────────
# CPU SYSTEM INFO
# ─────────────────────────────────────────────
def print_system_info():
    print("\n" + "=" * 58)
    print("  SYSTEM INFORMATION")
    print("=" * 58)
    print(f"  PyTorch Version  : {torch.__version__}")
    print(f"  GPU Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on       : CPU")
        cpu_cores = os.cpu_count()
        print(f"  CPU Cores        : {cpu_cores}")
        print()
        print("  ⚠️  CPU Training Tips:")
        print("     • Use YOLOv8-Nano model (fastest)")
        print("     • Use 416×416 image size (not 640)")
        print("     • Expect ~2–5 min/epoch on a modern CPU")
        print("     • For submission demo, 20–30 epochs is enough")
        print("     • Google Colab gives FREE GPU — use the .ipynb!")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────
# CPU-OPTIMIZED TRAINING
# ─────────────────────────────────────────────
def train_cpu(
    dataset_yaml: str = "./configs/dataset.yaml",
    epochs: int = 30,
    batch_size: int = 4,
    img_size: int = 416,
    project: str = "./runs/train",
    name: str = "yolo_selfdriving_cpu",
    resume: bool = False,
):
    """
    CPU-optimized YOLOv8 training.
    Uses YOLOv8-Nano with reduced image size & batch for memory efficiency.

    Expected training time on CPU:
    - 20 epochs ≈ 30–60 min  (good for demo)
    - 50 epochs ≈ 2–3 hours  (better accuracy)
    """
    print_system_info()

    device = "cpu"

    if resume:
        last_ckpt = Path(project) / name / "weights" / "last.pt"
        model = YOLO(str(last_ckpt)) if last_ckpt.exists() else YOLO("yolov8n.pt")
        print(f"🔄 Resuming from: {last_ckpt}")
    else:
        # Nano = smallest & fastest — perfect for CPU
        model = YOLO("yolov8n.pt")

    print(f"🚀 CPU Training — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Model   : YOLOv8-Nano (3.2M params — fastest)")
    print(f"   Dataset : {dataset_yaml}")
    print(f"   Epochs  : {epochs}")
    print(f"   Batch   : {batch_size}  (CPU-safe)")
    print(f"   ImgSize : {img_size}×{img_size}  (reduced for speed)")
    print(f"   Device  : CPU\n")

    est_min = epochs * 2.5  # ~2.5 min/epoch on avg CPU
    print(f"   ⏱️  Estimated time: {est_min:.0f}–{est_min*1.5:.0f} minutes on CPU")
    print(f"   💡 Tip: Open notebooks/YOLO_SelfDriving_Colab.ipynb for free GPU!\n")

    start = time.time()

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=2,          # Low workers prevents CPU overload
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,    # MUST use pretrained — saves massive training time on CPU

        # Optimizer — AdamW converges faster (fewer epochs needed)
        optimizer="AdamW",
        lr0=0.005,          # Slightly lower LR for stable CPU training
        lrf=0.0005,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2,    # Shorter warmup

        # Lighter augmentation (faster per-epoch on CPU)
        mosaic=0.5,         # Reduced mosaic probability
        mixup=0.0,          # Disable mixup (expensive on CPU)
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Output
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
        patience=10,        # Shorter patience for CPU runs
        resume=resume,
    )

    elapsed = (time.time() - start) / 60
    best_map = results.results_dict.get("metrics/mAP50(B)", 0)

    print(f"\n✅ Training complete in {elapsed:.1f} minutes")
    print(f"   Best mAP@0.5 : {best_map:.4f}")
    print(f"   Weights saved: {project}/{name}/weights/best.pt")

    return model, results


# ─────────────────────────────────────────────
# TRAINING CURVE PLOT
# ─────────────────────────────────────────────
def plot_training_curves(results_csv: str, save_path: str = "./results/training_curves.png"):
    if not os.path.exists(results_csv):
        print(f"⚠️  results.csv not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("YOLOv8 Training Curves — SIC/AI/003 (CPU Run)", fontsize=14, fontweight="bold")

    plots = [
        ("train/box_loss",      "Train Box Loss",    "royalblue"),
        ("train/cls_loss",      "Train Class Loss",  "darkorange"),
        ("train/dfl_loss",      "Train DFL Loss",    "seagreen"),
        ("metrics/mAP50(B)",    "mAP @ 0.5",         "crimson"),
        ("metrics/mAP50-95(B)", "mAP @ 0.5:0.95",   "purple"),
        ("val/box_loss",        "Val Box Loss",       "saddlebrown"),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), plots):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2, marker="o", markersize=3)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.35)
        else:
            ax.text(0.5, 0.5, f"{title}\n(not available)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Training curves saved: {save_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CPU Training — SIC/AI/003")
    parser.add_argument("--data",    default="./configs/dataset.yaml")
    parser.add_argument("--epochs",  default=30,  type=int, help="Epochs (20-30 for demo, 50+ for accuracy)")
    parser.add_argument("--batch",   default=4,   type=int, help="Batch size (keep 4-8 on CPU)")
    parser.add_argument("--imgsz",   default=416, type=int, help="Image size (416 or 320 for CPU)")
    parser.add_argument("--project", default="./runs/train")
    parser.add_argument("--name",    default="yolo_selfdriving_cpu")
    parser.add_argument("--resume",  action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 58)
    print("  SIC/AI/003 — CPU-Optimized YOLOv8 Training")
    print("=" * 58)

    model, results = train_cpu(
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    csv = Path(args.project) / args.name / "results.csv"
    plot_training_curves(str(csv), "./results/training_curves.png")

    print("\n🏁 Done! Run detection demo:")
    print(f"   python src/detect.py --model {args.project}/{args.name}/weights/best.pt --source 0")


if __name__ == "__main__":
    main()
