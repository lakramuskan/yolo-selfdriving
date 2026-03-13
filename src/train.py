"""
============================================================
Training Pipeline — YOLOv8 Object Detection for Self-Driving
Project Code: SIC/AI/003
Team: Naman Kumar, Aryan Kaushik, Muskan, Adhisha Pahuja, Sukesh Bhardwaj
============================================================

This script handles:
1. Loading YOLOv8 pre-trained weights
2. Fine-tuning on self-driving dataset
3. Model checkpointing
4. Training visualization
5. Exporting trained model (ONNX / TensorRT)
"""

import os
import sys
import time
import yaml
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Suppress minor warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ Missing dependencies. Run:  pip install ultralytics torch")
    sys.exit(1)


# ─────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "autorickshaw",
    "bus", "truck", "traffic_light", "stop_sign",
    "speed_breaker", "pothole", "animal", "construction"
]


# ─────────────────────────────────────────────
# SYSTEM INFO
# ─────────────────────────────────────────────
def print_system_info():
    """Display GPU/CPU info before training starts."""
    print("\n" + "=" * 60)
    print("  SYSTEM INFORMATION")
    print("=" * 60)
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device      : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Memory      : {mem:.1f} GB")
        print(f"  CUDA Version    : {torch.version.cuda}")
    else:
        print("  ⚠️  No GPU found — training will be slow on CPU")
        print("     Recommend: Use Google Colab (Runtime > Change Runtime > GPU)")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────
MODEL_OPTIONS = {
    "nano":   "yolov8n.pt",   # 3.2M params  — fastest, least accurate
    "small":  "yolov8s.pt",   # 11.2M params — good speed/accuracy balance
    "medium": "yolov8m.pt",   # 25.9M params — recommended for this project
    "large":  "yolov8l.pt",   # 43.7M params — high accuracy, needs more VRAM
    "xlarge": "yolov8x.pt",   # 68.2M params — best accuracy
}

def select_model(size: str = "medium") -> str:
    """Return model checkpoint name based on size."""
    model = MODEL_OPTIONS.get(size, MODEL_OPTIONS["medium"])
    print(f"📦 Using model: {model}")
    return model


# ─────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(
    dataset_yaml: str = "./configs/dataset.yaml",
    model_size: str = "medium",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,             # None = auto-detect
    project: str = "./runs/train",
    name: str = "yolo_selfdriving",
    resume: bool = False,
    pretrained: bool = True,
    workers: int = 4,
):
    """
    Train YOLOv8 on the self-driving object detection dataset.

    Args:
        dataset_yaml: Path to dataset.yaml config file
        model_size:   One of nano/small/medium/large/xlarge
        epochs:       Number of training epochs
        batch_size:   Batch size (-1 for auto)
        img_size:     Input image size
        device:       'cpu', '0', '0,1' (multi-GPU) or None for auto
        project:      Save directory
        name:         Experiment name
        resume:       Resume from last checkpoint
        pretrained:   Use COCO pretrained weights
        workers:      DataLoader workers
    """
    print_system_info()

    # Auto-select device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    # Load model
    model_weight = select_model(model_size)

    if resume:
        # Resume training from last checkpoint
        last_ckpt = Path(project) / name / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"🔄 Resuming from: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"⚠️  No checkpoint found at {last_ckpt}. Starting fresh.")
            model = YOLO(model_weight)
    else:
        model = YOLO(model_weight)

    print(f"\n🚀 Starting Training — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Dataset : {dataset_yaml}")
    print(f"   Epochs  : {epochs}")
    print(f"   Batch   : {batch_size}")
    print(f"   ImgSize : {img_size}×{img_size}")
    print(f"   Device  : {device}")
    print(f"   Output  : {project}/{name}\n")

    start_time = time.time()

    # ── TRAIN ──
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=pretrained,
        resume=resume,

        # Optimizer
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        degrees=0.0,
        translate=0.1,
        scale=0.5,

        # Output
        save=True,
        save_period=10,   # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,

        # Early stopping
        patience=20,
    )

    elapsed = (time.time() - start_time) / 60
    print(f"\n✅ Training complete in {elapsed:.1f} minutes")

    # Print best metrics
    best_map = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"   Best mAP@0.5 : {best_map:.4f}")

    return model, results


# ─────────────────────────────────────────────
# MODEL EXPORT
# ─────────────────────────────────────────────
def export_model(model_path: str, formats: list = ["onnx"]):
    """
    Export trained model to deployment formats.
    Supported: onnx, torchscript, tflite, coreml, engine (TensorRT)
    """
    print(f"\n📤 Exporting model: {model_path}")
    model = YOLO(model_path)

    for fmt in formats:
        try:
            print(f"   Exporting to {fmt.upper()}...")
            export_path = model.export(format=fmt, imgsz=640, optimize=True)
            print(f"   ✅ Saved: {export_path}")
        except Exception as e:
            print(f"   ❌ Failed {fmt}: {e}")


# ─────────────────────────────────────────────
# TRAINING CURVE PLOT
# ─────────────────────────────────────────────
def plot_training_results(results_csv: str, save_path: str = "./results/training_curves.png"):
    """
    Plot training loss and mAP curves from results.csv.
    """
    if not os.path.exists(results_csv):
        print(f"⚠️  results.csv not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # remove whitespace from column names

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("YOLOv8 Training Results — SIC/AI/003", fontsize=15, fontweight="bold")

    metrics = [
        ("train/box_loss",   "Train Box Loss",      "tab:blue"),
        ("train/cls_loss",   "Train Class Loss",    "tab:orange"),
        ("train/dfl_loss",   "Train DFL Loss",      "tab:green"),
        ("metrics/mAP50(B)", "mAP@0.5",             "tab:red"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95",    "tab:purple"),
        ("val/box_loss",     "Val Box Loss",         "tab:brown"),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(alpha=0.35)
        else:
            ax.set_title(f"{title} (N/A)")
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Training curves saved: {save_path}")


# ─────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Self-Driving Object Detection — SIC/AI/003"
    )
    parser.add_argument("--data",       default="./configs/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--model",      default="medium", choices=MODEL_OPTIONS.keys(), help="Model size")
    parser.add_argument("--epochs",     default=100, type=int,  help="Training epochs")
    parser.add_argument("--batch",      default=16,  type=int,  help="Batch size")
    parser.add_argument("--imgsz",      default=640, type=int,  help="Image size")
    parser.add_argument("--device",     default=None, help="Device: 0, cpu, 0,1")
    parser.add_argument("--project",    default="./runs/train", help="Save directory")
    parser.add_argument("--name",       default="yolo_selfdriving", help="Experiment name")
    parser.add_argument("--resume",     action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-pretrain",action="store_true", help="Train from scratch (no pretrained weights)")
    parser.add_argument("--workers",    default=4, type=int, help="DataLoader workers")
    parser.add_argument("--export",     default=None, nargs="+",
                        help="Export formats after training (e.g. onnx torchscript)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  SIC/AI/003 — YOLOv8 Self-Driving Object Detection")
    print("  Training Pipeline")
    print("=" * 60)

    # Train
    model, results = train(
        dataset_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=not args.no_pretrain,
        workers=args.workers,
    )

    # Plot training curves
    results_csv = Path(args.project) / args.name / "results.csv"
    plot_training_results(str(results_csv))

    # Export if requested
    if args.export:
        best_weights = Path(args.project) / args.name / "weights" / "best.pt"
        if best_weights.exists():
            export_model(str(best_weights), formats=args.export)
        else:
            print(f"⚠️  best.pt not found for export: {best_weights}")

    print("\n🏁 Done! Best weights saved at:")
    print(f"   {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
