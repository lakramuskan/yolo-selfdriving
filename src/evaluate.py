"""
============================================================
Model Evaluation & Metrics Analysis
Project: Object Detection for Self-Driving using YOLO
Project Code: SIC/AI/003
Team: Naman Kumar, Aryan Kaushik, Muskan, Adhisha Pahuja, Sukesh Bhardwaj
============================================================

This script handles:
1. Model evaluation on validation / test set
2. Per-class mAP, Precision, Recall, F1 computation
3. Confusion matrix generation
4. Speed benchmarking (FPS / latency)
5. Comparative analysis of multiple models
6. Results reporting & export
"""

import os
import sys
import cv2
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ Missing deps. Run:  pip install ultralytics torch")
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
# CORE EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model_path: str, dataset_yaml: str, split: str = "val",
                   conf: float = 0.001, iou: float = 0.60,
                   save_dir: str = "./results/eval"):
    """
    Run full evaluation using Ultralytics built-in validator.
    Computes: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 per class.

    Args:
        model_path:   Path to .pt weights
        dataset_yaml: Dataset config YAML
        split:        'val' or 'test'
        conf:         Confidence threshold (use low value for full PR curve)
        iou:          IoU threshold for matching
        save_dir:     Directory to save results
    """
    print("\n" + "=" * 60)
    print(f"  EVALUATING MODEL: {Path(model_path).name}")
    print(f"  Dataset: {dataset_yaml} | Split: {split}")
    print("=" * 60)

    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    metrics = model.val(
        data=dataset_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=640,
        plots=True,
        save_json=True,
        project=save_dir,
        name="eval_run",
        verbose=True,
    )

    # ── PRINT RESULTS ──
    print("\n" + "─" * 60)
    print(f"  OVERALL METRICS")
    print("─" * 60)
    print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")
    f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-10)
    print(f"  F1 Score       : {f1:.4f}")
    print("─" * 60)

    # Per-class results
    if hasattr(metrics.box, "ap_class_index"):
        print("\n  PER-CLASS RESULTS:")
        print(f"  {'Class':<18} {'AP@0.5':>8} {'Precision':>10} {'Recall':>8}")
        print("  " + "-" * 46)
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
            ap   = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            prec = metrics.box.p[i]    if i < len(metrics.box.p)    else 0
            rec  = metrics.box.r[i]    if i < len(metrics.box.r)    else 0
            print(f"  {name:<18} {ap:>8.4f} {prec:>10.4f} {rec:>8.4f}")

    return metrics


# ─────────────────────────────────────────────
# SPEED BENCHMARK
# ─────────────────────────────────────────────
def benchmark_speed(model_path: str, img_size: int = 640,
                    n_warmup: int = 10, n_runs: int = 100,
                    device: str = None):
    """
    Benchmark model inference speed.
    Reports: FPS, mean latency, min/max latency, throughput.
    """
    print(f"\n⏱️  Speed Benchmark: {Path(model_path).name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)

    # Create dummy input
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # Warmup
    print(f"   Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        model(dummy, verbose=False)

    # Benchmark
    latencies = []
    print(f"   Benchmarking ({n_runs} runs)...")
    for _ in tqdm(range(n_runs), desc="Benchmarking"):
        t0 = time.perf_counter()
        model(dummy, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)

    print("\n" + "─" * 45)
    print(f"  SPEED BENCHMARK RESULTS ({device.upper()})")
    print("─" * 45)
    print(f"  Mean Latency    : {latencies.mean():.2f} ms")
    print(f"  Std Dev         : {latencies.std():.2f} ms")
    print(f"  Min Latency     : {latencies.min():.2f} ms")
    print(f"  Max Latency     : {latencies.max():.2f} ms")
    print(f"  P95 Latency     : {np.percentile(latencies, 95):.2f} ms")
    print(f"  Mean FPS        : {1000/latencies.mean():.1f}")
    print(f"  Target (≥30fps) : {'✅ PASS' if 1000/latencies.mean() >= 30 else '❌ FAIL'}")
    print("─" * 45)

    return {
        "mean_ms":  latencies.mean(),
        "std_ms":   latencies.std(),
        "min_ms":   latencies.min(),
        "max_ms":   latencies.max(),
        "p95_ms":   np.percentile(latencies, 95),
        "mean_fps": 1000 / latencies.mean(),
    }


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(metrics, save_path: str = "./results/confusion_matrix.png"):
    """
    Plot and save a styled confusion matrix from evaluation metrics.
    """
    if not hasattr(metrics, "confusion_matrix") or metrics.confusion_matrix is None:
        print("⚠️  Confusion matrix not available in metrics.")
        return

    cm = metrics.confusion_matrix.matrix
    n  = cm.shape[0]

    # Class labels (add "background" for last row/col if nc+1)
    labels = CLASS_NAMES[:n-1] + ["background"] if n > len(CLASS_NAMES) else CLASS_NAMES[:n]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("Confusion Matrix — YOLOv8 Self-Driving Detection\nSIC/AI/003",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Confusion matrix saved: {save_path}")


# ─────────────────────────────────────────────
# PER-CLASS BAR CHART
# ─────────────────────────────────────────────
def plot_per_class_metrics(metrics, save_path: str = "./results/per_class_metrics.png"):
    """
    Plot per-class Precision, Recall, and AP@0.5 as grouped bar chart.
    """
    if not hasattr(metrics.box, "ap_class_index"):
        print("⚠️  Per-class metrics not available.")
        return

    names, ap50s, precs, recs = [], [], [], []
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        names.append(CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}")
        ap50s.append(metrics.box.ap50[i]  if i < len(metrics.box.ap50) else 0)
        precs.append(metrics.box.p[i]     if i < len(metrics.box.p)    else 0)
        recs.append(metrics.box.r[i]      if i < len(metrics.box.r)    else 0)

    x     = np.arange(len(names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, ap50s, width, label="AP@0.5",    color="#2E75B6", alpha=0.88)
    ax.bar(x,         precs, width, label="Precision",  color="#70AD47", alpha=0.88)
    ax.bar(x + width, recs,  width, label="Recall",     color="#ED7D31", alpha=0.88)

    ax.set_xlabel("Object Class", fontsize=12)
    ax.set_ylabel("Score (0–1)", fontsize=12)
    ax.set_title("Per-Class Metrics — AP@0.5 / Precision / Recall\nSIC/AI/003",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1.2, label="Target (0.75)")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Per-class metrics chart saved: {save_path}")


# ─────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────
def compare_models(model_configs: list, dataset_yaml: str,
                   save_path: str = "./results/model_comparison.png"):
    """
    Compare multiple trained models on mAP, speed, and model size.

    Args:
        model_configs: List of {"name": str, "path": str} dicts
        dataset_yaml:  Dataset config
        save_path:     Output chart path
    """
    print("\n📊 Comparing models...")
    results = []

    for cfg in model_configs:
        print(f"\n   Testing: {cfg['name']}")
        model_path = cfg["path"]
        if not os.path.exists(model_path):
            print(f"   ⚠️  Not found: {model_path}")
            continue

        # Size
        size_mb = os.path.getsize(model_path) / (1024 ** 2)

        # Speed
        speed = benchmark_speed(model_path, n_warmup=5, n_runs=30)

        # mAP
        model = YOLO(model_path)
        val_metrics = model.val(data=dataset_yaml, verbose=False, plots=False)
        map50 = val_metrics.box.map50

        results.append({
            "Model":    cfg["name"],
            "mAP@0.5": round(map50, 4),
            "FPS":      round(speed["mean_fps"], 1),
            "Size (MB)":round(size_mb, 1),
        })

    if not results:
        return

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison — SIC/AI/003", fontsize=14, fontweight="bold")
    colors = ["#2E75B6", "#70AD47", "#ED7D31", "#9E4EBB"][:len(df)]

    for ax, col, title, color_idx in zip(
        axes,
        ["mAP@0.5", "FPS", "Size (MB)"],
        ["mAP@0.5 (↑ higher is better)", "FPS (↑ faster is better)", "Model Size MB (↓ smaller is better)"],
        range(3)
    ):
        ax.bar(df["Model"], df[col], color=colors[:len(df)], alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.35)
        for bar, val in zip(ax.patches, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Model comparison chart saved: {save_path}")

    # Save CSV
    csv_path = save_path.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Results CSV saved: {csv_path}")

    return df


# ─────────────────────────────────────────────
# GENERATE FULL REPORT
# ─────────────────────────────────────────────
def generate_report(model_path: str, dataset_yaml: str, speed_stats: dict,
                    metrics, save_path: str = "./results/eval_report.json"):
    """
    Export a comprehensive JSON evaluation report.
    """
    report = {
        "project":    "SIC/AI/003 — Object Detection for Self-Driving",
        "model":      Path(model_path).name,
        "dataset":    dataset_yaml,
        "timestamp":  __import__("datetime").datetime.now().isoformat(),
        "overall_metrics": {
            "mAP_at_50":      round(metrics.box.map50, 4),
            "mAP_at_50_95":   round(metrics.box.map,   4),
            "precision":      round(metrics.box.mp,     4),
            "recall":         round(metrics.box.mr,     4),
            "f1_score":       round(2 * metrics.box.mp * metrics.box.mr /
                                    (metrics.box.mp + metrics.box.mr + 1e-10), 4),
        },
        "speed_benchmark": speed_stats,
        "targets_met": {
            "mAP_target_0.75": metrics.box.map50 >= 0.75,
            "fps_target_30":   speed_stats.get("mean_fps", 0) >= 30,
        }
    }

    # Per-class
    per_class = {}
    if hasattr(metrics.box, "ap_class_index"):
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
            per_class[name] = {
                "ap50":      round(float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0, 4),
                "precision": round(float(metrics.box.p[i])    if i < len(metrics.box.p)    else 0, 4),
                "recall":    round(float(metrics.box.r[i])    if i < len(metrics.box.r)    else 0, 4),
            }
    report["per_class_metrics"] = per_class

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Full report saved: {save_path}")
    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation — SIC/AI/003")
    parser.add_argument("--model",    required=True, help="Path to .pt weights")
    parser.add_argument("--data",     default="./configs/dataset.yaml", help="Dataset YAML")
    parser.add_argument("--split",    default="val", choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--conf",     default=0.001, type=float, help="Confidence threshold")
    parser.add_argument("--iou",      default=0.60,  type=float, help="IoU threshold")
    parser.add_argument("--save-dir", default="./results/eval", help="Results directory")
    parser.add_argument("--benchmark",action="store_true", help="Run speed benchmark")
    parser.add_argument("--report",   action="store_true", help="Generate JSON report")
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # Full evaluation
    metrics = evaluate_model(
        model_path=args.model,
        dataset_yaml=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_dir=args.save_dir,
    )

    # Plots
    plot_per_class_metrics(metrics, save_path=f"{args.save_dir}/per_class_metrics.png")
    plot_confusion_matrix(metrics,  save_path=f"{args.save_dir}/confusion_matrix.png")

    # Speed benchmark
    speed_stats = {}
    if args.benchmark:
        speed_stats = benchmark_speed(args.model)

    # JSON report
    if args.report:
        generate_report(args.model, args.data, speed_stats, metrics,
                        save_path=f"{args.save_dir}/eval_report.json")

    print("\n🏁 Evaluation complete!")


if __name__ == "__main__":
    main()
