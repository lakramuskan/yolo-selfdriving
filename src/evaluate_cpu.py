"""
============================================================
Evaluation & Metrics — CPU Optimized
Project Code: SIC/AI/003
Team: Naman Kumar, Aryan Kaushik, Muskan, Adhisha Pahuja, Sukesh Bhardwaj
============================================================
"""

import os
import sys
import cv2
import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

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
# MAIN EVALUATION
# ─────────────────────────────────────────────
def evaluate(model_path, dataset_yaml, split="val",
             conf=0.001, iou=0.60, save_dir="./results/eval"):
    """Run full evaluation. Uses CPU automatically."""
    print("\n" + "=" * 58)
    print(f"  EVALUATING: {Path(model_path).name}")
    print(f"  Dataset: {dataset_yaml}  |  Split: {split}")
    print("=" * 58)

    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    metrics = model.val(
        data=dataset_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=416,          # 416 for CPU speed
        device="cpu",
        plots=True,
        save_json=True,
        project=save_dir,
        name="eval_run",
        verbose=True,
    )

    # Print results table
    mp   = metrics.box.mp
    mr   = metrics.box.mr
    map50 = metrics.box.map50
    map   = metrics.box.map
    f1   = 2 * mp * mr / (mp + mr + 1e-10)

    print("\n" + "─" * 50)
    print("  RESULTS")
    print("─" * 50)
    print(f"  mAP @ 0.5      : {map50:.4f}  {'✅' if map50 >= 0.75 else '⚠️ (target: 0.75)'}")
    print(f"  mAP @ 0.5:0.95 : {map:.4f}")
    print(f"  Precision      : {mp:.4f}")
    print(f"  Recall         : {mr:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print("─" * 50)

    if hasattr(metrics.box, "ap_class_index"):
        print("\n  PER-CLASS:")
        print(f"  {'Class':<18} {'AP@0.5':>8} {'Prec':>8} {'Recall':>8}")
        print("  " + "-" * 44)
        for i, ci in enumerate(metrics.box.ap_class_index):
            name = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else f"cls_{ci}"
            ap   = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            p    = metrics.box.p[i]    if i < len(metrics.box.p)    else 0
            r    = metrics.box.r[i]    if i < len(metrics.box.r)    else 0
            print(f"  {name:<18} {ap:>8.4f} {p:>8.4f} {r:>8.4f}")

    return metrics


# ─────────────────────────────────────────────
# CPU SPEED BENCHMARK
# ─────────────────────────────────────────────
def benchmark_cpu(model_path, img_size=320, n_warmup=5, n_runs=30):
    """Benchmark CPU inference speed."""
    print(f"\n⏱️  CPU Speed Benchmark ({img_size}×{img_size} input)")
    model = YOLO(model_path)
    model.to("cpu")

    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    print(f"   Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        model(dummy, verbose=False, device="cpu")

    latencies = []
    for _ in tqdm(range(n_runs), desc="   Benchmarking"):
        t0 = time.perf_counter()
        model(dummy, verbose=False, device="cpu")
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    fps = 1000 / lat.mean()

    print(f"\n  Mean Latency  : {lat.mean():.1f} ms")
    print(f"  Std Dev       : {lat.std():.1f} ms")
    print(f"  Min / Max     : {lat.min():.1f} / {lat.max():.1f} ms")
    print(f"  P95 Latency   : {np.percentile(lat, 95):.1f} ms")
    print(f"  Mean FPS      : {fps:.1f}")
    print(f"  Target ≥30fps : {'✅ PASS' if fps >= 30 else f'⚠️ {fps:.1f} FPS (use GPU for 30+ FPS)'}")

    return {"mean_ms": lat.mean(), "mean_fps": fps, "p95_ms": np.percentile(lat, 95)}


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def plot_per_class(metrics, save_path="./results/eval/per_class.png"):
    if not hasattr(metrics.box, "ap_class_index"):
        return

    names, ap50s, precs, recs = [], [], [], []
    for i, ci in enumerate(metrics.box.ap_class_index):
        names.append(CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else f"cls_{ci}")
        ap50s.append(float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0)
        precs.append(float(metrics.box.p[i])    if i < len(metrics.box.p)    else 0)
        recs.append(float(metrics.box.r[i])     if i < len(metrics.box.r)    else 0)

    if not names:
        return

    x, w = np.arange(len(names)), 0.27
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 6))
    ax.bar(x - w, ap50s, w, label="AP@0.5",   color="#2E75B6", alpha=0.86)
    ax.bar(x,     precs, w, label="Precision", color="#70AD47", alpha=0.86)
    ax.bar(x + w, recs,  w, label="Recall",    color="#ED7D31", alpha=0.86)
    ax.axhline(0.75, color="red", ls="--", lw=1.3, label="Target (0.75)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics — SIC/AI/003", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Chart saved: {save_path}")


def plot_confusion_matrix(metrics, save_path="./results/eval/confusion_matrix.png"):
    if not hasattr(metrics, "confusion_matrix") or metrics.confusion_matrix is None:
        print("⚠️  Confusion matrix not in metrics (may need more validation images).")
        return

    cm = metrics.confusion_matrix.matrix
    n  = cm.shape[0]
    labels = CLASS_NAMES[:n-1] + ["background"] if n > len(CLASS_NAMES) else CLASS_NAMES[:n]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, linewidths=0.4, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("Confusion Matrix — SIC/AI/003", fontsize=13, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Confusion matrix saved: {save_path}")


# ─────────────────────────────────────────────
# JSON REPORT
# ─────────────────────────────────────────────
def save_report(model_path, dataset_yaml, metrics, speed, save_path="./results/eval/report.json"):
    mp = metrics.box.mp
    mr = metrics.box.mr
    report = {
        "project":   "SIC/AI/003 — Object Detection for Self-Driving",
        "model":     Path(model_path).name,
        "device":    "CPU",
        "dataset":   dataset_yaml,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "metrics": {
            "mAP_50":     round(metrics.box.map50, 4),
            "mAP_50_95":  round(metrics.box.map,   4),
            "precision":  round(mp, 4),
            "recall":     round(mr, 4),
            "f1":         round(2 * mp * mr / (mp + mr + 1e-10), 4),
        },
        "speed": speed,
        "targets_met": {
            "mAP_0.75": metrics.box.map50 >= 0.75,
            "fps_30":   speed.get("mean_fps", 0) >= 30,
        },
    }

    per_cls = {}
    if hasattr(metrics.box, "ap_class_index"):
        for i, ci in enumerate(metrics.box.ap_class_index):
            name = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else f"cls_{ci}"
            per_cls[name] = {
                "ap50": round(float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0, 4),
                "precision": round(float(metrics.box.p[i]) if i < len(metrics.box.p) else 0, 4),
                "recall": round(float(metrics.box.r[i]) if i < len(metrics.box.r) else 0, 4),
            }
    report["per_class"] = per_cls

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved: {save_path}")
    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CPU Evaluation — SIC/AI/003")
    parser.add_argument("--model",     required=True)
    parser.add_argument("--data",      default="./configs/dataset.yaml")
    parser.add_argument("--split",     default="val", choices=["val", "test"])
    parser.add_argument("--save-dir",  default="./results/eval")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--report",    action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    metrics = evaluate(args.model, args.data, args.split, save_dir=args.save_dir)
    plot_per_class(metrics,        f"{args.save_dir}/per_class.png")
    plot_confusion_matrix(metrics, f"{args.save_dir}/confusion_matrix.png")

    speed = {}
    if args.benchmark:
        speed = benchmark_cpu(args.model, img_size=320)

    if args.report:
        save_report(args.model, args.data, metrics, speed,
                    f"{args.save_dir}/report.json")

    print("\n🏁 Evaluation complete!")


if __name__ == "__main__":
    main()
