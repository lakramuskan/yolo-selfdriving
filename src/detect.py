"""
============================================================
Real-Time Object Detection — Inference & Demo
Project: Object Detection for Self-Driving using YOLO
Project Code: SIC/AI/003
Team: Naman Kumar, Aryan Kaushik, Muskan, Adhisha Pahuja, Sukesh Bhardwaj
============================================================

This script handles:
1. Real-time webcam / video detection
2. Single image inference
3. Batch image inference
4. Performance monitoring (FPS, latency)
5. Detection result saving
"""

import os
import sys
import cv2
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ Missing dependencies. Run:  pip install ultralytics torch opencv-python")
    sys.exit(1)


# ─────────────────────────────────────────────
# CLASS CONFIGURATION
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "autorickshaw",
    "bus", "truck", "traffic_light", "stop_sign",
    "speed_breaker", "pothole", "animal", "construction"
]

# Assign distinct colors for each class (BGR)
CLASS_COLORS = [
    (0, 255, 0),       # person       — green
    (255, 128, 0),     # bicycle      — orange
    (0, 128, 255),     # car          — blue
    (255, 0, 128),     # motorcycle   — pink
    (128, 0, 255),     # autorickshaw — purple
    (0, 255, 255),     # bus          — cyan
    (255, 255, 0),     # truck        — yellow
    (0, 0, 255),       # traffic_light— red
    (50, 50, 200),     # stop_sign    — dark red
    (128, 128, 0),     # speed_breaker— olive
    (0, 128, 128),     # pothole      — teal
    (0, 80, 180),      # animal       — dark orange
    (180, 180, 180),   # construction — gray
]

# Danger classes that trigger warnings
DANGER_CLASSES = {"person", "animal", "speed_breaker", "pothole", "stop_sign", "traffic_light"}


# ─────────────────────────────────────────────
# FPS TRACKER
# ─────────────────────────────────────────────
class FPSTracker:
    """Smooth FPS calculation using a rolling window."""
    def __init__(self, window: int = 30):
        self.times = deque(maxlen=window)
        self.last_time = time.time()

    def update(self):
        now = time.time()
        self.times.append(now - self.last_time)
        self.last_time = now

    @property
    def fps(self) -> float:
        if len(self.times) == 0:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


# ─────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────
def draw_detection(frame: np.ndarray, box, cls_id: int, conf: float,
                   thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box with label and confidence on the frame.
    """
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
    name  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    label = f"{name}: {conf:.2f}"
    (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), color, -1)

    # Label text
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_hud(frame: np.ndarray, fps: float, n_detections: int,
             detection_counts: dict, danger: bool) -> np.ndarray:
    """
    Draw Heads-Up Display overlay with metrics on the frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # FPS
    fps_color = (0, 255, 80) if fps >= 25 else (0, 200, 255) if fps >= 15 else (0, 80, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

    # Detection count
    cv2.putText(frame, f"Objects: {n_detections}", (150, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Project tag
    tag = "SIC/AI/003 — YOLOv8 Self-Driving Detection"
    cv2.putText(frame, tag, (w - 500, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 220), 1)

    # Danger alert
    if danger:
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (0, h - 55), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(alert_overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "⚠  HAZARD DETECTED — CAUTION", (w // 2 - 240, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 80), 2)

    # Detection summary (bottom-left)
    y_offset = h - 60
    if detection_counts and not danger:
        for cls_name, count in list(detection_counts.items())[:5]:
            cls_id = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            cv2.putText(frame, f"  {cls_name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 22

    return frame


# ─────────────────────────────────────────────
# SINGLE IMAGE INFERENCE
# ─────────────────────────────────────────────
def detect_image(model_path: str, image_path: str, conf: float = 0.35,
                 iou: float = 0.45, save_dir: str = "./results/detections",
                 show: bool = True):
    """
    Run inference on a single image and save the annotated result.
    """
    print(f"\n🔍 Running detection on: {image_path}")
    model = YOLO(model_path)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not read image: {image_path}")
        return

    start = time.time()
    results = model(frame, conf=conf, iou=iou, verbose=False)[0]
    latency_ms = (time.time() - start) * 1000

    # Draw detections
    detection_counts = defaultdict(int)
    for box in results.boxes:
        cls_id = int(box.cls)
        conf_score = float(box.conf)
        draw_detection(frame, box.xyxy[0], cls_id, conf_score)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        detection_counts[cls_name] += 1

    n_det = len(results.boxes)

    # Draw info text
    cv2.putText(frame, f"Detections: {n_det} | Latency: {latency_ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    cv2.putText(frame, "SIC/AI/003 — YOLOv8",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 220), 1)

    # Save result
    os.makedirs(save_dir, exist_ok=True)
    stem = Path(image_path).stem
    out_path = os.path.join(save_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_path, frame)

    # Print summary
    print(f"   Detections  : {n_det}")
    print(f"   Latency     : {latency_ms:.1f} ms")
    print(f"   Saved to    : {out_path}")
    if detection_counts:
        print("   Classes     :", dict(detection_counts))

    if show:
        cv2.imshow("Detection Result — SIC/AI/003", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frame, detection_counts


# ─────────────────────────────────────────────
# BATCH IMAGE INFERENCE
# ─────────────────────────────────────────────
def detect_batch(model_path: str, images_dir: str, conf: float = 0.35,
                 iou: float = 0.45, save_dir: str = "./results/batch"):
    """
    Run inference on all images in a directory.
    """
    model = YOLO(model_path)
    image_files = list(Path(images_dir).glob("*.jpg")) + \
                  list(Path(images_dir).glob("*.png")) + \
                  list(Path(images_dir).glob("*.jpeg"))

    print(f"\n📦 Batch detection: {len(image_files)} images from {images_dir}")
    os.makedirs(save_dir, exist_ok=True)

    total_det = 0
    latencies = []

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        start = time.time()
        results = model(frame, conf=conf, iou=iou, verbose=False)[0]
        latencies.append((time.time() - start) * 1000)

        for box in results.boxes:
            draw_detection(frame, box.xyxy[0], int(box.cls), float(box.conf))

        total_det += len(results.boxes)
        out_path = os.path.join(save_dir, img_path.name)
        cv2.imwrite(out_path, frame)

    print(f"   Total detections : {total_det}")
    print(f"   Avg latency      : {np.mean(latencies):.1f} ms per image")
    print(f"   Avg FPS          : {1000/np.mean(latencies):.1f}")
    print(f"   Results saved to : {save_dir}")


# ─────────────────────────────────────────────
# REAL-TIME VIDEO / WEBCAM DETECTION
# ─────────────────────────────────────────────
def detect_realtime(model_path: str, source=0, conf: float = 0.35,
                    iou: float = 0.45, save_output: bool = False,
                    output_path: str = "./results/output.mp4"):
    """
    Run real-time object detection on webcam or video file.

    Args:
        model_path:   Path to YOLOv8 .pt weights
        source:       0 = webcam, or path to video file
        conf:         Confidence threshold (0-1)
        iou:          IoU threshold for NMS
        save_output:  Whether to save output video
        output_path:  Output video path
    """
    print(f"\n🎥 Starting real-time detection — SIC/AI/003")
    print(f"   Model  : {model_path}")
    print(f"   Source : {'Webcam' if source == 0 else source}")
    print(f"   Conf   : {conf}  |  IoU : {iou}")
    print("   Press Q to quit | S to screenshot | P to pause\n")

    # Load model
    model = YOLO(model_path)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        return

    # Get video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"   Resolution : {frame_w}×{frame_h}")
    print(f"   Source FPS : {source_fps:.1f}")

    # Video writer
    writer = None
    if save_output:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, source_fps, (frame_w, frame_h))
        print(f"   Saving to  : {output_path}")

    fps_tracker = FPSTracker(window=30)
    frame_count  = 0
    paused       = False
    screenshot_n = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠️  Stream ended.")
                    break
                frame_count += 1

            # ── INFERENCE ──
            results = model(frame, conf=conf, iou=iou, verbose=False)[0]

            # ── DRAW DETECTIONS ──
            detection_counts = defaultdict(int)
            danger_detected  = False

            for box in results.boxes:
                cls_id     = int(box.cls)
                conf_score = float(box.conf)
                cls_name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

                draw_detection(frame, box.xyxy[0], cls_id, conf_score)
                detection_counts[cls_name] += 1

                if cls_name in DANGER_CLASSES:
                    danger_detected = True

            # ── UPDATE & DRAW HUD ──
            fps_tracker.update()
            frame = draw_hud(frame, fps_tracker.fps, len(results.boxes),
                             dict(detection_counts), danger_detected)

            # ── DISPLAY ──
            cv2.imshow("SIC/AI/003 — YOLOv8 Self-Driving Detection", frame)

            # ── SAVE FRAME ──
            if writer:
                writer.write(frame)

            # ── KEYBOARD CONTROLS ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 Quit by user.")
                break
            elif key == ord("s"):
                ss_path = f"./results/screenshot_{screenshot_n:04d}.jpg"
                os.makedirs("./results", exist_ok=True)
                cv2.imwrite(ss_path, frame)
                print(f"📸 Screenshot saved: {ss_path}")
                screenshot_n += 1
            elif key == ord("p"):
                paused = not paused
                print("⏸  Paused" if paused else "▶️  Resumed")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Processed {frame_count} frames")
        print(f"   Avg FPS: {fps_tracker.fps:.1f}")


# ─────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Object Detection — SIC/AI/003"
    )
    parser.add_argument("--model",  required=True, help="Path to .pt weights file")
    parser.add_argument("--source", default="0",   help="'0'=webcam, or video/image path")
    parser.add_argument("--conf",   default=0.35,  type=float, help="Confidence threshold")
    parser.add_argument("--iou",    default=0.45,  type=float, help="IoU threshold")
    parser.add_argument("--mode",   default="realtime",
                        choices=["realtime", "image", "batch"],
                        help="Detection mode")
    parser.add_argument("--save",   action="store_true", help="Save output video/images")
    parser.add_argument("--output", default="./results/output.mp4", help="Output path")
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # Convert source to int if webcam
    source = int(args.source) if args.source.isdigit() else args.source

    if args.mode == "realtime":
        detect_realtime(
            model_path=args.model,
            source=source,
            conf=args.conf,
            iou=args.iou,
            save_output=args.save,
            output_path=args.output,
        )
    elif args.mode == "image":
        detect_image(
            model_path=args.model,
            image_path=str(source),
            conf=args.conf,
            iou=args.iou,
            save_dir="./results/detections",
        )
    elif args.mode == "batch":
        detect_batch(
            model_path=args.model,
            images_dir=str(source),
            conf=args.conf,
            iou=args.iou,
            save_dir="./results/batch",
        )


if __name__ == "__main__":
    main()
