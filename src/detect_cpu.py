"""
============================================================
Real-Time Inference Demo — CPU Optimized
Project Code: SIC/AI/003
Team: Naman Kumar, Aryan Kaushik, Muskan, Adhisha Pahuja, Sukesh Bhardwaj
============================================================

CPU MODE FEATURES:
- Frame skipping to maintain smooth display
- Resolution scaling for faster inference
- Threaded frame capture to avoid lag
- Works without CUDA / GPU
- Auto-saves annotated video
"""

import os
import sys
import cv2
import time
import threading
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import deque, defaultdict

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ Run:  pip install ultralytics torch opencv-python")
    sys.exit(1)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "autorickshaw",
    "bus", "truck", "traffic_light", "stop_sign",
    "speed_breaker", "pothole", "animal", "construction"
]

CLASS_COLORS = [
    (0, 255, 0),     (255, 128, 0),   (0, 128, 255),
    (255, 0, 128),   (128, 0, 255),   (0, 255, 255),
    (255, 255, 0),   (0, 0, 255),     (50, 50, 200),
    (128, 128, 0),   (0, 128, 128),   (0, 80, 180),
    (180, 180, 180),
]

DANGER_CLASSES = {"person", "animal", "speed_breaker", "pothole", "traffic_light", "stop_sign"}


# ─────────────────────────────────────────────
# THREADED FRAME READER (avoids lag on CPU)
# ─────────────────────────────────────────────
class ThreadedCapture:
    """Read frames in a background thread to prevent frame queue buildup."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def release(self):
        self.running = False
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()


# ─────────────────────────────────────────────
# FPS TRACKER
# ─────────────────────────────────────────────
class FPSTracker:
    def __init__(self, window=20):
        self.times = deque(maxlen=window)
        self.t = time.time()

    def update(self):
        now = time.time()
        self.times.append(now - self.t)
        self.t = now

    @property
    def fps(self):
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────
def draw_box(frame, box, cls_id, conf):
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
    name  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
    label = f"{name} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(frame, (x1, y1 - lh - bl - 4), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    return frame


def draw_hud(frame, fps, n_det, counts, danger):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 20, 40), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # FPS color: green=fast, yellow=ok, red=slow
    fps_color = (0, 230, 60) if fps >= 15 else (0, 200, 255) if fps >= 8 else (0, 60, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
    cv2.putText(frame, f"Objects: {n_det}", (150, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, "SIC/AI/003 | YOLOv8-CPU",
                (w - 310, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 210), 1)

    # Danger alert bar at bottom
    if danger:
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h - 52), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "HAZARD DETECTED — SLOW DOWN",
                    (w // 2 - 220, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 240, 60), 2)

    # Detection list
    if counts:
        y = h - 70 if not danger else h - 125
        for cls_name, cnt in list(counts.items())[:4]:
            cls_id = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
            cv2.putText(frame, f"{cls_name}: {cnt}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        CLASS_COLORS[cls_id % len(CLASS_COLORS)], 1)
            y -= 20

    return frame


# ─────────────────────────────────────────────
# SINGLE IMAGE DETECTION
# ─────────────────────────────────────────────
def detect_image(model_path, image_path, conf=0.30, save_dir="./results/detections", show=True):
    """Detect objects in a single image."""
    print(f"\n🔍 Detecting: {image_path}")
    model = YOLO(model_path)
    model.to("cpu")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Cannot read: {image_path}")
        return

    t0 = time.time()
    results = model(frame, conf=conf, iou=0.45, verbose=False, device="cpu")[0]
    latency = (time.time() - t0) * 1000

    counts = defaultdict(int)
    for box in results.boxes:
        cls_id = int(box.cls)
        draw_box(frame, box.xyxy[0], cls_id, float(box.conf))
        counts[CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)] += 1

    cv2.putText(frame, f"Detections: {len(results.boxes)} | {latency:.0f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 80), 2)
    cv2.putText(frame, "SIC/AI/003 | YOLOv8-CPU",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 210), 1)

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, Path(image_path).stem + "_detected.jpg")
    cv2.imwrite(out, frame)

    print(f"   Detections : {len(results.boxes)}")
    print(f"   Latency    : {latency:.1f} ms")
    print(f"   Saved      : {out}")
    print(f"   Classes    : {dict(counts)}")

    if show:
        cv2.imshow("SIC/AI/003 — Detection Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frame


# ─────────────────────────────────────────────
# BATCH DETECTION
# ─────────────────────────────────────────────
def detect_batch(model_path, images_dir, conf=0.30, save_dir="./results/batch"):
    """Detect objects in all images in a folder."""
    model = YOLO(model_path)
    model.to("cpu")

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files += list(Path(images_dir).glob(ext))

    print(f"\n📦 Batch detection: {len(files)} images from {images_dir}")
    os.makedirs(save_dir, exist_ok=True)

    total, latencies = 0, []
    for img_path in files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        t0 = time.time()
        results = model(frame, conf=conf, iou=0.45, verbose=False, device="cpu")[0]
        latencies.append((time.time() - t0) * 1000)
        for box in results.boxes:
            draw_box(frame, box.xyxy[0], int(box.cls), float(box.conf))
        total += len(results.boxes)
        cv2.imwrite(os.path.join(save_dir, img_path.name), frame)

    if latencies:
        print(f"   Total detections : {total}")
        print(f"   Avg latency      : {np.mean(latencies):.1f} ms/image")
        print(f"   Avg FPS          : {1000/np.mean(latencies):.1f}")
    print(f"   Results saved    : {save_dir}")


# ─────────────────────────────────────────────
# REAL-TIME WEBCAM / VIDEO (CPU OPTIMIZED)
# ─────────────────────────────────────────────
def detect_realtime(model_path, source=0, conf=0.30, iou=0.45,
                    infer_size=320, frame_skip=2,
                    save_output=False, output_path="./results/output_cpu.mp4"):
    """
    CPU-optimized real-time detection.

    CPU Optimizations applied:
    - Inference at 320×320 (faster than 640×640)
    - Process every Nth frame (frame_skip) to maintain display smoothness
    - Threaded capture to prevent buffer lag
    - Display at full resolution with last detection overlaid

    Args:
        infer_size:  Image size for inference (320 recommended for CPU)
        frame_skip:  Run inference every N frames (2=every other frame)
    """
    print(f"\n🎥 CPU Real-Time Detection — SIC/AI/003")
    print(f"   Model       : {model_path}")
    print(f"   Source      : {'Webcam' if source == 0 else source}")
    print(f"   Infer Size  : {infer_size}×{infer_size} (CPU-optimized)")
    print(f"   Frame Skip  : Every {frame_skip} frames")
    print("   Controls    : Q=Quit | S=Screenshot | P=Pause\n")

    model = YOLO(model_path)
    model.to("cpu")

    cap = ThreadedCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open: {source}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_output:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (fw, fh))
        print(f"   Saving to   : {output_path}")

    fps_tracker  = FPSTracker(window=20)
    frame_count  = 0
    last_boxes   = []          # Cache last detections between skipped frames
    last_counts  = {}
    last_danger  = False
    paused       = False
    ss_n         = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠️  Stream ended.")
                    break
                frame_count += 1

            # ── INFERENCE (every Nth frame) ──
            if frame_count % frame_skip == 0 and not paused:
                # Resize for faster CPU inference
                small = cv2.resize(frame, (infer_size, infer_size))
                results = model(small, conf=conf, iou=iou, verbose=False, device="cpu")[0]

                last_boxes  = []
                last_counts = defaultdict(int)
                last_danger = False

                # Scale boxes back to display resolution
                sx = fw / infer_size
                sy = fh / infer_size
                for box in results.boxes:
                    cls_id = int(box.cls)
                    conf_s = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    scaled = [x1*sx, y1*sy, x2*sx, y2*sy]
                    last_boxes.append((scaled, cls_id, conf_s))
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    last_counts[cls_name] += 1
                    if cls_name in DANGER_CLASSES:
                        last_danger = True

            # ── DRAW CACHED DETECTIONS ──
            for (box, cls_id, conf_s) in last_boxes:
                draw_box(frame, box, cls_id, conf_s)

            # ── HUD ──
            fps_tracker.update()
            frame = draw_hud(frame, fps_tracker.fps, len(last_boxes),
                             dict(last_counts), last_danger)

            cv2.imshow("SIC/AI/003 — YOLOv8 CPU Detection", frame)

            if writer:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 Quit.")
                break
            elif key == ord("s"):
                ss_path = f"./results/ss_{ss_n:04d}.jpg"
                os.makedirs("./results", exist_ok=True)
                cv2.imwrite(ss_path, frame)
                print(f"📸 Screenshot: {ss_path}")
                ss_n += 1
            elif key == ord("p"):
                paused = not paused
                print("⏸ Paused" if paused else "▶ Resumed")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Done — {frame_count} frames | Avg FPS: {fps_tracker.fps:.1f}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CPU Inference — SIC/AI/003")
    parser.add_argument("--model",       required=True, help="Path to .pt weights")
    parser.add_argument("--source",      default="0",   help="0=webcam, or path to video/image")
    parser.add_argument("--mode",        default="realtime",
                        choices=["realtime", "image", "batch"])
    parser.add_argument("--conf",        default=0.30,  type=float)
    parser.add_argument("--iou",         default=0.45,  type=float)
    parser.add_argument("--infer-size",  default=320,   type=int,
                        help="Inference resolution (320 recommended for CPU)")
    parser.add_argument("--frame-skip",  default=2,     type=int,
                        help="Run inference every N frames (higher=faster display, lower accuracy)")
    parser.add_argument("--save",        action="store_true", help="Save output video")
    parser.add_argument("--output",      default="./results/output_cpu.mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    if args.mode == "realtime":
        detect_realtime(args.model, src, args.conf, args.iou,
                        args.infer_size, args.frame_skip,
                        args.save, args.output)
    elif args.mode == "image":
        detect_image(args.model, str(src), args.conf)
    elif args.mode == "batch":
        detect_batch(args.model, str(src), args.conf)


if __name__ == "__main__":
    main()
