#!/bin/bash
# ============================================================
# Quick Start Script — SIC/AI/003
# Object Detection for Self-Driving using YOLO
# ============================================================

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   SIC/AI/003 — YOLOv8 Self-Driving Detection        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Install dependencies ──
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q
echo "✅ Dependencies installed"
echo ""

# ── Step 2: Prepare dataset structure ──
echo "📂 Setting up dataset directories..."
python src/dataset.py
echo ""

# ── Step 3: Train model ──
echo "🚀 Starting training..."
echo "   Model: YOLOv8-Medium | Epochs: 100 | Batch: 16"
python src/train.py \
    --data   configs/dataset.yaml \
    --model  medium \
    --epochs 100 \
    --batch  16 \
    --imgsz  640 \
    --name   yolo_selfdriving
echo ""

# ── Step 4: Evaluate ──
echo "📊 Running evaluation..."
python src/evaluate.py \
    --model  runs/train/yolo_selfdriving/weights/best.pt \
    --data   configs/dataset.yaml \
    --split  val \
    --benchmark \
    --report
echo ""

# ── Step 5: Demo inference ──
echo "🎥 Launch detection demo:"
echo "   python src/detect.py --model runs/train/yolo_selfdriving/weights/best.pt --source 0"
echo ""
echo "✅ All done! Results saved in ./runs/ and ./results/"
