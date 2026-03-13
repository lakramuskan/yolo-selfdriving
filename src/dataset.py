"""
============================================================
Dataset Preprocessing & Preparation Pipeline
Project: Object Detection for Self-Driving using YOLO
Project Code: SIC/AI/003
Team: Naman Kumar, Aryan Kaushik, Muskan, Adhisha Pahuja, Sukesh Bhardwaj
============================================================

This script handles:
1. Downloading datasets (COCO, BDD100K, custom)
2. Converting annotation formats to YOLO format
3. Splitting data into train/val/test sets
4. Applying data augmentation
5. Visualizing dataset statistics
"""

import os
import sys
import cv2
import yaml
import json
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "data_root": "./data",
    "image_size": 640,
    "train_ratio": 0.70,
    "val_ratio": 0.20,
    "test_ratio": 0.10,
    "random_seed": 42,
    "min_bbox_area": 400,      # Minimum bounding box area (pixels²) to keep
    "augment_factor": 2,       # How many augmented copies per image
}

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "autorickshaw",
    "bus", "truck", "traffic_light", "stop_sign",
    "speed_breaker", "pothole", "animal", "construction"
]

# COCO to our class mapping (COCO class_id -> our class_id)
COCO_TO_OURS = {
    0: 0,   # person -> person
    1: 1,   # bicycle -> bicycle
    2: 2,   # car -> car
    3: 3,   # motorcycle -> motorcycle
    5: 5,   # bus -> bus
    7: 6,   # truck -> truck
    9: 7,   # traffic light -> traffic_light
    11: 8,  # stop sign -> stop_sign
}


# ─────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────
def setup_directories(data_root: str):
    """Create the required YOLO directory structure."""
    dirs = [
        f"{data_root}/images/train",
        f"{data_root}/images/val",
        f"{data_root}/images/test",
        f"{data_root}/labels/train",
        f"{data_root}/labels/val",
        f"{data_root}/labels/test",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Directory structure created at: {data_root}")


# ─────────────────────────────────────────────
# FORMAT CONVERTERS
# ─────────────────────────────────────────────
def coco_bbox_to_yolo(bbox, img_width: int, img_height: int):
    """
    Convert COCO bounding box [x_min, y_min, width, height]
    to YOLO format [x_center, y_center, width, height] (normalized 0-1).
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height

    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_w   = max(0, min(1, norm_w))
    norm_h   = max(0, min(1, norm_h))

    return x_center, y_center, norm_w, norm_h


def yolo_to_bbox(yolo_label, img_width: int, img_height: int):
    """
    Convert YOLO format back to pixel bounding box [x_min, y_min, x_max, y_max].
    Used for visualization.
    """
    cls, xc, yc, w, h = yolo_label
    x_min = int((xc - w / 2) * img_width)
    y_min = int((yc - h / 2) * img_height)
    x_max = int((xc + w / 2) * img_width)
    y_max = int((yc + h / 2) * img_height)
    return int(cls), x_min, y_min, x_max, y_max


def convert_coco_annotations(coco_json_path: str, images_dir: str, output_labels_dir: str):
    """
    Convert a COCO format JSON annotation file to YOLO .txt label files.
    Each .txt file corresponds to one image.
    """
    print(f"\n📂 Converting COCO annotations: {coco_json_path}")
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Build lookup maps
    img_map = {img["id"]: img for img in coco["images"]}
    ann_map = {}  # image_id -> list of annotations
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        ann_map.setdefault(img_id, []).append(ann)

    skipped, converted = 0, 0
    for img_id, img_info in tqdm(img_map.items(), desc="Converting"):
        img_w = img_info["width"]
        img_h = img_info["height"]
        filename = Path(img_info["file_name"]).stem

        annotations = ann_map.get(img_id, [])
        yolo_lines = []

        for ann in annotations:
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in COCO_TO_OURS:
                continue  # Skip unmapped classes

            our_class_id = COCO_TO_OURS[coco_cat_id]
            bbox = ann["bbox"]

            # Filter tiny boxes
            if bbox[2] * bbox[3] < CONFIG["min_bbox_area"]:
                skipped += 1
                continue

            xc, yc, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
            yolo_lines.append(f"{our_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            label_path = os.path.join(output_labels_dir, f"{filename}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
            converted += 1

    print(f"   ✅ Converted: {converted} images | Skipped tiny boxes: {skipped}")


# ─────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
def split_dataset(images_dir: str, labels_dir: str, output_root: str,
                  train_r=0.70, val_r=0.20, seed=42):
    """
    Split image-label pairs into train/val/test sets.
    Maintains pairing between images and labels.
    """
    print(f"\n✂️  Splitting dataset: {train_r*100:.0f}% train / {val_r*100:.0f}% val / "
          f"{(1-train_r-val_r)*100:.0f}% test")

    # Gather all images that have matching labels
    image_files = sorted([
        f for f in Path(images_dir).glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])
    valid_pairs = []
    for img_path in image_files:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            valid_pairs.append((img_path, label_path))

    print(f"   Found {len(valid_pairs)} valid image-label pairs")
    if len(valid_pairs) == 0:
        print("   ⚠️  No valid pairs found. Make sure images and labels directories are correct.")
        return

    random.seed(seed)
    random.shuffle(valid_pairs)

    n = len(valid_pairs)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    splits = {
        "train": valid_pairs[:n_train],
        "val":   valid_pairs[n_train:n_train + n_val],
        "test":  valid_pairs[n_train + n_val:],
    }

    for split_name, pairs in splits.items():
        img_out_dir = Path(output_root) / "images" / split_name
        lbl_out_dir = Path(output_root) / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in tqdm(pairs, desc=f"Copying {split_name}"):
            shutil.copy2(img_path, img_out_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_out_dir / lbl_path.name)

        print(f"   {split_name:6s}: {len(pairs):5d} samples → {img_out_dir}")


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────
def get_augmentation_pipeline():
    """
    Define augmentation pipeline using Albumentations.
    Optimized for driving scene robustness.
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5),

        # Weather & lighting simulation
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.RandomGamma(p=0.3),

        # Blur & noise (simulating camera artifacts)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),

        # Occlusion simulation (cutout/dropout)
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32,
                        min_holes=1, fill_value=0, p=0.2),

        # Weather effects
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.RandomRain(blur_value=2, p=0.1),

    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_area=0.001,
        min_visibility=0.3,
    ))


def augment_dataset(data_root: str, factor: int = 2):
    """
    Apply augmentation to training set.
    Creates `factor` additional augmented copies per image.
    """
    train_img_dir = Path(data_root) / "images" / "train"
    train_lbl_dir = Path(data_root) / "labels" / "train"

    image_files = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    print(f"\n🔄 Augmenting {len(image_files)} training images (factor={factor})...")

    transform = get_augmentation_pipeline()

    for img_path in tqdm(image_files, desc="Augmenting"):
        label_path = train_lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Read labels
        bboxes, class_labels = [], []
        with open(label_path) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    # Clamp and validate
                    bbox = [max(0.001, min(0.999, v)) for v in bbox]
                    bboxes.append(bbox)
                    class_labels.append(cls)

        if not bboxes:
            continue

        # Apply augmentation `factor` times
        for i in range(factor):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                if not aug_bboxes:
                    continue

                # Save augmented image
                aug_name = f"{img_path.stem}_aug{i+1}"
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(train_img_dir / f"{aug_name}.jpg"), aug_img_bgr)

                # Save augmented labels
                with open(train_lbl_dir / f"{aug_name}.txt", "w") as f:
                    for cls, bbox in zip(aug_labels, aug_bboxes):
                        xc, yc, bw, bh = bbox
                        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            except Exception as e:
                continue  # Skip failed augmentations

    new_count = len(list(train_img_dir.glob("*")))
    print(f"   ✅ Training set expanded to {new_count} images")


# ─────────────────────────────────────────────
# DATASET VISUALIZATION & STATISTICS
# ─────────────────────────────────────────────
def visualize_sample(data_root: str, split: str = "train", num_samples: int = 6,
                     save_path: str = "./results/sample_grid.jpg"):
    """
    Display a grid of sample images with bounding box annotations.
    """
    img_dir = Path(data_root) / "images" / split
    lbl_dir = Path(data_root) / "labels" / split

    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not image_files:
        print("⚠️  No images found for visualization.")
        return

    random.shuffle(image_files)
    samples = image_files[:num_samples]

    # Color palette
    np.random.seed(42)
    colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in CLASS_NAMES]

    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for idx, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Draw bounding boxes
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        color = colors[cls % len(colors)]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
                        cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        axes[idx].imshow(img)
        axes[idx].set_title(img_path.name, fontsize=9)
        axes[idx].axis("off")

    # Hide unused subplots
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Sample Annotations — {split.upper()} set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Sample grid saved: {save_path}")


def plot_class_distribution(data_root: str, save_path: str = "./results/class_distribution.png"):
    """
    Plot bar chart of class distribution across train/val/test splits.
    """
    print("\n📊 Computing class distribution...")
    split_counts = {}

    for split in ["train", "val", "test"]:
        lbl_dir = Path(data_root) / "labels" / split
        counts = {cls: 0 for cls in range(len(CLASS_NAMES))}

        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        if cls in counts:
                            counts[cls] += 1
        split_counts[split] = counts

    # Plot
    x = np.arange(len(CLASS_NAMES))
    width = 0.28
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#2E75B6", "#70AD47", "#ED7D31"]

    for i, (split, counts) in enumerate(split_counts.items()):
        vals = [counts[c] for c in range(len(CLASS_NAMES))]
        ax.bar(x + i * width, vals, width, label=split.capitalize(), color=colors[i], alpha=0.85)

    ax.set_xlabel("Object Class", fontsize=12)
    ax.set_ylabel("Instance Count", fontsize=12)
    ax.set_title("Class Distribution — Train / Val / Test", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Class distribution saved: {save_path}")


def dataset_statistics(data_root: str):
    """Print a summary table of dataset statistics."""
    print("\n" + "=" * 55)
    print("  DATASET STATISTICS — SIC/AI/003")
    print("=" * 55)

    total_images, total_labels = 0, 0
    for split in ["train", "val", "test"]:
        img_dir  = Path(data_root) / "images" / split
        lbl_dir  = Path(data_root) / "labels" / split
        n_images = len(list(img_dir.glob("*")))
        n_labels = len(list(lbl_dir.glob("*.txt")))
        total_images += n_images
        total_labels += n_labels
        print(f"  {split.upper():8s} | Images: {n_images:6d} | Labels: {n_labels:6d}")

    print(f"  {'TOTAL':8s} | Images: {total_images:6d} | Labels: {total_labels:6d}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SIC/AI/003 — Dataset Preprocessing Pipeline")
    print("=" * 55)

    data_root = CONFIG["data_root"]

    # Step 1: Create directory structure
    setup_directories(data_root)

    # Step 2: If you have a COCO JSON, convert it:
    # convert_coco_annotations(
    #     coco_json_path="./raw/annotations/instances_train2017.json",
    #     images_dir="./raw/images/train2017",
    #     output_labels_dir="./raw/labels_yolo"
    # )

    # Step 3: Split dataset
    # split_dataset(
    #     images_dir="./raw/images/all",
    #     labels_dir="./raw/labels_yolo",
    #     output_root=data_root,
    #     train_r=CONFIG["train_ratio"],
    #     val_r=CONFIG["val_ratio"],
    # )

    # Step 4: Augment training data
    # augment_dataset(data_root, factor=CONFIG["augment_factor"])

    # Step 5: Statistics & visualization
    dataset_statistics(data_root)
    # visualize_sample(data_root, split="train", save_path="./results/samples.jpg")
    # plot_class_distribution(data_root, save_path="./results/class_dist.png")

    print("\n✅ Dataset preprocessing complete!")
    print("   Next step: Run  python src/train.py\n")


if __name__ == "__main__":
    main()
