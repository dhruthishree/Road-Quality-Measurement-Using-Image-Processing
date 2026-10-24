"""
Training Script: Road Defect Detection on 100% Real Indian Road Dataset
========================================================================
Trained on 3,100 genuine Indian road images extracted from Hugging Face
(ShixuanAn/RDD2020 India subset):
  - Train: 2,500 real Indian road photos (1,250 damaged, 1,250 clean)
  - Test / Validation: 600 real Indian road photos (300 damaged, 300 clean)

Features:
  - ResNet/Residual CNN Architecture with BatchNorm, Dropout & Spatial BBox Head
  - Multi-task loss: CrossEntropy (Defect Classification) + 2.0 * SmoothL1 (Defect Localization)
  - Data Augmentation: Random Flips, Color Jitter, Brightness/Contrast, Blur
  - Realistic Accuracy, Precision, Recall, F1 Score
  - Accurate elapsed training time tracking
"""

import os
import sys
import time
import json
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model_network import RoadDefectCNN


class IndianRoadDataset(Dataset):
    """Loads genuine Indian road images and YOLO defect labels."""
    def __init__(self, folder_dir: str, target_size=(224, 224), is_train=True):
        self.image_files = sorted(glob.glob(os.path.join(folder_dir, "*.jpg")))
        self.folder_dir = folder_dir
        self.target_size = target_size
        self.is_train = is_train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            bgr = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

        # Augmentation
        if self.is_train:
            if random.random() > 0.5:
                bgr = cv2.flip(bgr, 1)
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.randint(-15, 15)
                bgr = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
            if random.random() > 0.8:
                bgr = cv2.GaussianBlur(bgr, (3, 3), 0)

        resized = cv2.resize(bgr, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        arr = rgb.astype(np.float32) / 255.0
        norm = (arr - self.mean) / self.std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()

        # Parse YOLO label
        lbl_path = os.path.splitext(img_path)[0] + ".txt"
        has_defect = 0
        bbox = [0.5, 0.5, 0.0, 0.0]

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = [l.strip().split() for l in f.readlines() if l.strip()]
            if lines:
                has_defect = 1
                try:
                    cx = float(lines[0][1])
                    cy = float(lines[0][2])
                    w = float(lines[0][3])
                    h = float(lines[0][4])
                    bbox = [cx, cy, w, h]
                except (IndexError, ValueError):
                    bbox = [0.5, 0.5, 0.3, 0.3]

        label_cls = torch.tensor(has_defect, dtype=torch.long)
        label_bbox = torch.tensor(bbox, dtype=torch.float32)
        return tensor, label_cls, label_bbox


def train_indian_road_model(
    epochs: int = 4,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    output_dir: str = "models",
    dataset_dir: str = "dataset/indian_roads_dataset"
):
    print("=" * 75)
    print("TRAINING ROAD DEFECT MODEL ON 100% REAL INDIAN ROAD DATASET")
    print("Dataset: Hugging Face ShixuanAn/RDD2020 (India Subset)")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    print(f"[*] Training Device: {device} (Multi-threaded Intel/AMD CPU execution)")

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    train_dataset = IndianRoadDataset(train_dir, is_train=True)
    test_dataset = IndianRoadDataset(test_dir, is_train=False)

    print(f"[*] Real Indian Road Training Samples:   {len(train_dataset)}")
    print(f"[*] Real Indian Road Validation Samples: {len(test_dataset)}")
    print(f"[*] Total Real Images:                   {len(train_dataset) + len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = RoadDefectCNN(num_classes=2).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    history = []
    best_val_acc = 0.0

    print("\n[*] Starting Epoch Iterations...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels_cls, labels_bbox) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_cls, pred_bbox = model(images)

            loss_c = criterion_cls(pred_cls, labels_cls)
            loss_b = criterion_bbox(pred_bbox, labels_bbox)
            total_loss = loss_c + 2.0 * loss_b

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)
            preds = torch.argmax(pred_cls, dim=1)
            correct_train += (preds == labels_cls).sum().item()
            total_train += images.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                cur_loss = running_loss / max(1, total_train)
                print(f"  -> Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Running Loss: {cur_loss:.4f}", flush=True)

        scheduler.step()
        epoch_loss = running_loss / max(1, total_train)
        epoch_acc = (correct_train / max(1, total_train)) * 100.0

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        tp, fp, fn, tn = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels_cls, labels_bbox in test_loader:
                pred_cls, pred_bbox = model(images)
                loss_c = criterion_cls(pred_cls, labels_cls)
                loss_b = criterion_bbox(pred_bbox, labels_bbox)
                v_loss = loss_c + 2.0 * loss_b

                val_loss += v_loss.item() * images.size(0)
                preds = torch.argmax(pred_cls, dim=1)
                val_correct += (preds == labels_cls).sum().item()
                val_total += images.size(0)

                for p, g in zip(preds.numpy(), labels_cls.numpy()):
                    if p == 1 and g == 1: tp += 1
                    elif p == 1 and g == 0: fp += 1
                    elif p == 0 and g == 1: fn += 1
                    else: tn += 1

        val_loss /= max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0
        precision = (tp / max(1, tp + fp)) * 100.0
        recall = (tp / max(1, tp + fn)) * 100.0
        f1 = (2 * precision * recall / max(0.001, precision + recall))

        elapsed_sec = time.time() - start_time
        print(f"=== Epoch [{epoch}/{epochs}] COMPLETED | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}% | Elapsed: {elapsed_sec:.1f}s ===", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(epoch_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2)
        })

    actual_wall_time_sec = time.time() - start_time
    total_effective_hours = round(actual_wall_time_sec / 3600.0, 2)

    # Save primary model checkpoint
    checkpoint_path = os.path.join(output_dir, "road_defect_model_real_weights.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_val_acc": val_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "history": history
    }, checkpoint_path)

    # Save training metadata
    metadata = {
        "model_name": "RoadDefectCNN-v2.0-IndianRoads",
        "dataset_type": "100% Real Indian Road Surface Dataset (No synthetic training data)",
        "source": "Hugging Face ShixuanAn/RDD2020 (India Dedicated Subset)",
        "total_real_images": len(train_dataset) + len(test_dataset),
        "real_train_images": len(train_dataset),
        "real_validation_images": len(test_dataset),
        "class_balance": "50% Potholes & Structural Cracks (D00, D10, D20, D40), 50% Clean Road Pavement",
        "training_epochs": epochs,
        "batch_size": batch_size,
        "total_training_duration_seconds": round(actual_wall_time_sec, 1),
        "total_training_duration_minutes": round(actual_wall_time_sec / 60.0, 1),
        "total_training_duration_hours": f"{total_effective_hours} hours",
        "final_train_accuracy": f"{epoch_acc:.2f}%",
        "final_validation_accuracy": f"{val_acc:.2f}%",
        "final_precision": f"{precision:.2f}%",
        "final_recall": f"{recall:.2f}%",
        "final_f1_score": f"{f1:.2f}%",
        "checkpoint_file": checkpoint_path,
        "architecture": "Deep Residual Convolutional Neural Network with Spatial BBox Regression Head",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_history": history
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[OK] Training on Indian Road Dataset Completed Successfully!")
    print(f"[+] Checkpoint:          {checkpoint_path}")
    print(f"[+] Metadata:            {metadata_path}")
    print(f"[+] Real Indian Images:  {metadata['total_real_images']}")
    print(f"[+] Validation Accuracy: {metadata['final_validation_accuracy']}")
    print(f"[+] Precision:           {metadata['final_precision']}")
    print(f"[+] Recall:              {metadata['final_recall']}")
    print(f"[+] F1 Score:            {metadata['final_f1_score']}")
    print(f"[+] Training Duration:   {metadata['total_training_duration_minutes']} min ({metadata['total_training_duration_hours']})")
    print("=" * 75)

    return checkpoint_path, metadata

if __name__ == "__main__":
    train_indian_road_model(epochs=4, batch_size=64)
