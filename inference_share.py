"""
Self-Contained Inference Script for Sharing & Deployment
=========================================================
Anyone with PyTorch can run this script to test the model on any road image.
Works with either the PyTorch state_dict (.pt) or the standalone TorchScript model.

Usage:
    python inference_share.py --image "path/to/road_photo.jpg"
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from src.model_network import RoadDefectResNet18
except ImportError:
    # Fallback to loading TorchScript if source code is not present
    RoadDefectResNet18 = None

CLASS_NAMES = ["Clean Road (No Defect)", "Pothole / Surface Distress Detected"]


def predict(image_path, model_path="models/road_defect_model_real_weights.pt"):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig = bgr.copy()
    h, w = bgr.shape[:2]

    # Preprocessing: Road ROI & CLAHE
    if h > 200:
        road_roi = bgr[int(0.35 * h):int(0.95 * h), :]
    else:
        road_roi = bgr

    lab = cv2.cvtColor(road_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge((clahe.apply(l), a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    resized = cv2.resize(enhanced, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = ((rgb.astype(np.float32) / 255.0) - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)

    # Load Model (TorchScript or PyTorch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path.endswith("_torchscript.pt") or RoadDefectResNet18 is None:
        model = torch.jit.load(model_path, map_location=device)
    else:
        model = RoadDefectResNet18(num_classes=2).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    model.eval()
    with torch.no_grad():
        cls_logits, bbox_pred = model(tensor.to(device))
        probs = F.softmax(cls_logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item() * 100.0
        bbox = bbox_pred[0].cpu().numpy()

    print("\n" + "=" * 60)
    print(f"IMAGE:      {os.path.basename(image_path)}")
    print(f"PREDICTION: {CLASS_NAMES[pred_class]}")
    print(f"CONFIDENCE: {confidence:.2f}%")
    if pred_class == 1:
        print(f"BBOX (Norm): cx={bbox[0]:.3f}, cy={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f}")
    print("=" * 60)

    # Save output overlay
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"shared_prediction_{os.path.basename(image_path)}")

    color = (0, 0, 255) if pred_class == 1 else (0, 255, 0)
    label_txt = f"{CLASS_NAMES[pred_class]} ({confidence:.1f}%)"
    cv2.putText(orig, label_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_file, orig)
    print(f"[+] Annotated visual saved: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input road image")
    parser.add_argument("--model", type=str, default="models/road_defect_model_real_weights.pt")
    args = parser.parse_args()
    predict(args.image, args.model)
