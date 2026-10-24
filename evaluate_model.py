"""
Model Evaluation & Inference Script
Run the trained RoadDefectCNN model on test images, display predictions,
compute classification metrics, and generate a visual evaluation report.

Usage:
    python evaluate_model.py                    # Evaluate on test set + show sample predictions
    python evaluate_model.py --image path.jpg   # Run inference on a single image
    python evaluate_model.py --folder path/     # Run inference on all images in a folder
"""

import os
import sys
import cv2
import json
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from src.model_network import RoadDefectCNN


CLASS_NAMES = ["No Defect (Clean Road)", "Defect Detected (Pothole/Crack)"]
SEVERITY_MAP = {
    (0.0, 0.02): "Minor Surface Wear",
    (0.02, 0.06): "Moderate Distress",
    (0.06, 0.15): "Significant Damage",
    (0.15, 1.0): "Severe Structural Failure"
}

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_trained_model(checkpoint_path: str = "models/road_defect_model_real_weights.pt"):
    """Load the trained RoadDefectCNN from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[!] Please run training first: python train_model.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadDefectCNN(num_classes=2).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[OK] Loaded trained model from: {checkpoint_path}")
    print(f"[+] Training epochs completed: {checkpoint.get('epoch', 'N/A')}")
    print(f"[+] Final validation accuracy: {checkpoint.get('final_val_acc', 'N/A'):.1f}%")
    print(f"[+] Device: {device}")

    return model, device, checkpoint


def preprocess_image(img_path: str, target_size=(224, 224)):
    """Load and preprocess a single image for inference."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    original = bgr.copy()
    resized = cv2.resize(bgr, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    arr = rgb.astype(np.float32) / 255.0
    norm = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)

    return tensor, original, resized


def get_severity_label(bbox_area: float) -> str:
    """Map predicted bounding box area to severity description."""
    for (lo, hi), label in SEVERITY_MAP.items():
        if lo <= bbox_area < hi:
            return label
    return "Unknown"


def predict_single(model, device, img_path: str) -> dict:
    """Run inference on a single image and return prediction results."""
    tensor, original, resized = preprocess_image(img_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        cls_logits, bbox_pred = model(tensor)
        probs = F.softmax(cls_logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        bbox = bbox_pred[0].cpu().numpy()

    bbox_area = bbox[2] * bbox[3]  # w * h normalized
    severity = get_severity_label(bbox_area) if pred_class == 1 else "N/A"

    return {
        "file": os.path.basename(img_path),
        "path": img_path,
        "predicted_class": pred_class,
        "class_name": CLASS_NAMES[pred_class],
        "confidence": round(confidence * 100, 2),
        "bbox_normalized": [round(float(b), 4) for b in bbox],
        "bbox_area": round(float(bbox_area), 4),
        "severity": severity,
        "original_image": original,
        "resized_image": resized
    }


def draw_prediction_overlay(image: np.ndarray, result: dict) -> np.ndarray:
    """Draw prediction results directly on the image with styled overlay."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Draw bounding box if defect detected
    if result["predicted_class"] == 1:
        cx, cy, bw, bh = result["bbox_normalized"]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Red bounding box with thick border
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Corner brackets for style
        bracket_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        for (px, py), (dx, dy) in [
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))
        ]:
            cv2.line(vis, (px, py), (px + dx * bracket_len, py), (0, 100, 255), 3)
            cv2.line(vis, (px, py), (px, py + dy * bracket_len), (0, 100, 255), 3)

    # Top banner
    banner_h = 50
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

    # Status text
    if result["predicted_class"] == 1:
        status_text = f"DEFECT DETECTED | {result['confidence']:.1f}% | {result['severity']}"
        status_color = (68, 68, 239)  # Red BGR
    else:
        status_text = f"ROAD CLEAR | {result['confidence']:.1f}% Confidence"
        status_color = (129, 185, 16)  # Green BGR

    cv2.putText(vis, status_text, (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    # File name at bottom
    cv2.putText(vis, result["file"], (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return vis


def evaluate_test_set(model, device, test_dir: str = "dataset/real_road_dataset/dataset/test"):
    """Evaluate model on the full test set and compute metrics."""
    image_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    if not image_files:
        print(f"[!] No test images found in {test_dir}")
        return None

    print(f"\n{'='*75}")
    print(f"EVALUATING ON {len(image_files)} REAL ROAD TEST IMAGES")
    print(f"{'='*75}\n")

    results = []
    correct = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    start_time = time.time()

    for i, img_path in enumerate(image_files):
        result = predict_single(model, device, img_path)
        results.append(result)

        # Ground truth from label file
        lbl_path = os.path.splitext(img_path)[0] + ".txt"
        gt_has_defect = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    gt_has_defect = 1

        pred = result["predicted_class"]
        if pred == gt_has_defect:
            correct += 1
        if pred == 1 and gt_has_defect == 1:
            tp += 1
        elif pred == 1 and gt_has_defect == 0:
            fp += 1
        elif pred == 0 and gt_has_defect == 1:
            fn += 1
        else:
            tn += 1

        status = "✓" if pred == gt_has_defect else "✗"
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  Processed {i+1}/{len(image_files)} images... "
                  f"Running Accuracy: {correct/(i+1)*100:.1f}%")

    elapsed = time.time() - start_time
    accuracy = correct / len(image_files) * 100
    precision = tp / max(1, tp + fp) * 100
    recall = tp / max(1, tp + fn) * 100
    f1 = 2 * precision * recall / max(1, precision + recall)
    fps_throughput = len(image_files) / elapsed

    metrics = {
        "total_images": len(image_files),
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "inference_time_seconds": round(elapsed, 2),
        "throughput_fps": round(fps_throughput, 1)
    }

    print(f"\n{'='*75}")
    print(f"EVALUATION RESULTS ON 100% REAL ROAD TEST DATASET")
    print(f"{'='*75}")
    print(f"  Total Test Images:    {metrics['total_images']}")
    print(f"  Accuracy:             {metrics['accuracy']:.2f}%")
    print(f"  Precision:            {metrics['precision']:.2f}%")
    print(f"  Recall (Sensitivity): {metrics['recall']:.2f}%")
    print(f"  F1 Score:             {metrics['f1_score']:.2f}%")
    print(f"  True Positives:       {tp}")
    print(f"  False Positives:      {fp}")
    print(f"  True Negatives:       {tn}")
    print(f"  False Negatives:      {fn}")
    print(f"  Inference Time:       {elapsed:.2f}s ({fps_throughput:.1f} FPS)")
    print(f"{'='*75}")

    return metrics, results


def generate_visual_grid(results: list, output_path: str = "output/evaluation_grid.jpg",
                         grid_cols: int = 4, max_images: int = 16):
    """Create a visual grid of predictions for review."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sample = results[:max_images]
    cell_size = 280
    grid_cols = min(grid_cols, len(sample))
    grid_rows = (len(sample) + grid_cols - 1) // grid_cols

    canvas = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 30)

    for i, result in enumerate(sample):
        row, col = divmod(i, grid_cols)
        img = cv2.resize(result["original_image"], (cell_size, cell_size))
        annotated = draw_prediction_overlay(img, result)
        y1 = row * cell_size
        x1 = col * cell_size
        canvas[y1:y1 + cell_size, x1:x1 + cell_size] = annotated

    cv2.imwrite(output_path, canvas)
    print(f"[+] Visual evaluation grid saved: {output_path}")
    return output_path


def save_evaluation_report(metrics: dict, output_path: str = "output/evaluation_report.json"):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load training metadata if available
    training_meta = {}
    meta_path = "models/training_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            training_meta = json.load(f)

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": training_meta.get("model_name", "RoadDefectCNN-v1.0"),
        "dataset_type": training_meta.get("dataset_type", "100% Real Road Dataset"),
        "dataset_source": training_meta.get("source", "Brazilian NDTI / Global RDD"),
        "total_real_images_trained_on": training_meta.get("total_real_images", "N/A"),
        "training_duration": training_meta.get("total_training_duration_hours", "N/A"),
        "training_epochs": training_meta.get("training_epochs", "N/A"),
        "evaluation_metrics": metrics
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[+] Evaluation report saved: {output_path}")
    return report


def run_single_image_inference(img_path: str, model, device):
    """Interactive inference on a single image with visual output."""
    print(f"\n[*] Running inference on: {img_path}")

    result = predict_single(model, device, img_path)

    print(f"\n{'='*60}")
    print(f"  PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"  File:         {result['file']}")
    print(f"  Class:        {result['class_name']}")
    print(f"  Confidence:   {result['confidence']:.1f}%")
    print(f"  Severity:     {result['severity']}")
    print(f"  BBox (YOLO):  cx={result['bbox_normalized'][0]:.3f}, "
          f"cy={result['bbox_normalized'][1]:.3f}, "
          f"w={result['bbox_normalized'][2]:.3f}, "
          f"h={result['bbox_normalized'][3]:.3f}")
    print(f"{'='*60}")

    # Save annotated result
    annotated = draw_prediction_overlay(result["original_image"], result)
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"prediction_{result['file']}")
    cv2.imwrite(out_path, annotated)
    print(f"[+] Annotated prediction saved: {out_path}")

    return result


def run_folder_inference(folder_path: str, model, device):
    """Run inference on all images in a folder."""
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        print(f"[!] No images found in: {folder_path}")
        return

    print(f"\n[*] Processing {len(image_files)} images from: {folder_path}")
    results = []
    defect_count = 0
    clean_count = 0

    for img_path in sorted(image_files):
        result = predict_single(model, device, img_path)
        results.append(result)
        if result["predicted_class"] == 1:
            defect_count += 1
        else:
            clean_count += 1

    print(f"\n{'='*60}")
    print(f"  FOLDER INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Images:     {len(results)}")
    print(f"  Defects Found:    {defect_count}")
    print(f"  Clean Roads:      {clean_count}")
    print(f"  Defect Rate:      {defect_count/len(results)*100:.1f}%")
    print(f"{'='*60}")

    # Generate visual grid
    grid_path = generate_visual_grid(results, "output/folder_inference_grid.jpg")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Road Defect Detection Model"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="models/road_defect_model_real_weights.pt",
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--image", type=str,
                        help="Path to a single image for inference")
    parser.add_argument("--folder", type=str,
                        help="Path to a folder of images for batch inference")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip generating visual prediction grid")

    args = parser.parse_args()

    # Load trained model
    model, device, checkpoint = load_trained_model(args.checkpoint)

    if args.image:
        # Single image inference
        run_single_image_inference(args.image, model, device)

    elif args.folder:
        # Folder batch inference
        run_folder_inference(args.folder, model, device)

    else:
        # Full test set evaluation (default)
        metrics, results = evaluate_test_set(model, device)

        if metrics:
            # Save evaluation report
            save_evaluation_report(metrics)

            # Generate visual grid
            if not args.no_grid:
                generate_visual_grid(results, "output/evaluation_grid.jpg",
                                     grid_cols=4, max_images=16)

            print(f"\n[OK] Full evaluation complete!")
            print(f"[+] Run 'python evaluate_model.py --image path/to/photo.jpg' "
                  f"to test on your own road images.")


if __name__ == "__main__":
    main()
