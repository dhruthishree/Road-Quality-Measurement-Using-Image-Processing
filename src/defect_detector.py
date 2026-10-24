"""
Road Surface Defect Detection & Metric Characterization Engine
Identifies potholes, cracks, and surface degradation, filtering out shadows and markings,
and computes real-world metric dimensions (m²).
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from src.config import CONFIG
from src.preprocessing import PreprocessingEngine
from src.geometry import GeometryEngine

class DefectDetector:
    """
    Detects and classifies road surface irregularities:
    - Potholes (localized depressions / cavities)
    - Alligator / fatigue cracking networks
    - Longitudinal and transverse cracks
    - Surface patches & utility trench repairs
    Calculates metric area (m²), length (m), width (m), and severity rating.
    """

    def __init__(self, geometry_engine: Optional[GeometryEngine] = None, preprocessor: Optional[PreprocessingEngine] = None):
        self.geom = geometry_engine or GeometryEngine()
        self.prep = preprocessor or PreprocessingEngine()
        self.cfg = CONFIG.detector

    def get_road_roi_mask(self, shape: Tuple[int, int], horizon_y: int) -> np.ndarray:
        """Constructs a drivable road trapezoid mask excluding sky, shoulders, and vehicle hood."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Trapezoid top begins safely below horizon
        top_y = max(horizon_y + 25, int(h * 0.48))
        bottom_y = int(h * 0.88) # Cut off vehicle hood safely
        
        # Inward margin at top and bottom to stay strictly within drivable lane
        top_w_half = int(w * 0.15)
        cx = int(w * 0.5)
        
        pts = np.array([
            [cx - top_w_half, top_y],
            [cx + top_w_half, top_y],
            [int(w * 0.86), bottom_y],
            [int(w * 0.14), bottom_y]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def detect_defects(self, frame: np.ndarray, horizon_y: Optional[int] = None) -> List[Dict]:
        """
        Runs complete defect detection on a single camera frame.
        Returns list of structured defect dictionaries with physical metrics.
        """
        h, w = frame.shape[:2]
        if horizon_y is None:
            horizon_y = self.geom.nominal_horizon_y

        # 1. Preprocess: shadow attenuation, CLAHE, lane line detection, edge-preserved gray
        enhanced_bgr, gray_denoised, markings_mask = self.prep.preprocess_pipeline(frame)
        
        # Dilate markings mask to shield painted line fringes
        kernel_mark = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated_markings = cv2.dilate(markings_mask, kernel_mark, iterations=1)
        
        # 2. Road ROI mask
        road_roi = self.get_road_roi_mask((h, w), horizon_y)
        
        # Exclude painted road markings from defect search space
        effective_roi = cv2.bitwise_and(road_roi, cv2.bitwise_not(dilated_markings))

        # 3. Multi-modal Candidate Extraction:
        # A. Local dark depressions (pothole cavities) via adaptive thresholding
        local_mean = cv2.boxFilter(gray_denoised, cv2.CV_32F, (35, 35))
        diff = local_mean - gray_denoised.astype(np.float32)
        # Cavity pixels are significantly darker than local road average (at least 25 units darker)
        _, dark_cavities = cv2.threshold(diff, 26.0, 255, cv2.THRESH_BINARY)
        dark_cavities = dark_cavities.astype(np.uint8)
        
        # B. Morphological Black-Hat for fine crack lines
        kernel_crack = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray_denoised, cv2.MORPH_BLACKHAT, kernel_crack)
        _, crack_candidates = cv2.threshold(blackhat, 24, 255, cv2.THRESH_BINARY)

        # Combine defect candidates within effective road ROI
        defect_candidates = cv2.bitwise_or(dark_cavities, crack_candidates)
        defect_candidates = cv2.bitwise_and(defect_candidates, effective_roi)

        # Morphological cleanup: remove single isolated pixel noise
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        defect_candidates = cv2.morphologyEx(defect_candidates, cv2.MORPH_OPEN, cleanup_kernel)
        
        # Bridge close crack fragments
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        defect_candidates = cv2.morphologyEx(defect_candidates, cv2.MORPH_CLOSE, close_kernel)

        # 4. Connected Component & Contour Analysis
        contours, _ = cv2.findContours(defect_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_defects = []

        for cnt in contours:
            pixel_area = cv2.contourArea(cnt)
            # Filter out microscopic noise and excessively huge background masks
            if pixel_area < 45 or pixel_area > 35000:
                continue

            # Bounding box in camera frame
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw // 2
            cy = y + bh // 2

            # Compute real-world ground-plane metric coordinates and metric area
            metric_area_m2, metric_len_m, metric_width_m = self.geom.calculate_defect_metric_area(cnt)
            ground_x_m, ground_y_m = self.geom.pixel_to_ground_metric(cx, cy)

            # Filter out microscopic noise below physical threshold
            if metric_area_m2 < self.cfg.min_pothole_area_m2 and metric_len_m < self.cfg.min_crack_length_m:
                continue

            # Geometric shape characteristics
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4.0 * math.pi * pixel_area / (perimeter * perimeter + 1e-5)
            aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))
            
            # Solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) + 1e-5
            solidity = pixel_area / hull_area

            # Local intensity statistics
            mask_single = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            mean_val = cv2.mean(gray_denoised, mask=mask_single)[0]
            
            # Dilated ring around defect to measure ambient asphalt background
            dilated = cv2.dilate(mask_single, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
            ring_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(mask_single))
            ambient_val = cv2.mean(gray_denoised, mask=ring_mask)[0] + 1e-5
            
            contrast_ratio = (ambient_val - mean_val) / ambient_val

            # Filter out low-contrast normal asphalt texture fluctuations
            if contrast_ratio < 0.15 and pixel_area < 1000:
                continue

            # Defect Classification Heuristic
            defect_type = "unknown"
            confidence = 0.70

            if aspect_ratio >= self.cfg.min_defect_aspect_ratio_crack or (circularity < 0.25 and solidity < 0.55):
                # Slender or dendritic geometry -> Crack
                if solidity < 0.40 and pixel_area > 120:
                    defect_type = "crack_alligator"
                    confidence = 0.88
                elif bw > bh * 1.5:
                    defect_type = "crack_transverse"
                    confidence = 0.85
                else:
                    defect_type = "crack_longitudinal"
                    confidence = 0.87
            elif contrast_ratio >= 0.24 and circularity >= 0.20:
                # Dark localized depression with high contrast -> Pothole
                defect_type = "pothole"
                confidence = min(0.98, 0.75 + contrast_ratio * 0.5)
            elif 0.12 <= contrast_ratio < 0.22 and solidity > 0.65 and 0.25 <= metric_area_m2 <= 3.5:
                # Moderate contrast with solid regular boundary -> Asphalt patch
                defect_type = "patch"
                confidence = 0.82
            elif metric_area_m2 >= 0.08 and contrast_ratio >= 0.20:
                defect_type = "pothole"
                confidence = 0.78
            elif aspect_ratio > 1.8 and metric_len_m >= 0.35 and metric_area_m2 >= 0.035:
                defect_type = "crack_longitudinal"
                confidence = 0.75
            else:
                # Ambient texture speckle, discard
                continue

            # Physical metric sanity checks
            if defect_type == "pothole" and metric_area_m2 < 0.08:
                continue
            if "crack" in defect_type and (metric_len_m < 0.35 or metric_area_m2 < 0.030):
                continue
            if defect_type == "patch" and metric_area_m2 < 0.20:
                continue

            # Compute Severity Rating (1 to 5)
            # Pothole: based on area and depth contrast
            # Crack: based on length and opening width
            if defect_type == "pothole":
                if metric_area_m2 > 0.50 or contrast_ratio > 0.40:
                    severity = 5 # Critical / Severe
                elif metric_area_m2 > 0.25 or contrast_ratio > 0.28:
                    severity = 4 # High
                elif metric_area_m2 > 0.10:
                    severity = 3 # Moderate
                else:
                    severity = 2 # Low
            elif "crack" in defect_type:
                if metric_len_m > 1.2 or defect_type == "crack_alligator":
                    severity = 4
                elif metric_len_m > 0.6:
                    severity = 3
                else:
                    severity = 2
            else: # patch
                severity = 1 if metric_area_m2 < 0.5 else 2

            detected_defects.append({
                "type": defect_type,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "center_px": [int(cx), int(cy)],
                "pixel_area": int(pixel_area),
                "metric_area_m2": round(metric_area_m2, 3),
                "metric_length_m": round(metric_len_m, 2),
                "metric_width_m": round(metric_width_m, 2),
                "ground_x_m": round(ground_x_m, 2),
                "ground_y_m": round(ground_y_m, 2),
                "confidence": round(float(confidence), 2),
                "severity": int(severity),
                "contrast_ratio": round(float(contrast_ratio), 2),
                "contour": cnt.tolist()
            })

        return detected_defects

    def annotate_frame(
        self,
        frame: np.ndarray,
        defects: List[Dict],
        telemetry: Optional[Dict] = None,
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Renders HUD overlays: Bounding boxes, defect tags with metric areas (m²),
        and top telemetry status banner.
        """
        vis = frame.copy()
        
        # Color mapping for defect categories
        color_map = {
            "pothole": (0, 0, 240),            # Red
            "crack_alligator": (0, 140, 255),   # Orange
            "crack_longitudinal": (0, 220, 255),# Yellow
            "crack_transverse": (30, 200, 240), # Gold
            "patch": (220, 150, 0)              # Blue
        }

        # Draw detected defects
        for d in defects:
            x, y, bw, bh = d["bbox"]
            dtype = d["type"]
            color = color_map.get(dtype, (0, 255, 0))
            
            # Bounding box
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
            
            # Draw contour if requested
            if draw_contours and "contour" in d:
                cnt = np.array(d["contour"], dtype=np.int32)
                cv2.drawContours(vis, [cnt], -1, color, 1)

            # Informative label tag: Type, Metric Area (m²), Severity
            label = f"{dtype.upper()[:4]} | {d['metric_area_m2']:.2f}m² | S:{d['severity']}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # Tag background badge
            cv2.rectangle(vis, (x, max(15, y - lh - 6)), (x + lw + 6, max(15, y)), color, -1)
            cv2.putText(vis, label, (x + 3, max(15, y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return vis
