"""
Illumination-Invariant Preprocessing & Shadow Attenuation Engine
Solves Gap 2: Disentangles shadows, lighting artifacts, and paint markings from genuine surface defects.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from src.config import CONFIG

class PreprocessingEngine:
    """
    Advanced multi-stage image preprocessor:
    1. Multi-scale Retinex & CLAHE luminance compensation in CIELAB color space.
    2. Shadow segmentation and luminance restoration.
    3. Paint marking (lane divider) color segmentation & suppression.
    4. Edge-preserving bilateral filtering to suppress asphalt aggregate noise.
    """

    def __init__(self, clip_limit: float = 2.5, grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.bilateral_sigma_color = CONFIG.detector.bilateral_sigma_color
        self.bilateral_sigma_space = CONFIG.detector.bilateral_sigma_space

    def enhance_luminance_clahe(self, bgr_image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L-channel in CIELAB space to equalize lighting while preserving chrominance."""
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def multi_scale_retinex(self, img_channel: np.ndarray, scales: list = [15, 80, 250]) -> np.ndarray:
        """Compute Single/Multi-Scale Retinex for dynamic range compression."""
        img_float = np.maximum(img_channel.astype(np.float32), 1.0)
        log_img = np.log10(img_float)
        retinex = np.zeros_like(img_float)
        
        for sigma in scales:
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
            blur = np.maximum(blur, 1.0)
            retinex += (log_img - np.log10(blur))
            
        retinex /= len(scales)
        # Normalize to 0-255
        mean = np.mean(retinex)
        std = np.std(retinex) + 1e-5
        norm = 128 + 128 * (retinex - mean) / (3.0 * std)
        return np.clip(norm, 0, 255).astype(np.uint8)

    def detect_and_attenuate_shadows(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects ambient shadows (tree foliage, bridges, vehicles)
        using HSV/LAB ratio analysis and boosts shadow luminance.
        Returns:
            attenuated_bgr: Image with shadows balanced.
            shadow_mask: Binary mask of detected shadows.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Shadows have low V (brightness) and relatively high S / (V + 1) ratio compared to asphalt
        ratio = (s.astype(np.float32) + 1.0) / (v.astype(np.float32) + 1.0)
        ratio_norm = cv2.normalize(ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold shadow candidates: low V and high ratio
        _, shadow_v_thresh = cv2.threshold(v, 75, 255, cv2.THRESH_BINARY_INV)
        _, shadow_r_thresh = cv2.threshold(ratio_norm, 110, 255, cv2.THRESH_BINARY)
        raw_shadow = cv2.bitwise_and(shadow_v_thresh, shadow_r_thresh)
        
        # Morphological opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_mask = cv2.morphologyEx(raw_shadow, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # Attenuate shadows by boosting luminance in shadow regions
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        
        # Smooth mask for gradual boundary blending
        soft_mask = cv2.GaussianBlur(shadow_mask.astype(np.float32) / 255.0, (15, 15), 5)
        
        # Boost L channel in shadows
        l_boosted = np.clip(l.astype(np.float32) * 1.55 + 25.0, 0, 255).astype(np.uint8)
        l_final = ((1.0 - soft_mask) * l + soft_mask * l_boosted).astype(np.uint8)
        
        attenuated_lab = cv2.merge([l_final, a, b_ch])
        attenuated_bgr = cv2.cvtColor(attenuated_lab, cv2.COLOR_LAB2BGR)
        return attenuated_bgr, shadow_mask

    def detect_paint_markings(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Detects road painted markings (white dashed lane lines and yellow shoulder stripes).
        Returns a binary mask used to suppress painted lines from defect detection.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # White lane lines (low saturation, high value)
        lower_white = np.array([0, 0, 185], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Yellow shoulder stripes (hue ~18-35, medium-high saturation, high value)
        lower_yellow = np.array([15, 80, 140], dtype=np.uint8)
        upper_yellow = np.array([36, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        markings_mask = cv2.bitwise_or(mask_white, mask_yellow)
        # Dilate slightly to cover line fringes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.dilate(markings_mask, kernel, iterations=1)

    def preprocess_pipeline(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete end-to-end preprocessing pipeline.
        Returns:
            enhanced_bgr: Normalized, shadow-attenuated image.
            gray_denoised: Bilateral edge-preserved grayscale image.
            markings_mask: Mask of lane markings for false-positive suppression.
        """
        # 1. Shadow attenuation
        shadow_compensated, shadow_mask = self.detect_and_attenuate_shadows(frame)
        
        # 2. CLAHE equalization
        enhanced_bgr = self.enhance_luminance_clahe(shadow_compensated)
        
        # 3. Paint marking mask
        markings_mask = self.detect_paint_markings(frame)
        
        # 4. Bilateral filtering for noise smoothing with edge preservation
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        gray_denoised = cv2.bilateralFilter(gray, 7, self.bilateral_sigma_color, self.bilateral_sigma_space)
        
        return enhanced_bgr, gray_denoised, markings_mask
