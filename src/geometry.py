"""
Perspective Geometry, Homography, Inverse Perspective Mapping (IPM) & Road Inclination Engine
Solves Gap 3: Eliminates perspective distortion, estimates real-world metric defect area (m²),
and dynamically calculates road inclination/pitch grade (%).
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional, Dict
from src.config import CameraCalibration, CONFIG

class GeometryEngine:
    """
    Computes:
    1. Camera perspective projection & Inverse Perspective Mapping (IPM) homography.
    2. Dynamic Vanishing Point & Horizon tracking to determine pitch / road inclination (%).
    3. Metric coordinate mapping (pixels -> real-world meters ahead of bumper).
    4. Metric surface area (m²) and physical dimensions for identified defects.
    """

    def __init__(self, calib: Optional[CameraCalibration] = None):
        self.calib = calib or CONFIG.camera
        self.w = self.calib.frame_width
        self.h = self.calib.frame_height
        
        # Focal lengths in pixels derived from Field of View
        self.fx = (self.w / 2.0) / math.tan(math.radians(self.calib.fov_horizontal_deg / 2.0))
        self.fy = (self.h / 2.0) / math.tan(math.radians(self.calib.fov_vertical_deg / 2.0))
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

        # Nominal horizon y position (for downward tilted dashcam, horizon appears above optical center in world or at default road perspective)
        self.nominal_horizon_y = float(self.h * 0.44)  # ~316.8px on 720p
        
        # Initialize IPM Homography matrices
        self.H_ipm, self.H_inv = self._compute_ipm_matrices()

    def _compute_ipm_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates homography matrix mapping camera image plane to bird's-eye metric ground plane.
        Ground ROI:
          X: [-ground_width_m / 2, +ground_width_m / 2] (e.g. -3.5m to +3.5m)
          Y: [ground_near_dist_m, ground_far_dist_m]     (e.g. 3.0m to 25.0m ahead)
        IPM Image:
          u_ipm: [0, ipm_width]
          v_ipm: [0, ipm_height]
        """
        c = self.calib
        # Four physical ground points (in camera coordinates: X right, Y down, Z forward)
        # Note: on road plane at Y = +c.camera_height_m
        h_cam = c.camera_height_m
        z_near = c.ground_near_dist_m
        z_far = c.ground_far_dist_m
        half_w = c.ground_width_m / 2.0

        # Camera rotation matrix for pitch angle (looking down by pitch_angle_deg)
        theta = math.radians(c.pitch_angle_deg)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)]
        ])

        # Project 4 ground points into camera pixels
        # 1: Near-Left, 2: Near-Right, 3: Far-Right, 4: Far-Left
        ground_pts_3d = [
            np.array([-half_w, h_cam, z_near]),
            np.array([+half_w, h_cam, z_near]),
            np.array([+half_w, h_cam, z_far]),
            np.array([-half_w, h_cam, z_far])
        ]

        img_pts = []
        for P in ground_pts_3d:
            # Rotate into camera frame
            P_cam = R_pitch.dot(P)
            # Perspective divide
            u = self.fx * (P_cam[0] / P_cam[2]) + self.cx
            v = self.fy * (P_cam[1] / P_cam[2]) + self.cy
            img_pts.append([u, v])

        # Destination points in top-down IPM image (Bird's Eye View)
        # IPM coordinate system: top (v=0) is far distance, bottom (v=ipm_height) is near distance
        dst_pts = np.array([
            [0, c.ipm_height],                  # Near-Left
            [c.ipm_width, c.ipm_height],        # Near-Right
            [c.ipm_width, 0],                   # Far-Right
            [0, 0]                              # Far-Left
        ], dtype=np.float32)

        src_pts = np.array(img_pts, dtype=np.float32)
        H_ipm = cv2.getPerspectiveTransform(src_pts, dst_pts)
        H_inv = np.linalg.inv(H_ipm)
        return H_ipm, H_inv

    def generate_ipm_view(self, frame: np.ndarray) -> np.ndarray:
        """Warps camera perspective frame to top-down metric Bird's Eye View (BEV)."""
        return cv2.warpPerspective(
            frame, self.H_ipm,
            (self.calib.ipm_width, self.calib.ipm_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

    def pixel_to_ground_metric(self, u: float, v: float) -> Tuple[float, float]:
        """
        Maps a 2D image pixel coordinate (u, v) to real-world ground-plane metric coordinates (X_m, Y_m)
        where X_m is lateral offset from camera centerline (meters)
        and Y_m is longitudinal distance ahead of front bumper (meters).
        """
        pt = np.array([[[u, v]]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pt, self.H_ipm)[0][0]
        u_ipm, v_ipm = warped_pt[0], warped_pt[1]

        # Convert IPM pixel coords to physical meters
        c = self.calib
        # u_ipm spans [-ground_width_m/2, +ground_width_m/2]
        x_m = ((u_ipm / float(c.ipm_width)) - 0.5) * c.ground_width_m
        # v_ipm spans [ground_far_dist_m (top v=0), ground_near_dist_m (bottom v=ipm_height)]
        y_m = c.ground_far_dist_m - (v_ipm / float(c.ipm_height)) * (c.ground_far_dist_m - c.ground_near_dist_m)
        return float(x_m), float(y_m)

    def calculate_defect_metric_area(self, contour: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates the true physical area in square meters (m²), length (m), and width (m)
        for a defect contour using Inverse Perspective Mapping.
        """
        if len(contour) < 3:
            return 0.0, 0.0, 0.0

        # Transform all contour points into IPM plane
        pts_float = contour.astype(np.float32)
        ipm_pts = cv2.perspectiveTransform(pts_float, self.H_ipm)

        # Scale factors: meters per IPM pixel
        c = self.calib
        dx_m_per_px = c.ground_width_m / float(c.ipm_width)
        dy_m_per_px = (c.ground_far_dist_m - c.ground_near_dist_m) / float(c.ipm_height)

        # Compute contour polygon area in IPM pixels
        ipm_area_px = cv2.contourArea(ipm_pts)
        metric_area_m2 = ipm_area_px * (dx_m_per_px * dy_m_per_px)

        # Fit minimum area bounding box in IPM plane to get length and width
        rect = cv2.minAreaRect(ipm_pts)
        w_px, h_px = rect[1]
        dim1_m = w_px * dx_m_per_px
        dim2_m = h_px * dy_m_per_px
        length_m = max(dim1_m, dim2_m)
        width_m = min(dim1_m, dim2_m)

        return float(metric_area_m2), float(length_m), float(width_m)

    def estimate_road_inclination(self, frame: np.ndarray, prev_pitch: Optional[float] = None) -> Dict[str, float]:
        """
        Estimates dynamic road inclination (pitch angle in degrees & grade in %)
        by tracking vanishing point horizon line and edge convergence.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Focus on upper road band (between 25% and 65% of frame height)
        roi_top = int(h * 0.25)
        roi_bottom = int(h * 0.65)
        band = gray[roi_top:roi_bottom, :]
        
        # Detect prominent structural edges (lane lines, road shoulders, curb borders)
        edges = cv2.Canny(band, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)
        
        horizon_candidates = []
        if lines is not None and len(lines) >= 2:
            # Separate lines into left-leaning and right-leaning
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Adjust y back to full frame
                y1 += roi_top
                y2 += roi_top
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) < 1e-4:
                    continue
                slope = dy / float(dx)
                # Left lane lines typically have negative slope (e.g. -0.4 to -2.0)
                if -2.5 < slope < -0.2:
                    left_lines.append((x1, y1, x2, y2))
                # Right lane lines typically have positive slope (e.g. +0.2 to +2.5)
                elif 0.2 < slope < 2.5:
                    right_lines.append((x1, y1, x2, y2))
                    
            # Compute pairwise line intersections
            for l_line in left_lines[:6]:
                x1, y1, x2, y2 = l_line
                m1 = (y2 - y1) / float(x2 - x1)
                b1 = y1 - m1 * x1
                for r_line in right_lines[:6]:
                    x3, y3, x4, y4 = r_line
                    m2 = (y4 - y3) / float(x4 - x3)
                    b2 = y3 - m2 * x3
                    if abs(m1 - m2) < 0.15:
                        continue
                    ix = (b2 - b1) / (m1 - m2)
                    iy = m1 * ix + b1
                    # Candidate must lie near center lateral strip
                    if (w * 0.25) < ix < (w * 0.75) and (h * 0.20) < iy < (h * 0.65):
                        horizon_candidates.append(iy)

        if horizon_candidates:
            detected_horizon_y = float(np.median(horizon_candidates))
        else:
            # Fallback to smooth nominal horizon
            detected_horizon_y = float(self.nominal_horizon_y)

        # Temporal smoothing if previous pitch available
        if prev_pitch is not None:
            prev_horizon_y = self.cy - self.fy * math.tan(math.radians(prev_pitch))
            smoothed_horizon_y = 0.75 * prev_horizon_y + 0.25 * detected_horizon_y
        else:
            smoothed_horizon_y = detected_horizon_y

        # Delta from nominal horizon translates directly to pitch angle
        delta_y = self.nominal_horizon_y - smoothed_horizon_y
        # pitch in radians = arctan(delta_y / fy)
        pitch_rad = math.atan(delta_y / self.fy)
        pitch_deg = math.degrees(pitch_rad)
        
        # Road Grade % = tan(pitch_rad) * 100%
        road_grade_pct = math.tan(pitch_rad) * 100.0

        return {
            "pitch_deg": round(pitch_deg, 2),
            "road_grade_pct": round(road_grade_pct, 2),
            "horizon_y": int(round(smoothed_horizon_y)),
            "is_uphill": road_grade_pct > 1.5,
            "is_downhill": road_grade_pct < -1.5,
            "is_flat": abs(road_grade_pct) <= 1.5
        }
