"""
Unified Road Quality Measurement Pipeline
Orchestrates preprocessing, perspective geometry, defect detection, tracking, aggregation, and reporting.
"""

import cv2
import numpy as np
import os
from typing import Generator, Dict, List, Optional, Tuple
from src.config import CameraCalibration, CONFIG
from src.preprocessing import PreprocessingEngine
from src.geometry import GeometryEngine
from src.defect_detector import DefectDetector
from src.tracker import SpatialTemporalTracker
from src.aggregator import ContinuousRoadAggregator
from src.report_generator import ReportGenerator

class RoadQualityPipeline:
    """End-to-End Pipeline for continuous road quality measurement."""

    def __init__(self, calib: Optional[CameraCalibration] = None):
        self.calib = calib or CONFIG.camera
        self.prep = PreprocessingEngine()
        self.geom = GeometryEngine(self.calib)
        self.detector = DefectDetector(self.geom, self.prep)
        self.tracker = SpatialTemporalTracker()
        self.aggregator = ContinuousRoadAggregator(
            segment_length_m=CONFIG.aggregator.segment_length_m,
            road_width_m=self.calib.ground_width_m
        )
        self.report_gen = ReportGenerator(CONFIG.reports_dir)
        
        self.current_distance_m = 0.0
        self.current_speed_kmh = 45.0
        self.prev_pitch = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float = 20.0,
        speed_kmh: Optional[float] = None
    ) -> Dict:
        """
        Executes one full processing cycle on a video frame:
        1. Horizon & dynamic pitch/inclination estimation
        2. Defect detection with metric area quantification
        3. Multi-object tracking across time/distance
        4. IPM top-down Bird's Eye View transformation
        5. Continuous segment telemetry aggregation
        """
        if speed_kmh is not None:
            self.current_speed_kmh = speed_kmh
        
        # Advance vehicle odometry
        dt = 1.0 / fps
        speed_mps = self.current_speed_kmh / 3.6
        self.current_distance_m += speed_mps * dt

        # 1. Road Inclination & Horizon Estimation
        inclination_info = self.geom.estimate_road_inclination(frame, prev_pitch=self.prev_pitch)
        pitch_deg = inclination_info["pitch_deg"]
        grade_pct = inclination_info["road_grade_pct"]
        horizon_y = inclination_info["horizon_y"]
        self.prev_pitch = pitch_deg

        # 2. Defect Detection & Metric Area Extraction
        frame_defects = self.detector.detect_defects(frame, horizon_y=horizon_y)

        # 3. Spatial-Temporal Tracking (Prevent double-counting)
        active_tracks = self.tracker.update(frame_defects, frame_idx, self.current_distance_m)

        # 4. Continuous Segment Aggregation
        self.aggregator.update_telemetry(self.current_distance_m, pitch_deg, grade_pct, self.current_speed_kmh)

        # 5. Top-down IPM Bird's Eye View
        ipm_bev = self.geom.generate_ipm_view(frame)

        # 6. Annotate camera frame with bounding boxes, metric area tags, and telemetry
        annotated_frame = self.detector.annotate_frame(frame, frame_defects)
        
        # Telemetry overlay banner
        current_seg = self.aggregator.current_segment
        seg_id = current_seg.segment_id + 1 if current_seg else 1
        
        # Top banner on annotated frame
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, 42), (15, 23, 42), -1)
        cv2.line(annotated_frame, (0, 42), (w, 42), (59, 130, 246), 2)
        
        status_color = (16, 185, 129) if grade_pct > -3.0 and len(frame_defects) == 0 else (239, 68, 68)
        hud_text = f"DIST: {self.current_distance_m:.1f}m | GRADE: {grade_pct:+.1f}% | PITCH: {pitch_deg:+.1f}* | DEFECTS: {len(frame_defects)} | SEG: #{seg_id}"
        cv2.putText(annotated_frame, hud_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        
        return {
            "frame_idx": frame_idx,
            "distance_m": round(self.current_distance_m, 2),
            "speed_kmh": round(self.current_speed_kmh, 1),
            "pitch_deg": pitch_deg,
            "road_grade_pct": grade_pct,
            "horizon_y": horizon_y,
            "defects_in_frame": frame_defects,
            "active_tracks_count": len(active_tracks),
            "annotated_frame": annotated_frame,
            "ipm_bev": ipm_bev,
            "current_segment_id": seg_id
        }

    def process_video(
        self,
        video_path: str,
        output_annotated_video_path: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> Dict:
        """Batch processes an entire video file and generates comprehensive reports."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_writer = None
        if output_annotated_video_path:
            os.makedirs(os.path.dirname(output_annotated_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_annotated_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            result = self.process_frame(frame, frame_idx, fps=fps)
            if out_writer:
                out_writer.write(result["annotated_frame"])

            frame_idx += 1

        cap.release()
        if out_writer:
            out_writer.release()

        # Finalize tracking and segment aggregation
        unique_defects = self.tracker.finalize()
        self.aggregator.assign_completed_defects(unique_defects)
        summary_data = self.aggregator.compute_overall_summary()

        # Generate reports
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        html_report = self.report_gen.generate_html_report(
            summary_data, unique_defects,
            route_title=base_name,
            filename=f"{base_name}_audit_report.html"
        )
        seg_csv, defect_csv = self.report_gen.export_csv_reports(
            summary_data, unique_defects, prefix=base_name
        )

        return {
            "summary": summary_data,
            "unique_defects": unique_defects,
            "html_report_path": html_report,
            "segments_csv_path": seg_csv,
            "defects_csv_path": defect_csv,
            "processed_frames": frame_idx
        }
