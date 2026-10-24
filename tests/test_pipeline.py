"""
Unit & Integration Test Suite for Road Quality Measurement System
"""

import unittest
import numpy as np
import cv2
import os
import glob
from src.config import CameraCalibration, CONFIG
from src.preprocessing import PreprocessingEngine
from src.geometry import GeometryEngine
from src.defect_detector import DefectDetector
from src.tracker import SpatialTemporalTracker
from src.aggregator import ContinuousRoadAggregator, RoadSegment
from src.report_generator import ReportGenerator

class TestRoadQualityPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.geom = GeometryEngine()
        cls.prep = PreprocessingEngine()
        cls.detector = DefectDetector(cls.geom, cls.prep)
        cls.test_images = glob.glob(os.path.join(os.path.dirname(__file__), "..", "dataset", "test_images", "*.jpg"))

    def test_01_preprocessing_illumination_and_shadows(self):
        """Test shadow attenuation and lane line suppression."""
        sample_img = np.full((720, 1280, 3), 70, dtype=np.uint8)
        # Add artificial shadow
        cv2.rectangle(sample_img, (300, 300), (600, 500), (25, 25, 28), -1)
        # Add white lane marking
        cv2.line(sample_img, (640, 350), (640, 600), (240, 240, 245), 10)

        enhanced, gray, markings_mask = self.prep.preprocess_pipeline(sample_img)
        self.assertEqual(enhanced.shape, sample_img.shape)
        self.assertEqual(gray.shape, (720, 1280))
        # Marking mask should identify the white line
        self.assertGreater(np.sum(markings_mask > 0), 500)

    def test_02_geometry_and_ipm_mapping(self):
        """Test camera perspective to ground-plane metric coordinates."""
        # Bottom center pixel (near vehicle bumper)
        x_m, y_m = self.geom.pixel_to_ground_metric(640, 700)
        self.assertAlmostEqual(x_m, 0.0, delta=0.5) # Should be centered laterally
        self.assertGreater(y_m, 2.0)
        self.assertLess(y_m, 8.0)

        # Far ground pixel
        x_far, y_far = self.geom.pixel_to_ground_metric(640, 350)
        self.assertGreater(y_far, y_m)

    def test_03_road_inclination_estimation(self):
        """Test vanishing point and road pitch grade calculation."""
        sample_img = np.full((720, 1280, 3), 70, dtype=np.uint8)
        # Draw converging road lane lines
        cv2.line(sample_img, (540, 320), (100, 720), (200, 200, 200), 4)
        cv2.line(sample_img, (740, 320), (1180, 720), (200, 200, 200), 4)

        info = self.geom.estimate_road_inclination(sample_img)
        self.assertIn("pitch_deg", info)
        self.assertIn("road_grade_pct", info)
        self.assertIn("horizon_y", info)
        self.assertIsInstance(info["is_flat"], bool)

    def test_04_defect_detection_on_benchmark_images(self):
        """Test defect extraction and metric characterization on generated images."""
        if self.test_images:
            test_img = cv2.imread(self.test_images[0])
            defects = self.detector.detect_defects(test_img)
            self.assertIsInstance(defects, list)
            for d in defects:
                self.assertIn("type", d)
                self.assertIn("metric_area_m2", d)
                self.assertGreater(d["metric_area_m2"], 0.0)
                self.assertIn("severity", d)
                self.assertGreaterEqual(d["severity"], 1)
                self.assertLessEqual(d["severity"], 5)

    def test_05_spatial_temporal_tracking_eliminates_duplicates(self):
        """Verify tracker tracks the same defect across successive frames with 1 unique ID."""
        tracker = SpatialTemporalTracker()
        
        # Simulate defect approaching vehicle over 5 successive frames
        defect_sim = {
            "type": "pothole",
            "bbox": [500, 400, 60, 40],
            "center_px": [530, 420],
            "pixel_area": 1800,
            "metric_area_m2": 0.45,
            "metric_length_m": 0.70,
            "metric_width_m": 0.65,
            "ground_x_m": 0.20,
            "ground_y_m": 12.0,
            "confidence": 0.90,
            "severity": 4,
            "contrast_ratio": 0.35,
            "contour": []
        }

        for f_idx in range(5):
            # Defect gets closer
            defect_sim["ground_y_m"] -= 1.5
            dist = f_idx * 1.5
            tracker.update([defect_sim], frame_idx=f_idx, current_distance_m=dist)

        final_defects = tracker.finalize()
        # Exactly 1 physical defect should be registered, NOT 5!
        self.assertEqual(len(final_defects), 1)
        self.assertEqual(final_defects[0]["track_id"], 1)
        self.assertEqual(final_defects[0]["frames_tracked"], 5)

    def test_06_continuous_road_aggregator_and_rqi_classification(self):
        """Test segment chaining, ASTM D6433 deduct curve, and Good/Fair/Bad rating."""
        aggregator = ContinuousRoadAggregator(segment_length_m=50.0)
        
        # Segment 1: Perfect smooth road (0m to 50m)
        for i in range(25):
            aggregator.update_telemetry(current_dist_m=i * 2.0, pitch_deg=0.5, grade_pct=0.9, speed_kmh=50.0)
        
        # Segment 2: Severely distressed road (50m to 100m)
        for i in range(25, 50):
            aggregator.update_telemetry(current_dist_m=i * 2.0, pitch_deg=1.5, grade_pct=2.6, speed_kmh=40.0)

        # Assign severe defects to Segment 2 (chainage 75m)
        severe_defects = [
            {"track_id": 1, "type": "pothole", "chainage_distance_m": 72.0, "smoothed_area_m2": 0.85},
            {"track_id": 2, "type": "pothole", "chainage_distance_m": 80.0, "smoothed_area_m2": 1.20},
            {"track_id": 3, "type": "crack_alligator", "chainage_distance_m": 85.0, "smoothed_area_m2": 1.50}
        ]
        aggregator.assign_completed_defects(severe_defects)
        summary = aggregator.compute_overall_summary()
        
        segments = summary["segments"]
        self.assertEqual(len(segments), 2)
        
        # Segment 1 must be GOOD
        self.assertEqual(segments[0]["quality_rating"], "Good")
        self.assertGreaterEqual(segments[0]["rqi_score"], 90.0)
        
        # Segment 2 must be BAD due to severe potholes and alligator cracks
        self.assertEqual(segments[1]["quality_rating"], "Bad")
        self.assertLess(segments[1]["rqi_score"], 60.0)

    def test_07_report_generation(self):
        """Test HTML and CSV report generation."""
        rep_dir = os.path.join(os.path.dirname(__file__), "..", "output", "test_reports")
        gen = ReportGenerator(rep_dir)
        
        dummy_summary = {
            "total_survey_distance_m": 100.0,
            "total_segments_surveyed": 2,
            "total_unique_potholes": 1,
            "total_unique_cracks": 1,
            "total_defect_area_m2": 0.75,
            "overall_average_rqi": 78.5,
            "overall_rating": "Fair",
            "segments": [
                {
                    "segment_id": 0, "chainage_start_m": 0, "chainage_end_m": 50,
                    "length_m": 50, "pothole_count": 0, "crack_count": 0, "patch_count": 0,
                    "total_defects": 0, "pothole_area_m2": 0, "crack_area_m2": 0,
                    "total_defect_area_m2": 0, "defect_density_pct": 0, "mean_grade_pct": 1.0,
                    "grade_std_pct": 0.2, "rqi_score": 98.0, "quality_rating": "Good",
                    "action_recommendation": "Routine maintenance"
                },
                {
                    "segment_id": 1, "chainage_start_m": 50, "chainage_end_m": 100,
                    "length_m": 50, "pothole_count": 1, "crack_count": 1, "patch_count": 0,
                    "total_defects": 2, "pothole_area_m2": 0.5, "crack_area_m2": 0.25,
                    "total_defect_area_m2": 0.75, "defect_density_pct": 0.21, "mean_grade_pct": 2.0,
                    "grade_std_pct": 0.5, "rqi_score": 59.0, "quality_rating": "Bad",
                    "action_recommendation": "Resurfacing required"
                }
            ]
        }
        
        dummy_defects = [
            {"track_id": 1, "type": "pothole", "chainage_distance_m": 65.0, "smoothed_area_m2": 0.5,
             "smoothed_length_m": 0.8, "smoothed_width_m": 0.7, "severity": 4, "confidence": 0.92}
        ]

        html_path = gen.generate_html_report(dummy_summary, dummy_defects)
        seg_csv, def_csv = gen.export_csv_reports(dummy_summary, dummy_defects)

        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(os.path.exists(seg_csv))
        self.assertTrue(os.path.exists(def_csv))

if __name__ == "__main__":
    unittest.main()
