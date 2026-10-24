"""
Continuous Road Segment Aggregator & Civil Engineering Scoring Engine
Solves Gap 5: Computes continuous chainage-indexed defect density (%), road inclination roughness,
and standardized Road Quality Index (RQI / PCI 0-100) with Good/Fair/Bad classification.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from src.config import CONFIG

class RoadSegment:
    """Represents a continuous discrete road segment (e.g., 50 meters of surveyed pavement)."""

    def __init__(self, segment_id: int, start_dist_m: float, end_dist_m: float, road_width_m: float = 7.0):
        self.segment_id = segment_id
        self.start_dist_m = start_dist_m
        self.end_dist_m = end_dist_m
        self.length_m = end_dist_m - start_dist_m
        self.road_width_m = road_width_m
        self.total_area_m2 = self.length_m * road_width_m
        
        # Telemetry samples inside this segment
        self.pitch_angles = []
        self.road_grades_pct = []
        self.speeds_kmh = []
        self.frames_count = 0
        
        # Defects tracked inside this segment
        self.defects: List[Dict] = []
        self.is_finalized = False
        
        # Computed metrics
        self.pothole_area_m2 = 0.0
        self.crack_area_m2 = 0.0
        self.patch_area_m2 = 0.0
        self.total_defect_area_m2 = 0.0
        
        self.defect_density_pct = 0.0
        self.pothole_density_pct = 0.0
        self.crack_density_pct = 0.0
        
        self.mean_grade_pct = 0.0
        self.grade_std_pct = 0.0
        self.rqi_score = 100.0
        self.quality_rating = "Good"
        self.action_recommendation = "No intervention required."

    def add_telemetry(self, pitch_deg: float, grade_pct: float, speed_kmh: float):
        """Append frame-level inclination and vehicle speed."""
        self.pitch_angles.append(pitch_deg)
        self.road_grades_pct.append(grade_pct)
        self.speeds_kmh.append(speed_kmh)
        self.frames_count += 1

    def add_defect(self, defect_dict: Dict):
        """Register a unique physical defect detected in this segment."""
        self.defects.append(defect_dict)

    def finalize(self):
        """
        Computes final engineering parameters:
        Defect density %, inclination penalty, RQI (0-100), and Good/Fair/Bad rating.
        """
        self.is_finalized = True
        
        # 1. Aggregate defect surface areas by category
        self.pothole_area_m2 = sum(d["smoothed_area_m2"] for d in self.defects if d["type"] == "pothole")
        self.crack_area_m2 = sum(d["smoothed_area_m2"] for d in self.defects if "crack" in d["type"])
        self.patch_area_m2 = sum(d["smoothed_area_m2"] for d in self.defects if d["type"] == "patch")
        self.total_defect_area_m2 = self.pothole_area_m2 + self.crack_area_m2 + self.patch_area_m2

        # 2. Defect Densities (%)
        effective_area = max(10.0, self.total_area_m2)
        self.pothole_density_pct = (self.pothole_area_m2 / effective_area) * 100.0
        self.crack_density_pct = (self.crack_area_m2 / effective_area) * 100.0
        self.patch_density_pct = (self.patch_area_m2 / effective_area) * 100.0
        self.defect_density_pct = (self.total_defect_area_m2 / effective_area) * 100.0

        # 3. Inclination & Slope Roughness
        if self.road_grades_pct:
            self.mean_grade_pct = float(np.mean(self.road_grades_pct))
            self.grade_std_pct = float(np.std(self.road_grades_pct))
        else:
            self.mean_grade_pct = 0.0
            self.grade_std_pct = 0.0

        # Slope roughness penalty for excessive grade fluctuations (> 2.0% variance)
        slope_penalty = max(0.0, (self.grade_std_pct - 1.2) * 2.5) if self.grade_std_pct > 1.2 else 0.0

        # 4. Standardized Road Quality Index (RQI / PCI) Calculation
        # Calibrated according to ASTM D6433 / IRC:SP:16 Pavement Condition Index deduct curves
        cfg = CONFIG.aggregator
        pothole_count = sum(1 for d in self.defects if d["type"] == "pothole")
        crack_count = sum(1 for d in self.defects if "crack" in d["type"])
        patch_count = sum(1 for d in self.defects if d["type"] == "patch")

        # Potholes cause immediate severe vehicle impact and roughness
        deduct_pothole = min(65.0, self.pothole_area_m2 * 22.0 + pothole_count * 8.0)
        # Cracks indicate structural fatigue and moisture intrusion risk
        deduct_crack = min(35.0, self.crack_area_m2 * 14.0 + crack_count * 2.5)
        # Patches represent ride quality discontinuity
        deduct_patch = min(18.0, self.patch_area_m2 * 6.0 + patch_count * 2.0)
        # Dynamic slope roughness penalty
        deduct_slope = slope_penalty * 3.5

        total_deduct = deduct_pothole + deduct_crack + deduct_patch + deduct_slope
        self.rqi_score = round(max(0.0, min(100.0, 100.0 - total_deduct)), 1)

        # 5. Civil Engineering Classification
        if self.rqi_score >= cfg.rqi_good_threshold:
            self.quality_rating = "Good"
            self.action_recommendation = "Pavement in Good condition. Routine preventative maintenance / sweeping."
        elif self.rqi_score >= cfg.rqi_fair_threshold:
            self.quality_rating = "Fair"
            if self.pothole_density_pct > 0.4:
                self.action_recommendation = "Moderate distress. Cold/hot-mix patch potholes and seal open joints."
            else:
                self.action_recommendation = "Minor distress. Surface crack sealing and micro-surfacing recommended."
        else:
            self.quality_rating = "Bad"
            self.action_recommendation = "Severe structural distress. Mill surface and apply dense bituminous asphalt overlay."

    def to_dict(self) -> Dict:
        """Structured dictionary for export and UI rendering."""
        return {
            "segment_id": self.segment_id,
            "chainage_start_m": round(self.start_dist_m, 1),
            "chainage_end_m": round(self.end_dist_m, 1),
            "length_m": round(self.length_m, 1),
            "pothole_count": sum(1 for d in self.defects if d["type"] == "pothole"),
            "crack_count": sum(1 for d in self.defects if "crack" in d["type"]),
            "patch_count": sum(1 for d in self.defects if d["type"] == "patch"),
            "total_defects": len(self.defects),
            "pothole_area_m2": round(self.pothole_area_m2, 3),
            "crack_area_m2": round(self.crack_area_m2, 3),
            "total_defect_area_m2": round(self.total_defect_area_m2, 3),
            "defect_density_pct": round(self.defect_density_pct, 2),
            "mean_grade_pct": round(self.mean_grade_pct, 2),
            "grade_std_pct": round(self.grade_std_pct, 2),
            "rqi_score": self.rqi_score,
            "quality_rating": self.quality_rating,
            "action_recommendation": self.action_recommendation
        }

class ContinuousRoadAggregator:
    """
    Manages continuous spatial-temporal road chaining across entire routes.
    Organizes continuous measurements into fixed-length road segments.
    """

    def __init__(self, segment_length_m: float = 50.0, road_width_m: float = 7.0):
        self.seg_len_m = segment_length_m
        self.road_width_m = road_width_m
        self.segments: List[RoadSegment] = []
        self.current_segment: Optional[RoadSegment] = None
        self._init_segment(0, 0.0)

    def _init_segment(self, seg_id: int, start_m: float):
        """Start a new segment."""
        self.current_segment = RoadSegment(
            segment_id=seg_id,
            start_dist_m=start_m,
            end_dist_m=start_m + self.seg_len_m,
            road_width_m=self.road_width_m
        )
        self.segments.append(self.current_segment)

    def update_telemetry(self, current_dist_m: float, pitch_deg: float, grade_pct: float, speed_kmh: float):
        """Feed continuous frame telemetry and handle segment boundaries."""
        if current_dist_m >= self.current_segment.end_dist_m:
            self.current_segment.finalize()
            next_seg_id = len(self.segments)
            self._init_segment(next_seg_id, self.current_segment.end_dist_m)
            
        self.current_segment.add_telemetry(pitch_deg, grade_pct, speed_kmh)

    def assign_completed_defects(self, unique_defects: List[Dict]):
        """Assigns confirmed tracked defects to their corresponding distance segment and recalculates metrics."""
        for defect in unique_defects:
            chainage_m = defect["chainage_distance_m"]
            # Find matching segment
            matched = False
            for seg in self.segments:
                if seg.start_dist_m <= chainage_m < seg.end_dist_m:
                    seg.add_defect(defect)
                    matched = True
                    break
            if not matched and self.segments:
                self.segments[-1].add_defect(defect)
        
        # Re-finalize all segments with their assigned defects
        for seg in self.segments:
            seg.finalize()

    def finalize(self) -> List[Dict]:
        """Finalize any pending segment and return all segment summaries."""
        if self.current_segment and not self.current_segment.is_finalized:
            self.current_segment.finalize()
        return [s.to_dict() for s in self.segments]

    def compute_overall_summary(self) -> Dict:
        """Computes executive route-level metrics."""
        seg_dicts = [s.to_dict() for s in self.segments]
        if not seg_dicts:
            return {}

        total_dist_m = self.segments[-1].end_dist_m if self.segments else 0.0
        total_potholes = sum(s["pothole_count"] for s in seg_dicts)
        total_cracks = sum(s["crack_count"] for s in seg_dicts)
        total_defect_area = sum(s["total_defect_area_m2"] for s in seg_dicts)
        
        # Mean RQI weighted by segment length
        avg_rqi = float(np.mean([s["rqi_score"] for s in seg_dicts]))
        
        # Segment counts by classification
        good_count = sum(1 for s in seg_dicts if s["quality_rating"] == "Good")
        fair_count = sum(1 for s in seg_dicts if s["quality_rating"] == "Fair")
        bad_count = sum(1 for s in seg_dicts if s["quality_rating"] == "Bad")
        
        if avg_rqi >= CONFIG.aggregator.rqi_good_threshold:
            overall_rating = "Good"
        elif avg_rqi >= CONFIG.aggregator.rqi_fair_threshold:
            overall_rating = "Fair"
        else:
            overall_rating = "Bad"

        return {
            "total_survey_distance_m": round(total_dist_m, 1),
            "total_segments_surveyed": len(seg_dicts),
            "total_unique_potholes": total_potholes,
            "total_unique_cracks": total_cracks,
            "total_defect_area_m2": round(total_defect_area, 3),
            "overall_average_rqi": round(avg_rqi, 1),
            "overall_rating": overall_rating,
            "good_segments_count": good_count,
            "fair_segments_count": fair_count,
            "bad_segments_count": bad_count,
            "segments": seg_dicts
        }
