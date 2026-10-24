"""
System Configuration & Calibration Parameters
Road Quality Measurement using Image Processing
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import os

@dataclass
class CameraCalibration:
    """Camera mounting and intrinsic parameters for perspective geometry & IPM."""
    # Camera installation parameters
    camera_height_m: float = 1.35          # Mounting height above road plane in meters
    pitch_angle_deg: float = 8.5           # Downward pitch angle of camera relative to horizon
    roll_angle_deg: float = 0.0            # Roll angle (camera tilt)
    fov_vertical_deg: float = 48.0         # Vertical Field of View
    fov_horizontal_deg: float = 72.0       # Horizontal Field of View
    
    # Image frame defaults
    frame_width: int = 1280
    frame_height: int = 720
    
    # Ground plane ROI in meters relative to front bumper (camera footprint)
    ground_near_dist_m: float = 3.0        # Nearest distance captured in road ROI
    ground_far_dist_m: float = 25.0        # Farthest distance mapped in road ROI
    ground_width_m: float = 7.0            # Width of ground ROI (covers 2 lanes + shoulders)
    
    # Inverse Perspective Mapping (IPM) Bird's Eye View resolution
    ipm_width: int = 400                   # Width in pixels for top-down road plane
    ipm_height: int = 600                  # Height in pixels for top-down road plane

@dataclass
class DefectDetectionConfig:
    """Thresholds and feature filters for defect segmentation and classification."""
    # Preprocessing
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    
    # Retinex shadow suppression parameters
    retinex_scales: List[int] = field(default_factory=lambda: [15, 80, 250])
    shadow_luminance_threshold: int = 65
    
    # Road ROI (fraction of image from top to ignore sky / hood)
    roi_top_fraction: float = 0.45          # Discard upper 45% (sky, horizon, trees)
    roi_bottom_fraction: float = 0.95       # Discard bottom 5% (vehicle bonnet / hood)
    roi_side_margin: float = 0.08           # Discard extreme outer edges
    
    # Defect geometry filtering in metric units (meters)
    min_pothole_area_m2: float = 0.02       # ~14cm x 14cm minimum detectable pothole
    max_pothole_area_m2: float = 4.0        # Upper limit for single localized pothole
    min_crack_length_m: float = 0.25        # 25cm minimum crack line length
    min_defect_aspect_ratio_crack: float = 2.8 # Eccentricity threshold for crack classification
    
    # Confidence & depth severity proxies
    depth_severity_intensity_weight: float = 0.6
    edge_contrast_weight: float = 0.4

@dataclass
class AggregationConfig:
    """Road segment chaining, scoring weights, and quality classification."""
    segment_length_m: float = 50.0          # Segment chunk length for continuous aggregation (50m)
    min_frames_per_segment: int = 15        # Minimum frames required to finalize a segment
    
    # Quality score weights (ASTM D6433 / IRC aligned distress weighting)
    # RQI (0-100) = 100 - (w_pothole * Pothole_Density + w_crack * Crack_Density + w_patch * Patch_Density + w_incline * Slope_Penalty)
    weight_pothole_density: float = 6.5     # High impact on roughness & vehicular safety
    weight_crack_density: float = 3.2       # Structural degradation indicator
    weight_patch_density: float = 1.8       # Repaired surface irregularities
    weight_roughness_incline: float = 1.2    # Dynamic slope variation penalty
    
    # Classification thresholds for Road Quality Index (RQI)
    rqi_good_threshold: float = 80.0        # RQI >= 80 -> Good (Smooth, regular maintenance)
    rqi_fair_threshold: float = 60.0        # 60 <= RQI < 80 -> Fair (Moderate defects, scheduled maintenance)
                                            # RQI < 60 -> Bad / Critical (Immediate resurfacing needed)

@dataclass
class SystemConfig:
    """Master configuration binding all subsystems."""
    camera: CameraCalibration = field(default_factory=CameraCalibration)
    detector: DefectDetectionConfig = field(default_factory=DefectDetectionConfig)
    aggregator: AggregationConfig = field(default_factory=AggregationConfig)
    
    # Output directories
    output_dir: str = os.path.join(os.path.dirname(__file__), "..", "output")
    reports_dir: str = os.path.join(os.path.dirname(__file__), "..", "output", "reports")
    dataset_dir: str = os.path.join(os.path.dirname(__file__), "..", "dataset")

# Global singleton configuration
CONFIG = SystemConfig()
