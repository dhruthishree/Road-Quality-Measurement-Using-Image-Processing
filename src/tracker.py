"""
Spatial-Temporal Defect Tracker & Odometry Chaining Engine
Solves Gap 4: Eliminates duplicate detections across consecutive frames using persistent ID tracking
and spatial-temporal trajectory smoothing.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from collections import deque

class TrackedDefect:
    """Represents a unique physical road defect tracked across consecutive frames."""
    
    def __init__(self, track_id: int, initial_defect: Dict, frame_idx: int, distance_m: float):
        self.track_id = track_id
        self.defect_type = initial_defect["type"]
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.first_distance_m = distance_m
        self.last_distance_m = distance_m
        self.hit_streak = 1
        self.time_since_update = 0
        self.is_confirmed = False
        
        # Metric dimensions history for smoothing
        self.areas_m2 = [initial_defect["metric_area_m2"]]
        self.lengths_m = [initial_defect["metric_length_m"]]
        self.widths_m = [initial_defect["metric_width_m"]]
        self.severities = [initial_defect["severity"]]
        self.confidences = [initial_defect["confidence"]]
        
        # Ground coordinate trajectory (X lateral, Y longitudinal)
        self.ground_positions = deque(maxlen=30)
        self.ground_positions.append((initial_defect["ground_x_m"], initial_defect["ground_y_m"]))
        
        # Last known image bounding box
        self.last_bbox = initial_defect["bbox"]
        self.last_center = initial_defect["center_px"]

    def update(self, defect: Dict, frame_idx: int, distance_m: float):
        """Update track with new frame detection."""
        self.last_frame = frame_idx
        self.last_distance_m = distance_m
        self.hit_streak += 1
        self.time_since_update = 0
        
        self.areas_m2.append(defect["metric_area_m2"])
        self.lengths_m.append(defect["metric_length_m"])
        self.widths_m.append(defect["metric_width_m"])
        self.severities.append(defect["severity"])
        self.confidences.append(defect["confidence"])
        
        self.ground_positions.append((defect["ground_x_m"], defect["ground_y_m"]))
        self.last_bbox = defect["bbox"]
        self.last_center = defect["center_px"]
        
        # Confirm track after appearing in at least 3 frames
        if self.hit_streak >= 3:
            self.is_confirmed = True

    @property
    def smoothed_area_m2(self) -> float:
        """Median metric area across observations."""
        return float(np.median(self.areas_m2))

    @property
    def max_severity(self) -> int:
        """Maximum recorded severity index."""
        return int(max(self.severities))

    @property
    def mean_confidence(self) -> float:
        """Mean detection confidence."""
        return float(np.mean(self.confidences))

    def to_dict(self) -> Dict:
        """Summary dictionary for reporting."""
        return {
            "track_id": self.track_id,
            "type": self.defect_type,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_tracked": self.hit_streak,
            "smoothed_area_m2": round(self.smoothed_area_m2, 3),
            "smoothed_length_m": round(float(np.median(self.lengths_m)), 2),
            "smoothed_width_m": round(float(np.median(self.widths_m)), 2),
            "severity": self.max_severity,
            "confidence": round(self.mean_confidence, 2),
            "chainage_distance_m": round(self.first_distance_m, 2),
            "is_confirmed": self.is_confirmed
        }

class SpatialTemporalTracker:
    """
    Associates frame-by-frame defect detections with existing tracks using
    ground-plane Euclidean distance and metric size compatibility.
    """

    def __init__(self, max_cost_distance_m: float = 3.5, max_frames_to_kill: int = 5):
        self.max_cost_dist_m = max_cost_distance_m
        self.max_frames_to_kill = max_frames_to_kill
        self.next_track_id = 1
        self.active_tracks: List[TrackedDefect] = []
        self.completed_tracks: List[TrackedDefect] = []

    def update(self, frame_defects: List[Dict], frame_idx: int, current_distance_m: float) -> List[TrackedDefect]:
        """
        Matches frame detections to active tracks, updates matching tracks,
        spawns new tracks for unmatched detections, and retires inactive tracks.
        """
        # Increment time_since_update for all active tracks
        for t in self.active_tracks:
            t.time_since_update += 1

        unmatched_detections = list(range(len(frame_defects)))
        unmatched_tracks = list(range(len(self.active_tracks)))

        # Bipartite matching based on ground metric distance
        if self.active_tracks and frame_defects:
            cost_matrix = np.zeros((len(self.active_tracks), len(frame_defects)), dtype=np.float32)
            
            for t_idx, track in enumerate(self.active_tracks):
                last_gx, last_gy = track.ground_positions[-1]
                for d_idx, det in enumerate(frame_defects):
                    curr_gx = det["ground_x_m"]
                    curr_gy = det["ground_y_m"]
                    
                    # Lateral distance delta (should be nearly constant for stationary defect)
                    dx = abs(curr_gx - last_gx)
                    # Longitudinal distance should decrease as vehicle advances
                    dy = abs(curr_gy - last_gy)
                    dist = math.sqrt(dx * dx + dy * dy)
                    
                    # Penalty if defect types disagree
                    type_penalty = 0.0 if track.defect_type == det["type"] else 4.0
                    cost_matrix[t_idx, d_idx] = dist + type_penalty

            # Greedy assignment of lowest cost pairs
            matched_tracks = set()
            matched_dets = set()
            
            # Sort all possible pairs by cost
            pairs = []
            for t_i in range(len(self.active_tracks)):
                for d_i in range(len(frame_defects)):
                    pairs.append((cost_matrix[t_i, d_i], t_i, d_i))
            pairs.sort(key=lambda p: p[0])

            for cost, t_i, d_i in pairs:
                if cost > self.max_cost_dist_m:
                    break
                if t_i not in matched_tracks and d_i not in matched_dets:
                    self.active_tracks[t_i].update(frame_defects[d_i], frame_idx, current_distance_m)
                    matched_tracks.add(t_i)
                    matched_dets.add(d_i)

            unmatched_detections = [i for i in range(len(frame_defects)) if i not in matched_dets]
            unmatched_tracks = [i for i in range(len(self.active_tracks)) if i not in matched_tracks]

        # Spawn new tracks for unmatched detections
        for d_i in unmatched_detections:
            new_track = TrackedDefect(
                track_id=self.next_track_id,
                initial_defect=frame_defects[d_i],
                frame_idx=frame_idx,
                distance_m=current_distance_m
            )
            self.next_track_id += 1
            self.active_tracks.append(new_track)

        # Retire inactive tracks (tracks that passed behind vehicle or were lost)
        retained_tracks = []
        for t in self.active_tracks:
            if t.time_since_update > self.max_frames_to_kill:
                if t.is_confirmed or t.hit_streak >= 2:
                    self.completed_tracks.append(t)
            else:
                retained_tracks.append(t)
        self.active_tracks = retained_tracks

        return self.active_tracks

    def finalize(self) -> List[Dict]:
        """Flushes remaining active confirmed tracks and returns full unique defect catalog."""
        for t in self.active_tracks:
            if t.is_confirmed or t.hit_streak >= 2:
                self.completed_tracks.append(t)
        self.active_tracks = []
        return [t.to_dict() for t in self.completed_tracks]
