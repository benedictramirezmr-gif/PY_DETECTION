import cv2
import numpy as np
import time
import math

from config import LOITER_THRESHOLD_SECS, LOITER_DISTANCE_THRESHOLD, LOITER_ROI_DEFAULT


class TrackedSubject:
    """Represents a single tracked subject inside the loitering zone."""

    _next_id = 1

    def __init__(self, centroid: tuple, current_time: float):
        self.id = TrackedSubject._next_id
        TrackedSubject._next_id += 1
        self.centroid = centroid
        self.first_seen = current_time
        self.last_seen = current_time
        self.active = True

    @property
    def time_in_zone(self) -> float:
        """Total seconds this subject has been tracked inside the zone."""
        return self.last_seen - self.first_seen

    def update(self, centroid: tuple, current_time: float):
        """Update subject position and last-seen timestamp."""
        self.centroid = centroid
        self.last_seen = current_time
        self.active = True


class LoiteringDetector:
    """
    Loitering Detection Engine using centroid-based tracking.
    
    Tracks person bounding boxes across frames using Euclidean distance
    matching. Monitors time spent inside a configurable Region of Interest
    (ROI) polygon or rectangle. Fires a LOITERING WARNING alert when a
    subject remains inside the zone beyond the configured threshold.
    """

    def __init__(self, threshold_secs: float = None, distance_threshold: float = None):
        self.threshold_secs = threshold_secs or LOITER_THRESHOLD_SECS
        self.distance_threshold = distance_threshold or LOITER_DISTANCE_THRESHOLD

        # ROI defined as a list of polygon points [(x, y), ...]
        # Will be computed from frame dimensions on first frame
        self.roi_polygon = None
        self.roi_frac = LOITER_ROI_DEFAULT  # fractional default

        # Tracked subjects: list of TrackedSubject
        self.subjects = []

        # Stale subject timeout (seconds) — remove if not seen for this long
        self.stale_timeout = 5.0

    def set_roi_rect(self, x: int, y: int, w: int, h: int):
        """Define the ROI as a bounding rectangle (pixel coordinates)."""
        self.roi_polygon = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)

    def set_roi_polygon(self, points: list):
        """Define the ROI as an arbitrary polygon.
        
        Args:
            points: List of (x, y) tuples defining the polygon vertices.
        """
        self.roi_polygon = np.array(points, dtype=np.int32)

    def set_threshold(self, seconds: float):
        """Update the loitering time threshold."""
        self.threshold_secs = max(1, seconds)

    def _ensure_roi(self, frame_w: int, frame_h: int):
        """Initialize ROI from fractional defaults if not yet set."""
        if self.roi_polygon is None:
            fx, fy, fw, fh = self.roi_frac
            x = int(fx * frame_w)
            y = int(fy * frame_h)
            w = int(fw * frame_w)
            h = int(fh * frame_h)
            self.set_roi_rect(x, y, w, h)

    def _centroid(self, x: int, y: int, w: int, h: int) -> tuple:
        """Compute the centroid of a bounding box."""
        return (x + w // 2, y + h // 2)

    def _is_inside_roi(self, centroid: tuple) -> bool:
        """Check if a centroid point is inside the ROI polygon."""
        if self.roi_polygon is None:
            return True  # No ROI = entire frame
        result = cv2.pointPolygonTest(self.roi_polygon, centroid, False)
        return result >= 0  # >= 0 means inside or on edge

    def _find_nearest_subject(self, centroid: tuple) -> TrackedSubject:
        """Find the nearest existing subject within distance threshold."""
        best_match = None
        best_dist = float('inf')

        for subj in self.subjects:
            if not subj.active:
                continue
            dist = math.hypot(
                centroid[0] - subj.centroid[0],
                centroid[1] - subj.centroid[1]
            )
            if dist < self.distance_threshold and dist < best_dist:
                best_dist = dist
                best_match = subj

        return best_match

    def update(self, person_detections: list, current_time: float = None) -> list:
        """
        Process new person detections and return any loitering alerts.
        
        Args:
            person_detections: List of (x, y, w, h) bounding boxes for detected persons.
            current_time: Current timestamp (defaults to time.time()).
        
        Returns:
            List of alert dicts: [{"subject_id": int, "duration": float, "centroid": (x, y)}]
        """
        if current_time is None:
            current_time = time.time()

        # Mark all existing subjects as inactive for this frame
        for subj in self.subjects:
            subj.active = False

        alerts = []

        for (x, y, w, h, *_rest) in person_detections:
            centroid = self._centroid(x, y, w, h)

            # Only track subjects inside the ROI
            if not self._is_inside_roi(centroid):
                continue

            # Try to match to existing subject
            matched = self._find_nearest_subject(centroid)
            if matched:
                matched.update(centroid, current_time)
            else:
                # New subject entering the zone
                new_subj = TrackedSubject(centroid, current_time)
                self.subjects.append(new_subj)
                matched = new_subj

            # Check loitering threshold
            if matched.time_in_zone >= self.threshold_secs:
                alerts.append({
                    "subject_id": matched.id,
                    "duration": round(matched.time_in_zone, 1),
                    "centroid": matched.centroid
                })

        # Prune stale subjects (not seen for stale_timeout seconds)
        self.subjects = [
            s for s in self.subjects
            if (current_time - s.last_seen) < self.stale_timeout
        ]

        return alerts

    def draw_overlay(self, frame: np.ndarray, alerts: list = None) -> np.ndarray:
        """
        Draw the ROI zone, tracked subject timers, and loitering warnings on the frame.
        """
        h_frame, w_frame = frame.shape[:2]
        self._ensure_roi(w_frame, h_frame)

        # Draw semi-transparent ROI zone
        if self.roi_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.roi_polygon], (0, 200, 255))  # Yellow-ish
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [self.roi_polygon], True, (0, 200, 255), 2)

            # Label the zone
            roi_center = self.roi_polygon.mean(axis=0).astype(int)
            cv2.putText(
                frame, "RESTRICTED ZONE", (roi_center[0] - 80, roi_center[1] - int(h_frame * 0.15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2
            )

        # Draw tracked subject timers
        current_time = time.time()
        for subj in self.subjects:
            cx, cy = subj.centroid
            duration = round(current_time - subj.first_seen, 1)
            is_loitering = duration >= self.threshold_secs

            # Subject marker
            marker_color = (0, 0, 255) if is_loitering else (0, 255, 255)
            cv2.circle(frame, (cx, cy), 6, marker_color, -1)
            cv2.circle(frame, (cx, cy), 10, marker_color, 2)

            # Timer label
            timer_text = f"ID:{subj.id} {duration}s"
            if is_loitering:
                timer_text = f"⚠ LOITERING ID:{subj.id} {duration}s"
            cv2.putText(
                frame, timer_text, (cx + 14, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 2
            )

        # Draw loitering alert banner if any active alerts
        if alerts:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h_frame - 45), (w_frame, h_frame), (0, 0, 200), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            alert_text = f"⚠ LOITERING WARNING — {len(alerts)} subject(s) exceeded {self.threshold_secs}s threshold"
            cv2.putText(
                frame, alert_text, (15, h_frame - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return frame
