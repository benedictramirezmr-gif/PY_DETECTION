import cv2
import time
import numpy as np
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from ai_engine.action_recognizer import ActionRecognizer
from ai_engine.vehicle_detector import VehicleDetector
from ai_engine.hazard_detector import HazardDetector
from ai_engine.loitering_detector import LoiteringDetector
from ai_engine.face_recognizer import FaceRecognizer
from database.db_manager import log_detection_event, log_security_alert
from config import SNAPSHOTS_DIR


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage, int, float)
    alert_triggered = pyqtSignal(str, str, str, str)
    security_alert = pyqtSignal(str, str, str, str, float)

    def __init__(
        self,
        camera_source: str,
        enable_motion=True,
        enable_entity=True,
        enable_actions=True,
        enable_vehicles=True,
        enable_hazard=True,
        enable_loitering=True,
        enable_face_rec=True
    ):
        super().__init__()
        self.camera_source = camera_source
        self.enable_motion = enable_motion
        self.enable_entity = enable_entity
        self.enable_actions = enable_actions
        self.enable_vehicles = enable_vehicles
        self.enable_hazard = enable_hazard
        self.enable_loitering = enable_loitering
        self.enable_face_rec = enable_face_rec
        self.running = True

        # Current frame reference for snapshot capture (used by enrollment tab)
        self.current_frame = None

        # Initialize AI Detection Engines & Cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.fullbody_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        self.recognizer = ActionRecognizer()
        self.vehicle_detector = VehicleDetector()

        # New AI Engines
        self.hazard_detector = HazardDetector()
        self.loitering_detector = LoiteringDetector()
        self.face_recognizer = FaceRecognizer()

    def run(self):
        src = int(self.camera_source) if str(self.camera_source).isdigit() else self.camera_source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"Failed to open video source: {self.camera_source}")
            self.running = False

        snapshots_dir = Path(SNAPSHOTS_DIR)
        snapshots_dir.mkdir(exist_ok=True)

        prev_gray = None
        tracked_entities = []
        last_alert_time = 0
        last_hazard_alert_time = 0
        last_loiter_alert_time = 0
        last_face_alert_time = 0
        frame_counter = 0
        fps_start_time = time.time()
        fps = 0
        motion_pct = 0.0

        # Cached detection results (updated periodically, drawn every frame)
        cached_hazard_detections = []
        cached_face_recognitions = []
        cached_loiter_alerts = []

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            frame_counter += 1
            h, w, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

            # Store current frame for enrollment capture
            self.current_frame = frame.copy()

            # --- 1. MOTION GATING PASS ---
            if self.enable_motion and prev_gray is not None:
                frame_diff = cv2.absdiff(prev_gray, gray_blur)
                thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                motion_pixels = cv2.countNonZero(thresh)
                motion_pct = round((motion_pixels / float(w * h)) * 100, 2)
            else:
                motion_pct = 0.0

            prev_gray = gray_blur

            # Calculate FPS
            if frame_counter % 10 == 0:
                now = time.time()
                fps = int(10 / (now - fps_start_time + 1e-6))
                fps_start_time = now

            # --- 2. ENTITY DETECTION & SUBJECT LOCKING ---
            if frame_counter % 3 == 0 and self.enable_entity:
                tracked_entities = []
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
                )
                for (x, y, bw, bh) in faces:
                    tracked_entities.append((x, y, bw, bh, "LOCKED SUBJECT", (0, 255, 0)))

                if len(faces) == 0:
                    bodies = self.fullbody_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100)
                    )
                    for (x, y, bw, bh) in bodies:
                        tracked_entities.append((x, y, bw, bh, "HUMAN ENTITY", (255, 165, 0)))

            # Draw persistent subject bounding locks
            for (x, y, bw, bh, label, color) in tracked_entities:
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.rectangle(frame, (x, y - 25), (x + bw, y), color, -1)
                cv2.putText(
                    frame, label, (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )

            # --- 3. VEHICLE DETECTION PASS ---
            vehicles_detected = []
            if self.enable_vehicles:
                if frame_counter % 3 == 0:
                    self.last_vehicles_detected = self.vehicle_detector.detect_vehicles(frame)
                vehicles_detected = getattr(self, 'last_vehicles_detected', [])
                frame = self.vehicle_detector.draw_detections(frame, vehicles_detected)

            # --- 4. ACTION PREDICTION & GESTURE PASS ---
            action_str, gesture_str = "DISABLED", "DISABLED"
            if self.enable_actions:
                frame, action_str, gesture_str = self.recognizer.process_frame(frame)

            # Draw HUD Overlays
            cv2.putText(
                frame, f"ACTION: {action_str}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                frame, f"GESTURE: {gesture_str}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
            )

            # --- 5. TRIGGER DATABASE ALERTS (Original) ---
            current_time = time.time()
            has_vehicle = len(vehicles_detected) > 0
            
            if (action_str == "RUNNING" or gesture_str == "RAISED HAND / STOP" or motion_pct > 3.0 or has_vehicle) and (current_time - last_alert_time > 8):
                last_alert_time = current_time

                snapshot_filename = f"event_{int(current_time)}.jpg"
                snapshot_filepath = snapshots_dir / snapshot_filename
                cv2.imwrite(str(snapshot_filepath), frame)

                cam_ip_str = str(self.camera_source)
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                
                event_desc = f"{action_str} | {gesture_str}"
                if has_vehicle:
                    event_desc += f" | {vehicles_detected[0][4]}"

                log_detection_event(
                    camera_ip=cam_ip_str,
                    person_name=event_desc,
                    is_intruder=True,
                    confidence=0.91,
                    snapshot_path=str(snapshot_filepath)
                )

                self.alert_triggered.emit(timestamp_str, cam_ip_str, event_desc, "SECURITY TRIGGER")

            # --- 5.5 HAZARD DETECTION (YOLO) ---
            if self.enable_hazard:
                if frame_counter % 3 == 0:
                    cached_hazard_detections = self.hazard_detector.detect(frame)

                if cached_hazard_detections:
                    frame = self.hazard_detector.draw_alerts(frame, cached_hazard_detections)

                    # Emit security alert (throttled)
                    if cached_hazard_detections and (current_time - last_hazard_alert_time > 10):
                        last_hazard_alert_time = current_time

                        snapshot_filename = f"hazard_{int(current_time)}.jpg"
                        snapshot_filepath = snapshots_dir / snapshot_filename
                        cv2.imwrite(str(snapshot_filepath), frame)

                        cam_ip_str = str(self.camera_source)
                        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

                        for det in cached_hazard_detections:
                            label = det[4]
                            conf = det[5]
                            desc = f"{label} detected with {int(conf * 100)}% confidence"

                            log_security_alert(
                                camera_ip=cam_ip_str,
                                alert_type=f"HAZARD: {label}",
                                description=desc,
                                confidence=conf,
                                snapshot_path=str(snapshot_filepath)
                            )

                            self.security_alert.emit(
                                timestamp_str, cam_ip_str,
                                f"HAZARD: {label}", desc, conf
                            )

            # --- 5.6 LOITERING DETECTION ---
            if self.enable_loitering:
                # Feed person bounding boxes (from entity detection) into loitering tracker
                person_boxes = [(x, y, bw, bh) for (x, y, bw, bh, lbl, clr) in tracked_entities]
                cached_loiter_alerts = self.loitering_detector.update(person_boxes, current_time)
                frame = self.loitering_detector.draw_overlay(frame, cached_loiter_alerts)

                # Emit loitering alerts (throttled)
                if cached_loiter_alerts and (current_time - last_loiter_alert_time > 15):
                    last_loiter_alert_time = current_time

                    snapshot_filename = f"loiter_{int(current_time)}.jpg"
                    snapshot_filepath = snapshots_dir / snapshot_filename
                    cv2.imwrite(str(snapshot_filepath), frame)

                    cam_ip_str = str(self.camera_source)
                    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

                    for alert in cached_loiter_alerts:
                        desc = f"Subject ID:{alert['subject_id']} loitering for {alert['duration']}s"

                        log_security_alert(
                            camera_ip=cam_ip_str,
                            alert_type="LOITERING WARNING",
                            description=desc,
                            confidence=0.85,
                            snapshot_path=str(snapshot_filepath)
                        )

                        self.security_alert.emit(
                            timestamp_str, cam_ip_str,
                            "LOITERING WARNING", desc, 0.85
                        )

            # --- 5.7 FACE RECOGNITION ---
            if self.enable_face_rec:
                if frame_counter % 5 == 0:
                    cached_face_recognitions = self.face_recognizer.recognize(frame)

                if cached_face_recognitions:
                    frame = self.face_recognizer.draw_recognition(frame, cached_face_recognitions)

                    # Emit alerts for unknown / intruder faces (throttled)
                    unknown_faces = [r for r in cached_face_recognitions if not r[6]]
                    if unknown_faces and (current_time - last_face_alert_time > 10):
                        last_face_alert_time = current_time

                        snapshot_filename = f"intruder_{int(current_time)}.jpg"
                        snapshot_filepath = snapshots_dir / snapshot_filename
                        cv2.imwrite(str(snapshot_filepath), frame)

                        cam_ip_str = str(self.camera_source)
                        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

                        desc = f"{len(unknown_faces)} unknown face(s) detected"

                        log_security_alert(
                            camera_ip=cam_ip_str,
                            alert_type="INTRUDER DETECTED",
                            description=desc,
                            confidence=0.90,
                            snapshot_path=str(snapshot_filepath)
                        )

                        self.security_alert.emit(
                            timestamp_str, cam_ip_str,
                            "INTRUDER DETECTED", desc, 0.90
                        )

            # Convert BGR (OpenCV) to RGB (PyQt6 QImage)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)

            self.frame_ready.emit(qt_img, fps, motion_pct)
            time.sleep(0.02)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()