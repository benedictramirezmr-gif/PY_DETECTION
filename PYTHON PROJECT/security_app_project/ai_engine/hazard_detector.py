import cv2
import numpy as np
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[HazardDetector] Warning: ultralytics not installed. Hazard detection disabled.")

from config import HAZARD_MODEL_PATH, HAZARD_CONFIDENCE


class HazardDetector:
    """
    Hazard Detection Engine using Ultralytics YOLO.
    
    Detects dangerous objects: Knives, Firearms (if custom model provided),
    Fire, and Smoke. Falls back to COCO-pretrained classes when no custom
    model is available.
    
    COCO class IDs used:
        43 = knife
        
    Custom model class names (if trained): gun, pistol, rifle, fire, smoke, knife
    """

    # COCO classes that map to security hazards
    COCO_HAZARD_MAP = {
        43: ("KNIFE", (0, 0, 255)),           # Red
    }

    # Custom model hazard class names → display config
    CUSTOM_HAZARD_MAP = {
        "knife":   ("KNIFE DETECTED",   (0, 0, 255)),      # Red
        "gun":     ("FIREARM DETECTED", (0, 0, 255)),      # Red
        "pistol":  ("FIREARM DETECTED", (0, 0, 255)),      # Red
        "rifle":   ("FIREARM DETECTED", (0, 0, 255)),      # Red
        "fire":    ("FIRE DETECTED",    (0, 0, 255)),      # Red
        "smoke":   ("SMOKE DETECTED",   (0, 140, 255)),    # Orange
    }

    def __init__(self, model_path: str = None, confidence: float = None):
        self.confidence = confidence or HAZARD_CONFIDENCE
        self.model = None
        self.is_custom_model = False

        if not YOLO_AVAILABLE:
            print("[HazardDetector] YOLO unavailable — hazard detection will be skipped.")
            return

        model_file = model_path or HAZARD_MODEL_PATH

        try:
            self.model = YOLO(model_file)
            # Check if this is a custom-trained model by inspecting class names
            class_names = list(self.model.names.values()) if hasattr(self.model, 'names') else []
            custom_keys = set(self.CUSTOM_HAZARD_MAP.keys())
            if any(name.lower() in custom_keys for name in class_names):
                self.is_custom_model = True
                print(f"[HazardDetector] Loaded CUSTOM hazard model: {model_file}")
            else:
                self.is_custom_model = False
                print(f"[HazardDetector] Loaded COCO-pretrained model: {model_file}")
                print(f"[HazardDetector] Detecting: knife (COCO class 43). For gun/fire/smoke, supply a custom model.")
        except Exception as e:
            print(f"[HazardDetector] Error loading model '{model_file}': {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> list:
        """
        Run hazard detection on a single frame.
        
        Returns:
            List of tuples: [(x, y, w, h, label, confidence, color_bgr)]
        """
        if self.model is None:
            return []

        detections = []

        try:
            results = self.model(frame, conf=self.confidence, verbose=False)

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names.get(class_id, "").lower()

                    label = None
                    color = (0, 0, 255)

                    if self.is_custom_model:
                        # Use custom model class mapping
                        if class_name in self.CUSTOM_HAZARD_MAP:
                            label, color = self.CUSTOM_HAZARD_MAP[class_name]
                    else:
                        # Use COCO class ID mapping
                        if class_id in self.COCO_HAZARD_MAP:
                            label, color = self.COCO_HAZARD_MAP[class_id]

                    if label is not None:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        w = x2 - x1
                        h = y2 - y1
                        detections.append((x1, y1, w, h, label, round(conf, 2), color))

        except Exception as e:
            print(f"[HazardDetector] Detection error: {e}")

        return detections

    def draw_alerts(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws high-visibility alert overlays on the frame for detected hazards.
        
        Uses thick red/orange bounding boxes with pulsing opacity effect,
        large warning text, and a semi-transparent danger banner.
        """
        if not detections:
            return frame

        h_frame, w_frame = frame.shape[:2]

        # Draw semi-transparent warning banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 50), (0, 0, 180), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Pulsing effect based on time
        pulse = abs(int(time.time() * 4) % 6 - 3)  # oscillates 0-3
        thickness_base = 3

        # Banner text
        hazard_labels = [d[4] for d in detections]
        banner_text = "  ⚠ HAZARD ALERT: " + " | ".join(set(hazard_labels))
        cv2.putText(
            frame, banner_text, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        for (x, y, w, h, label, conf, color) in detections:
            # Thick bounding box with pulse
            box_thickness = thickness_base + pulse
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)

            # Corner accents for tactical look
            corner_len = min(w, h) // 4
            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), (255, 255, 255), 3)
            cv2.line(frame, (x, y), (x, y + corner_len), (255, 255, 255), 3)
            # Top-right
            cv2.line(frame, (x + w, y), (x + w - corner_len, y), (255, 255, 255), 3)
            cv2.line(frame, (x + w, y), (x + w, y + corner_len), (255, 255, 255), 3)
            # Bottom-left
            cv2.line(frame, (x, y + h), (x + corner_len, y + h), (255, 255, 255), 3)
            cv2.line(frame, (x, y + h), (x, y + h - corner_len), (255, 255, 255), 3)
            # Bottom-right
            cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), (255, 255, 255), 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), (255, 255, 255), 3)

            # Label background
            caption = f"{label} ({int(conf * 100)}%)"
            (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x, y - text_h - 14), (x + text_w + 12, y), color, -1)
            cv2.putText(
                frame, caption, (x + 6, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )

        return frame
