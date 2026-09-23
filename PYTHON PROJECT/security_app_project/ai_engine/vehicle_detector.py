import cv2
import numpy as np
from pathlib import Path


class VehicleDetector:
    """
    Vehicle Detection and Tracking Engine for CCTV / IP Camera Streams.
    Detects cars, buses, trucks, motorbikes, and bicycles.
    """

    # COCO Class Labels for MobileNet SSD
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    VEHICLE_CLASSES = {"car", "bus", "motorbike", "bicycle", "train"}

    def __init__(self, confidence_threshold: float = 0.45, prototxt_path: str = None, model_path: str = None):
        self.confidence_threshold = confidence_threshold
        self.net = None

        # Attempt to load MobileNet-SSD Caffe model if paths are provided/exist
        if prototxt_path and model_path and Path(prototxt_path).exists() and Path(model_path).exists():
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("[VehicleDetector] Loaded MobileNet SSD DNN Model successfully.")
            except Exception as e:
                print(f"[VehicleDetector] Warning: Could not load DNN model: {e}")

        # Fallback Haar Cascade Classifier for cars if DNN weights are not present
        if self.net is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
            if Path(cascade_path).exists():
                self.car_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self.car_cascade = None

    def detect_vehicles(self, frame: np.ndarray) -> list:
        """
        Processes a video frame and returns detected vehicles.
        
        Returns:
            list of tuples: [(x, y, w, h, label, confidence, color_bgr)]
        """
        h, w = frame.shape[:2]
        detections = []

        # --- MODE 1: DNN Model Processing (MobileNet SSD) ---
        if self.net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                0.007843, 
                (300, 300), 
                127.5
            )
            self.net.setInput(blob)
            output = self.net.forward()

            for i in range(output.shape[2]):
                confidence = output[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    class_id = int(output[0, 0, i, 1])
                    class_name = self.CLASSES[class_id] if class_id < len(self.CLASSES) else "unknown"

                    if class_name in self.VEHICLE_CLASSES:
                        box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        bw = endX - startX
                        bh = endY - startY

                        # Assign UI colors based on vehicle type
                        color = (0, 165, 255) if class_name == "car" else (255, 191, 0)

                        detections.append((
                            startX, startY, bw, bh, 
                            f"VEHICLE: {class_name.upper()}", 
                            round(float(confidence), 2), 
                            color
                        ))

        # --- MODE 2: Fallback Cascade Detection ---
        elif hasattr(self, 'car_cascade') and self.car_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = self.car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

            for (x, y, bw, bh) in cars:
                detections.append((x, y, bw, bh, "VEHICLE: CAR", 0.85, (0, 165, 255)))

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draws bounding boxes and labels for all detected vehicles on the frame."""
        for (x, y, bw, bh, label, conf, color) in detections:
            # Draw primary bounding rectangle
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # Draw background tag for high-visibility HUD text
            caption = f"{label} ({int(conf * 100)}%)"
            (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)

            # Overlay white text
            cv2.putText(
                frame, caption, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

        return frame