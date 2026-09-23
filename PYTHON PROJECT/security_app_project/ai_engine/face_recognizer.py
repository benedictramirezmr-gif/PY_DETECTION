import cv2
import numpy as np
import pickle

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[FaceRecognizer] Warning: face_recognition library not installed.")
    print("[FaceRecognizer] Install with: pip install face_recognition (requires dlib + CMake)")
    print("[FaceRecognizer] Falling back to OpenCV Haar cascade detection only (no recognition).")

from config import FACE_RECOGNITION_TOLERANCE
from database.db_manager import (
    get_all_enrolled_faces,
    insert_enrolled_face as db_insert_enrolled_face
)


class FaceRecognizer:
    """
    Face Recognition & Enrolment Engine.
    
    Uses the face_recognition library (dlib 128-d embeddings) to match live
    camera faces against stored face embeddings from the MySQL database.
    
    Classifies faces as:
        - ENROLLED: [Person Name] (Green overlay)
        - UNKNOWN / INTRUDER (Red overlay)
    
    Graceful fallback: If face_recognition is not installed, uses OpenCV
    Haar cascade for face detection only (all faces labelled UNKNOWN).
    """

    def __init__(self, tolerance: float = None):
        self.tolerance = tolerance or FACE_RECOGNITION_TOLERANCE

        # In-memory cache of enrolled face data
        self.enrolled_names = []       # list of str
        self.enrolled_encodings = []   # list of np.ndarray (128-d each)

        # Fallback Haar cascade for detection-only mode
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load enrolled faces from database
        self.reload_enrolled_faces()

    def reload_enrolled_faces(self):
        """Reload enrolled face embeddings from the MySQL database into memory."""
        self.enrolled_names = []
        self.enrolled_encodings = []

        try:
            faces = get_all_enrolled_faces()
            for face in faces:
                embedding = face.get("embedding")
                name = face.get("full_name", "Unknown")
                if embedding is not None and len(embedding) > 0:
                    self.enrolled_names.append(name)
                    self.enrolled_encodings.append(np.array(embedding))

            print(f"[FaceRecognizer] Loaded {len(self.enrolled_names)} enrolled face(s) from database.")
        except Exception as e:
            print(f"[FaceRecognizer] Error loading enrolled faces: {e}")

    def recognize(self, frame: np.ndarray) -> list:
        """
        Detect and recognize faces in a frame.
        
        Returns:
            List of tuples: [(x, y, w, h, display_name, confidence, is_enrolled)]
            - display_name: "ENROLLED: John Doe" or "UNKNOWN / INTRUDER"
            - confidence: face_distance-based score (0-1, higher = better match)
            - is_enrolled: bool
        """
        results = []

        if FACE_REC_AVAILABLE:
            results = self._recognize_with_dlib(frame)
        else:
            results = self._detect_with_haar(frame)

        return results

    def _recognize_with_dlib(self, frame: np.ndarray) -> list:
        """Full recognition using face_recognition / dlib embeddings."""
        results = []

        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize for performance (process at 1/2 scale)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

        # Detect face locations and compute encodings
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up to original frame size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            x = left
            y = top
            w = right - left
            h = bottom - top

            display_name = "UNKNOWN / INTRUDER"
            confidence = 0.0
            is_enrolled = False

            if len(self.enrolled_encodings) > 0:
                # Compare against all enrolled faces
                face_distances = face_recognition.face_distance(self.enrolled_encodings, face_encoding)
                best_match_idx = np.argmin(face_distances)
                best_distance = face_distances[best_match_idx]

                if best_distance <= self.tolerance:
                    is_enrolled = True
                    display_name = f"ENROLLED: {self.enrolled_names[best_match_idx]}"
                    confidence = round(1.0 - best_distance, 2)
                else:
                    confidence = round(1.0 - best_distance, 2)

            results.append((x, y, w, h, display_name, confidence, is_enrolled))

        return results

    def _detect_with_haar(self, frame: np.ndarray) -> list:
        """Fallback: detection only using Haar cascade (no recognition)."""
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            results.append((x, y, w, h, "UNKNOWN / INTRUDER", 0.0, False))

        return results

    def draw_recognition(self, frame: np.ndarray, recognitions: list) -> np.ndarray:
        """
        Draw face recognition overlays on the frame.
        
        Green boxes + labels for enrolled faces.
        Red boxes + labels for unknown / intruder faces.
        """
        for (x, y, w, h, display_name, confidence, is_enrolled) in recognitions:
            if is_enrolled:
                color = (0, 220, 0)       # Green
                text_bg = (0, 140, 0)
            else:
                color = (0, 0, 255)       # Red
                text_bg = (0, 0, 180)

            # Face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Name label background
            caption = f"{display_name}"
            if confidence > 0:
                caption += f" ({int(confidence * 100)}%)"

            (text_w, text_h), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw label below the face box
            label_y = y + h
            cv2.rectangle(
                frame, (x, label_y), (x + text_w + 10, label_y + text_h + 14),
                text_bg, -1
            )
            cv2.putText(
                frame, caption, (x + 5, label_y + text_h + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

            # Draw corner markers for tactical look
            corner_len = min(w, h) // 5
            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), color, 3)
            cv2.line(frame, (x, y), (x, y + corner_len), color, 3)
            # Top-right
            cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, 3)
            cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, 3)
            # Bottom-left
            cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, 3)
            cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, 3)
            # Bottom-right
            cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, 3)

        return frame

    def enroll_face(self, name: str, role: str, notes: str, image_bgr: np.ndarray) -> tuple:
        """
        Enroll a new face from an image.
        
        Args:
            name: Person's full name.
            role: Role / department.
            notes: Additional notes.
            image_bgr: BGR image (OpenCV format) containing exactly one face.
        
        Returns:
            (success: bool, message: str)
        """
        if not FACE_REC_AVAILABLE:
            return (False, "face_recognition library is not installed. Cannot enroll faces.")

        if image_bgr is None or image_bgr.size == 0:
            return (False, "Invalid image provided.")

        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            face_locations = face_recognition.face_locations(rgb_image, model="hog")

            if len(face_locations) == 0:
                return (False, "No face detected in the image. Please provide a clear face photo.")

            if len(face_locations) > 1:
                return (False, f"Multiple faces ({len(face_locations)}) detected. Please provide an image with exactly one face.")

            # Extract 128-d face embedding
            encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if len(encodings) == 0:
                return (False, "Could not extract face embedding. Please try a different image.")

            embedding = encodings[0]

            # Serialize image to JPEG bytes
            _, img_buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_blob = img_buffer.tobytes()

            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)

            # Store in database
            row_id = db_insert_enrolled_face(name, role, notes, image_blob, embedding_blob)

            if row_id > 0:
                # Update in-memory cache
                self.enrolled_names.append(name)
                self.enrolled_encodings.append(embedding)
                return (True, f"Successfully enrolled '{name}' (ID: {row_id}).")
            else:
                return (False, "Database insert failed. Check database connection.")

        except Exception as e:
            return (False, f"Enrollment error: {str(e)}")
