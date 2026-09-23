import cv2
import mediapipe as mp

class ActionRecognizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        
        self.mp_draw = mp.solutions.drawing_utils

        self.prev_hip_y = None
        self.prev_ankle_x = None

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detected_action = "STANDING / IDLE"
        detected_gesture = "NONE"

        # --- POSE TRACKING (WALKING, RUNNING, VIEWING) ---
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            avg_hip_y = (left_hip[1] + right_hip[1]) / 2.0
            avg_ankle_x = (left_ankle[0] + right_ankle[0]) / 2.0

            if self.prev_hip_y is not None and self.prev_ankle_x is not None:
                vert_motion = abs(avg_hip_y - self.prev_hip_y)
                horiz_stride = abs(avg_ankle_x - self.prev_ankle_x)

                if horiz_stride > 0.04 or vert_motion > 0.03:
                    detected_action = "RUNNING"
                elif horiz_stride > 0.01 or vert_motion > 0.008:
                    detected_action = "WALKING"
                else:
                    detected_action = "VIEWING / LOITERING"

            self.prev_hip_y = avg_hip_y
            self.prev_ankle_x = avg_ankle_x

        # --- HAND GESTURE RECOGNITION ---
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]

                if (index_tip.y < wrist.y and middle_tip.y < wrist.y and 
                    ring_tip.y < wrist.y and pinky_tip.y < wrist.y):
                    detected_gesture = "RAISED HAND / STOP"
                elif index_tip.y < wrist.y and middle_tip.y > wrist.y:
                    detected_gesture = "POINTING / GESTURE"

        return frame, detected_action, detected_gesture
