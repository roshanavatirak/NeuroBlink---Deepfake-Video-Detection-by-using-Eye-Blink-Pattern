import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple

class LandmarkDetector:
    def __init__(self):
        """Initialize MediaPipe face mesh detector"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe 468 landmarks
        self.LEFT_EYE_INDICES = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        self.RIGHT_EYE_INDICES = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]
        
        # Upper and lower eyelid landmarks
        self.LEFT_EYE_UPPER = [159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.LEFT_EYE_LOWER = [33, 246, 161, 160]
        self.RIGHT_EYE_UPPER = [386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        self.RIGHT_EYE_LOWER = [362, 398, 384, 385]
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect 468 facial landmarks using MediaPipe
        Returns: Array of landmark coordinates or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append([x, y])
            
            return np.array(landmark_points)
        
        return None
    
    def get_eye_landmarks(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks from all facial landmarks
        Returns: (left_eye_landmarks, right_eye_landmarks)
        """
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        return left_eye, right_eye
    
    def get_eyelid_landmarks(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get upper and lower eyelid landmarks
        Returns: (left_upper, left_lower, right_upper, right_lower)
        """
        left_upper = landmarks[self.LEFT_EYE_UPPER]
        left_lower = landmarks[self.LEFT_EYE_LOWER]
        right_upper = landmarks[self.RIGHT_EYE_UPPER]
        right_lower = landmarks[self.RIGHT_EYE_LOWER]
        
        return left_upper, left_lower, right_upper, right_lower