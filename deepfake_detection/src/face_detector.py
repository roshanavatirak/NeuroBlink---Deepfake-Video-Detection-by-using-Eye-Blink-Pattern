import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize face detector using OpenCV DNN
        """
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.load_face_detector()
    
    def load_face_detector(self):
        """Load pre-trained face detection model"""
        try:
            # Using OpenCV's DNN face detector
            prototxt_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            
            # For this example, we'll use Haar Cascade as backup
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Face detector loaded successfully")
        except Exception as e:
            print(f"Error loading face detector: {e}")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame
        Returns: List of bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        return faces
    
    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Get the largest detected face"""
        if len(faces) == 0:
            return None
        
        # Find face with largest area
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        return largest_face