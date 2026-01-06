import numpy as np
from typing import Tuple, List
from scipy.spatial import distance as dist

class EyeAnalyzer:
    def __init__(self):
        """Initialize eye analyzer"""
        pass
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye landmark
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_eye_closure_ratio(self, upper_eyelid: np.ndarray, lower_eyelid: np.ndarray) -> float:
        """
        Calculate eye closure ratio based on eyelid distance
        """
        # Calculate average distance between upper and lower eyelids
        distances = []
        min_len = min(len(upper_eyelid), len(lower_eyelid))
        
        for i in range(min_len):
            distance = dist.euclidean(upper_eyelid[i], lower_eyelid[i])
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # Normalize by eye width (approximate)
        eye_width = dist.euclidean(upper_eyelid[0], upper_eyelid[-1])
        closure_ratio = avg_distance / eye_width if eye_width > 0 else 0
        
        return closure_ratio
    
    def detect_blink(self, ear: float, threshold: float = 0.25) -> bool:
        """
        Detect if eye is blinking based on EAR threshold
        """
        return ear < threshold
    
    def calculate_blink_features(self, ear_sequence: List[float], 
                               timestamps: List[float]) -> dict:
        """
        Calculate blink-related features from EAR sequence
        """
        ear_array = np.array(ear_sequence)
        time_array = np.array(timestamps)
        
        # Detect blinks
        blink_threshold = 0.25
        is_blink = ear_array < blink_threshold
        
        # Find blink events
        blink_starts = []
        blink_ends = []
        in_blink = False
        
        for i, blink in enumerate(is_blink):
            if blink and not in_blink:
                blink_starts.append(i)
                in_blink = True
            elif not blink and in_blink:
                blink_ends.append(i)
                in_blink = False
        
        # Ensure equal number of starts and ends
        min_len = min(len(blink_starts), len(blink_ends))
        blink_starts = blink_starts[:min_len]
        blink_ends = blink_ends[:min_len]
        
        # Calculate features
        features = {}
        
        # Blink rate (blinks per second)
        total_time = time_array[-1] - time_array[0] if len(time_array) > 1 else 1
        features['blink_rate'] = len(blink_starts) / total_time
        
        # Blink durations
        blink_durations = []
        for start, end in zip(blink_starts, blink_ends):
            duration = time_array[end] - time_array[start]
            blink_durations.append(duration)
        
        if blink_durations:
            features['avg_blink_duration'] = np.mean(blink_durations)
            features['std_blink_duration'] = np.std(blink_durations)
            features['max_blink_duration'] = np.max(blink_durations)
            features['min_blink_duration'] = np.min(blink_durations)
        else:
            features['avg_blink_duration'] = 0
            features['std_blink_duration'] = 0
            features['max_blink_duration'] = 0
            features['min_blink_duration'] = 0
        
        # Blink cycles (time between blinks)
        blink_cycles = []
        for i in range(1, len(blink_starts)):
            cycle = time_array[blink_starts[i]] - time_array[blink_starts[i-1]]
            blink_cycles.append(cycle)
        
        if blink_cycles:
            features['avg_blink_cycle'] = np.mean(blink_cycles)
            features['std_blink_cycle'] = np.std(blink_cycles)
        else:
            features['avg_blink_cycle'] = 0
            features['std_blink_cycle'] = 0
        
        # EAR statistics
        features['avg_ear'] = np.mean(ear_array)
        features['std_ear'] = np.std(ear_array)
        features['min_ear'] = np.min(ear_array)
        features['max_ear'] = np.max(ear_array)
        
        # Blink completeness (how closed eyes get during blinks)
        if blink_durations:
            min_ear_during_blinks = []
            for start, end in zip(blink_starts, blink_ends):
                min_ear = np.min(ear_array[start:end+1])
                min_ear_during_blinks.append(min_ear)
            features['avg_blink_completeness'] = np.mean(min_ear_during_blinks)
        else:
            features['avg_blink_completeness'] = features['min_ear']
        
        return features