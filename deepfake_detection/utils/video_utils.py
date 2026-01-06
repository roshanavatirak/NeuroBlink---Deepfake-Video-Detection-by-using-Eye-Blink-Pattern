import cv2
import os
from typing import List, Tuple

def extract_frames(video_path: str, output_dir: str, max_frames: int = None) -> List[str]:
    """
    Extract frames from video file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        frame_filename = f"frame_{frame_count:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    return frame_paths

def get_video_info(video_path: str) -> dict:
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

def resize_video(input_path: str, output_path: str, target_size: Tuple[int, int] = (640, 480)):
    """Resize video to target dimensions"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, target_size)
        out.write(resized_frame)
    
    cap.release()
    out.release()

def split_video(video_path: str, segment_duration: int = 10) -> List[str]:
    """Split video into segments of specified duration"""
    info = get_video_info(video_path)
    if not info:
        return []
    
    segments = []
    total_duration = info['duration']
    fps = info['fps']
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), f"{base_name}_segments")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    segment_count = 0
    
    while cap.get(cv2.CAP_PROP_POS_MSEC) < total_duration * 1000:
        segment_path = os.path.join(output_dir, f"segment_{segment_count:03d}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(segment_path, fourcc, fps, 
                             (int(info['width']), int(info['height'])))
        
        frames_written = 0
        max_frames = int(fps * segment_duration)
        
        while frames_written < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        if frames_written > 0:
            segments.append(segment_path)
        segment_count += 1
    
    cap.release()
    return segments