#!/usr/bin/env python3
"""
Fixed Prediction script for deepfake detection
"""

import os
import sys
import argparse
import cv2
import numpy as np
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extractor import FeatureExtractor
from src.classifier import DeepfakeClassifier
from utils.visualization import create_detection_report, visualize_blink_detection

def predict_single_video(video_path: str, model_path: str, 
                        save_report: bool = False, 
                        show_visualization: bool = False) -> tuple:
    """
    Predict if a single video is fake or real
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model
        save_report: Whether to save detailed HTML report
        show_visualization: Whether to show real-time visualization
    
    Returns:
        (prediction, confidence, features)
    """
    print(f"Analyzing video: {os.path.basename(video_path)}")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    print("Loading trained model...")
    classifier = DeepfakeClassifier()
    classifier.load_model(model_path)
    
    # Extract features
    print("Extracting features from video...")
    extractor = FeatureExtractor()
    features = extractor.extract_video_features(video_path)
    
    # Make prediction
    print("Making prediction...")
    prediction, confidence = classifier.predict(features)
    
    # Display results
    result = "FAKE" if prediction == 1 else "REAL"
    print(f"\n🎯 PREDICTION: {result}")
    print(f"🔍 CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"📊 VIDEO DURATION: {features.get('video_duration', 0):.2f} seconds")
    print(f"🎞️  TOTAL FRAMES: {features.get('total_frames', 0)}")
    print(f"📈 BLINK RATE: {features.get('avg_blink_rate', 0):.3f} blinks/sec")
    
    # Show key features
    print("\n📋 KEY FEATURES:")
    key_features = [
        ('avg_blink_rate', 'Average Blink Rate'),
        ('avg_avg_blink_duration', 'Average Blink Duration'),
        ('avg_avg_blink_cycle', 'Average Blink Cycle'),
        ('avg_avg_ear', 'Average EAR'),
        ('avg_std_ear', 'EAR Standard Deviation')
    ]
    
    for feature_key, feature_name in key_features:
        value = features.get(feature_key, 0)
        print(f"  • {feature_name}: {value:.6f}")
    
    # Save detailed report if requested
    if save_report:
        report_path = create_detection_report(
            video_path, prediction, confidence, features
        )
        print(f"\n📄 Detailed report saved: {report_path}")
    
    # Show real-time visualization if requested
    if show_visualization:
        show_real_time_analysis(video_path, extractor)
    
    return prediction, confidence, features

def show_real_time_analysis(video_path: str, extractor: FeatureExtractor):
    """Show real-time blink detection visualization"""
    print("\n🎬 Starting real-time analysis... (Press 'q' to quit)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for visualization")
        return
    
    frame_count = 0
    ear_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            faces = extractor.face_detector.detect_faces(frame)
            largest_face = extractor.face_detector.get_largest_face(faces)
            
            if largest_face is not None:
                x, y, w, h = largest_face
                face_region = frame[y:y+h, x:x+w]
                
                # Detect landmarks
                landmarks = extractor.landmark_detector.detect_landmarks(face_region)
                
                if landmarks is not None:
                    # Adjust landmarks to original frame coordinates
                    landmarks[:, 0] += x
                    landmarks[:, 1] += y
                    
                    # Get eye landmarks and calculate EAR
                    left_eye, right_eye = extractor.landmark_detector.get_eye_landmarks(landmarks)
                    left_ear = extractor.eye_analyzer.calculate_eye_aspect_ratio(left_eye[:6])
                    right_ear = extractor.eye_analyzer.calculate_eye_aspect_ratio(right_eye[:6])
                    
                    avg_ear = (left_ear + right_ear) / 2
                    ear_history.append(avg_ear)
                    
                    # Keep only last 30 values for display
                    if len(ear_history) > 30:
                        ear_history.pop(0)
                    
                    # Visualize blink detection
                    vis_frame = visualize_blink_detection(frame, landmarks, left_ear, right_ear)
                    
                    # Add EAR graph
                    if len(ear_history) > 1:
                        graph_height = 100
                        graph_width = min(300, len(ear_history) * 10)
                        graph_y = frame.shape[0] - graph_height - 20
                        
                        # Create mini graph
                        max_ear = max(ear_history) if ear_history else 0.4
                        min_ear = min(ear_history) if ear_history else 0.1
                        ear_range = max_ear - min_ear if max_ear > min_ear else 0.1
                        
                        for i in range(1, len(ear_history)):
                            x1 = int((i-1) * graph_width / len(ear_history)) + 20
                            y1 = int(graph_y + graph_height - (ear_history[i-1] - min_ear) / ear_range * graph_height)
                            x2 = int(i * graph_width / len(ear_history)) + 20
                            y2 = int(graph_y + graph_height - (ear_history[i] - min_ear) / ear_range * graph_height)
                            
                            cv2.line(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        
                        # Draw blink threshold line
                        if 0.25 >= min_ear and 0.25 <= max_ear:
                            threshold_y = int(graph_y + graph_height - (0.25 - min_ear) / ear_range * graph_height)
                            cv2.line(vis_frame, (20, threshold_y), (20 + graph_width, threshold_y), (0, 0, 255), 1)
                    
                    cv2.imshow('Deepfake Detection - Real-time Analysis', vis_frame)
                else:
                    cv2.putText(frame, "No landmarks detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Deepfake Detection - Real-time Analysis', frame)
            else:
                cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Deepfake Detection - Real-time Analysis', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def batch_predict(videos_dir: str, model_path: str, output_file: str = "predictions.json"):
    """Predict multiple videos in a directory"""
    print(f"Batch prediction for videos in: {videos_dir}")
    
    if not os.path.exists(videos_dir):
        raise FileNotFoundError(f"Directory not found: {videos_dir}")
    
    results = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    # Get all video files
    video_files = []
    for file in os.listdir(videos_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(videos_dir, file))
    
    if not video_files:
        print("No video files found in the directory!")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        try:
            print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
            prediction, confidence, features = predict_single_video(video_path, model_path)
            
            result = {
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'prediction': int(prediction),
                'prediction_label': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': float(confidence),
                'key_features': {
                    'blink_rate': features.get('avg_blink_rate', 0),
                    'avg_blink_duration': features.get('avg_avg_blink_duration', 0),
                    'video_duration': features.get('video_duration', 0),
                    'total_frames': features.get('total_frames', 0)
                }
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(video_path)}: {e}")
            results.append({
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'error': str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Batch prediction completed!")
    print(f"📄 Results saved to: {output_file}")
    
    # Print summary
    successful_predictions = [r for r in results if 'error' not in r]
    if successful_predictions:
        fake_count = sum(1 for r in successful_predictions if r['prediction'] == 1)
        real_count = len(successful_predictions) - fake_count
        print(f"\n📊 SUMMARY:")
        print(f"  • Total videos processed: {len(successful_predictions)}")
        print(f"  • Predicted as REAL: {real_count}")
        print(f"  • Predicted as FAKE: {fake_count}")
        print(f"  • Errors: {len(results) - len(successful_predictions)}")

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Prediction')
    parser.add_argument('--video', type=str, help='Path to single video file')
    parser.add_argument('--batch-dir', type=str, help='Directory containing multiple videos')
    parser.add_argument('--model', type=str, default='models/classifier.pkl',
                       help='Path to trained model file')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed HTML report')
    parser.add_argument('--visualize', action='store_true',
                       help='Show real-time visualization')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for batch predictions')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print("Please train a model first using: python train.py --real-dir data/real_videos --fake-dir data/fake_videos")
        sys.exit(1)
    
    try:
        if args.video:
            # Single video prediction
            if not os.path.exists(args.video):
                print(f"❌ Error: Video file not found: {args.video}")
                sys.exit(1)
            
            predict_single_video(args.video, args.model, args.save_report, args.visualize)
            
        elif args.batch_dir:
            # Batch prediction
            batch_predict(args.batch_dir, args.model, args.output)
            
        else:
            print("❌ Error: Please provide either --video or --batch-dir")
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()