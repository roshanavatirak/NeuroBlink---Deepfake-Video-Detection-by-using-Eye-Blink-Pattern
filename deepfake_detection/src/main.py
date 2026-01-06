# #!/usr/bin/env python3
# """
# Main module for deepfake detection using eye blink analysis
# """

# import os
# import sys
# import argparse
# from typing import List, Dict
# from .feature_extractor import FeatureExtractor
# from .classifier import DeepfakeClassifier

# def process_dataset(real_videos_dir: str, fake_videos_dir: str) -> tuple:
#     """
#     Process dataset and extract features
#     """
#     extractor = FeatureExtractor()
    
#     features_list = []
#     labels = []
    
#     print("Processing real videos...")
#     # Process real videos
#     if os.path.exists(real_videos_dir):
#         for filename in os.listdir(real_videos_dir):
#             if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#                 video_path = os.path.join(real_videos_dir, filename)
#                 try:
#                     features = extractor.extract_video_features(video_path)
#                     features_list.append(features)
#                     labels.append(0)  # Real video
#                     print(f"Processed real video: {filename}")
#                 except Exception as e:
#                     print(f"Error processing {filename}: {e}")
    
#     print("Processing fake videos...")
#     # Process fake videos
#     if os.path.exists(fake_videos_dir):
#         for filename in os.listdir(fake_videos_dir):
#             if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#                 video_path = os.path.join(fake_videos_dir, filename)
#                 try:
#                     features = extractor.extract_video_features(video_path)
#                     features_list.append(features)
#                     labels.append(1)  # Fake video
#                     print(f"Processed fake video: {filename}")
#                 except Exception as e:
#                     print(f"Error processing {filename}: {e}")
    
#     return features_list, labels

# def train_model(real_videos_dir: str, fake_videos_dir: str, model_path: str = "models/classifier.pkl"):
#     """Train the deepfake detection model"""
#     print("Starting training process...")
    
#     # Extract features from dataset
#     features_list, labels = process_dataset(real_videos_dir, fake_videos_dir)
    
#     if len(features_list) == 0:
#         print("No videos found in the specified directories!")
#         return
    
#     print(f"Found {len(features_list)} videos total")
    
#     # Train classifier
#     classifier = DeepfakeClassifier(model_type='random_forest')
#     results = classifier.train(features_list, labels)
    
#     # Save model
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     classifier.save_model(model_path)
    
#     print("\nTraining Results:")
#     print(f"Accuracy: {results['accuracy']:.4f}")
#     print(f"Cross-validation: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    
#     # Print feature importance
#     importance = classifier.get_feature_importance()
#     if importance:
#         print("\nTop 10 Most Important Features:")
#         for i, (feature, score) in enumerate(list(importance.items())[:10]):
#             print(f"{i+1}. {feature}: {score:.4f}")

# def predict_video(video_path: str, model_path: str = "models/classifier.pkl"):
#     """Predict if a video is fake or real"""
#     print(f"Analyzing video: {video_path}")
    
#     # Load model
#     classifier = DeepfakeClassifier()
#     classifier.load_model(model_path)
    
#     # Extract features
#     extractor = FeatureExtractor()
#     features = extractor.extract_video_features(video_path)
    
#     # Make prediction
#     prediction, confidence = classifier.predict(features)
    
#     result = "FAKE" if prediction == 1 else "REAL"
#     print(f"\nPrediction: {result}")
#     print(f"Confidence: {confidence:.4f}")
    
#     return prediction, confidence

# def main():
#     parser = argparse.ArgumentParser(description='Deepfake Detection using Eye Blink Analysis')
#     parser.add_argument('--mode', choices=['train', 'predict'], required=True,
#                        help='Mode: train or predict')
#     parser.add_argument('--real-dir', type=str, help='Directory containing real videos')
#     parser.add_argument('--fake-dir', type=str, help='Directory containing fake videos')
#     parser.add_argument('--video', type=str, help='Video file to analyze')
#     parser.add_argument('--model', type=str, default='models/classifier.pkl',
#                        help='Model file path')
    
#     args = parser.parse_args()
    
#     if args.mode == 'train':
#         if not args.real_dir or not args.fake_dir:
#             print("Error: --real-dir and --fake-dir are required for training")
#             sys.exit(1)
#         train_model(args.real_dir, args.fake_dir, args.model)
    
#     elif args.mode == 'predict':
#         if not args.video:
#             print("Error: --video is required for prediction")
#             sys.exit(1)
#         predict_video(args.video, args.model)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Main module for deepfake detection using eye blink analysis with caching
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Dict
from feature_extractor import FeatureExtractor
from classifier import DeepfakeClassifier


def process_dataset(real_videos_dir: str, fake_videos_dir: str, cache_dir="features") -> tuple:
    """
    Process dataset and extract features incrementally with caching (.npz files).
    """
    extractor = FeatureExtractor()
    os.makedirs(cache_dir, exist_ok=True)

    features_list = []
    labels = []

    def process_video(video_path: str, label: int):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cache_file = os.path.join(cache_dir, f"{video_name}.npz")

        if os.path.exists(cache_file):
            # ✅ Load cached features
            try:
                data = np.load(cache_file, allow_pickle=True)
                features = data["features"].item()  # stored as dict
                print(f"✔ Loaded cached features for {video_name}")
            except Exception as e:
                print(f"⚠ Error loading cache for {video_name}, re-extracting: {e}")
                features = extractor.extract_video_features(video_path)
                np.savez_compressed(cache_file, features=features)
        else:
            # ❌ Not cached → extract now
            try:
                features = extractor.extract_video_features(video_path)
                np.savez_compressed(cache_file, features=features)
                print(f"➡ Extracted and saved features for {video_name}")
            except Exception as e:
                print(f"❌ Error processing {video_name}: {e}")
                return None

        features_list.append(features)
        labels.append(label)

    print("Processing real videos...")
    if os.path.exists(real_videos_dir):
        for filename in os.listdir(real_videos_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                process_video(os.path.join(real_videos_dir, filename), 0)

    print("Processing fake videos...")
    if os.path.exists(fake_videos_dir):
        for filename in os.listdir(fake_videos_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                process_video(os.path.join(fake_videos_dir, filename), 1)

    return features_list, labels


def train_model(real_videos_dir: str, fake_videos_dir: str, model_path: str = "models/classifier.pkl"):
    """Train the deepfake detection model"""
    print("🚀 Starting training process...")

    # Extract features from dataset
    features_list, labels = process_dataset(real_videos_dir, fake_videos_dir)

    if len(features_list) == 0:
        print("⚠ No videos found in the specified directories!")
        return

    print(f"📊 Found {len(features_list)} videos total")

    # Train classifier
    classifier = DeepfakeClassifier(model_type='random_forest')
    results = classifier.train(features_list, labels)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)

    print("\n✅ Training Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")

    # Print feature importance
    importance = classifier.get_feature_importance()
    if importance:
        print("\n🏆 Top 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"{i+1}. {feature}: {score:.4f}")


def predict_video(video_path: str, model_path: str = "models/classifier.pkl", cache_dir="features"):
    """Predict if a video is fake or real (with caching)"""
    print(f"🔍 Analyzing video: {video_path}")

    # Load model
    classifier = DeepfakeClassifier()
    classifier.load_model(model_path)

    # Extract or load cached features
    extractor = FeatureExtractor()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_file = os.path.join(cache_dir, f"{video_name}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        features = data["features"].item()
        print(f"✔ Loaded cached features for {video_name}")
    else:
        features = extractor.extract_video_features(video_path)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_file, features=features)
        print(f"➡ Extracted and saved features for {video_name}")

    # Make prediction
    prediction, confidence = classifier.predict(features)

    result = "FAKE" if prediction == 1 else "REAL"
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.4f}")

    return prediction, confidence


def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection using Eye Blink Analysis (with caching)')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='Mode: train or predict')
    parser.add_argument('--real-dir', type=str, help='Directory containing real videos')
    parser.add_argument('--fake-dir', type=str, help='Directory containing fake videos')
    parser.add_argument('--video', type=str, help='Video file to analyze')
    parser.add_argument('--model', type=str, default='models/classifier.pkl',
                        help='Model file path')
    parser.add_argument('--cache-dir', type=str, default='features',
                        help='Directory to store cached features (.npz)')

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.real_dir or not args.fake_dir:
            print("❌ Error: --real-dir and --fake-dir are required for training")
            sys.exit(1)
        train_model(args.real_dir, args.fake_dir, args.model)

    elif args.mode == 'predict':
        if not args.video:
            print("❌ Error: --video is required for prediction")
            sys.exit(1)
        predict_video(args.video, args.model, args.cache_dir)


if __name__ == "__main__":
    main()
