#!/usr/bin/env python3
"""
Fixed Training script for deepfake detection model
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extractor import FeatureExtractor
from src.classifier import DeepfakeClassifier
from utils.visualization import plot_confusion_matrix, plot_feature_importance, plot_roc_curve
import warnings
warnings.filterwarnings('ignore')

def validate_dataset(real_dir: str, fake_dir: str) -> Tuple[List[str], List[str]]:
    """
    Validate dataset directories and get video file lists
    """
    print("🔍 Validating dataset...")
    
    # Check directories exist
    if not os.path.exists(real_dir):
        raise FileNotFoundError(f"Real videos directory not found: {real_dir}")
    
    if not os.path.exists(fake_dir):
        raise FileNotFoundError(f"Fake videos directory not found: {fake_dir}")
    
    # Get video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    real_videos = []
    for file in os.listdir(real_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            real_videos.append(os.path.join(real_dir, file))
    
    fake_videos = []
    for file in os.listdir(fake_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            fake_videos.append(os.path.join(fake_dir, file))
    
    print(f"📊 Dataset Summary:")
    print(f"  • Real videos: {len(real_videos)}")
    print(f"  • Fake videos: {len(fake_videos)}")
    print(f"  • Total videos: {len(real_videos) + len(fake_videos)}")
    
    if len(real_videos) == 0:
        raise ValueError("No real videos found! Please add video files to the real_videos directory.")
    
    if len(fake_videos) == 0:
        raise ValueError("No fake videos found! Please add video files to the fake_videos directory.")
    
    if len(real_videos) < 2 or len(fake_videos) < 2:
        print("⚠️  Warning: Very few videos detected. For better accuracy, use at least 10-20 videos per class.")
    
    return real_videos, fake_videos

def extract_features_from_videos(video_paths: List[str], labels: List[int]) -> Tuple[List[Dict], List[int]]:
    """
    Extract features from all videos
    """
    print("🔧 Extracting features from videos...")
    
    extractor = FeatureExtractor()
    features_list = []
    valid_labels = []
    failed_videos = []
    
    total_videos = len(video_paths)
    
    for i, (video_path, label) in enumerate(zip(video_paths, labels), 1):
        try:
            print(f"[{i}/{total_videos}] Processing: {os.path.basename(video_path)}")
            
            # Extract features
            start_time = time.time()
            features = extractor.extract_video_features(video_path)
            processing_time = time.time() - start_time
            
            # Check if features are valid
            if features.get('total_frames', 0) == 0:
                print(f"  ⚠️  Skipping {os.path.basename(video_path)}: No valid frames found")
                failed_videos.append(video_path)
                continue
            
            features_list.append(features)
            valid_labels.append(label)
            
            print(f"  ✅ Success ({processing_time:.2f}s) - Frames: {features.get('total_frames', 0)}, "
                  f"Duration: {features.get('video_duration', 0):.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(video_path)}: {e}")
            failed_videos.append(video_path)
            continue
    
    print(f"\n📈 Feature Extraction Summary:")
    print(f"  • Successfully processed: {len(features_list)} videos")
    print(f"  • Failed: {len(failed_videos)} videos")
    
    if failed_videos:
        print(f"  • Failed videos: {[os.path.basename(f) for f in failed_videos]}")
    
    if len(features_list) == 0:
        raise ValueError("No features extracted! Please check your video files.")
    
    return features_list, valid_labels

def analyze_features(features_list: List[Dict], labels: List[int]) -> Dict:
    """
    Analyze extracted features
    """
    print("📊 Analyzing extracted features...")
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    df['label'] = labels
    
    # Basic statistics
    real_features = df[df['label'] == 0]
    fake_features = df[df['label'] == 1]
    
    analysis = {
        'total_samples': len(df),
        'real_samples': len(real_features),
        'fake_samples': len(fake_features),
        'feature_count': len(df.columns) - 1,  # Exclude label column
        'feature_names': [col for col in df.columns if col != 'label']
    }
    
    # Key feature comparisons
    key_features = ['avg_blink_rate', 'avg_avg_blink_duration', 'avg_avg_ear', 'video_duration']
    
    print(f"📋 Feature Analysis:")
    print(f"  • Total features extracted: {analysis['feature_count']}")
    print(f"  • Real videos: {analysis['real_samples']}")
    print(f"  • Fake videos: {analysis['fake_samples']}")
    
    print(f"\n🔍 Key Feature Comparison (Real vs Fake):")
    for feature in key_features:
        if feature in df.columns:
            real_mean = real_features[feature].mean()
            fake_mean = fake_features[feature].mean()
            real_std = real_features[feature].std()
            fake_std = fake_features[feature].std()
            
            print(f"  • {feature}:")
            print(f"    - Real: {real_mean:.6f} ± {real_std:.6f}")
            print(f"    - Fake: {fake_mean:.6f} ± {fake_std:.6f}")
            print(f"    - Difference: {abs(real_mean - fake_mean):.6f}")
    
    return analysis

def train_and_evaluate_model(features_list: List[Dict], labels: List[int], 
                            model_type: str = 'random_forest',
                            save_path: str = 'models/classifier.pkl') -> Dict:
    """
    Train and evaluate the deepfake detection model
    """
    print(f"🤖 Training {model_type} model...")
    
    # Initialize classifier
    classifier = DeepfakeClassifier(model_type=model_type)
    
    # Train model
    start_time = time.time()
    results = classifier.train(features_list, labels)
    training_time = time.time() - start_time
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classifier.save_model(save_path)
    
    # Additional evaluation metrics
    try:
        X = classifier.prepare_features(features_list)
        X_scaled = classifier.scaler.transform(X)
        y_pred_proba = classifier.model.predict_proba(X_scaled)[:, 1]
        auc_score = roc_auc_score(labels, y_pred_proba)
        results['auc_score'] = auc_score
    except Exception as e:
        print(f"Warning: Could not calculate AUC score: {e}")
        results['auc_score'] = 0.0
    
    results['training_time'] = training_time
    results['model_type'] = model_type
    results['model_path'] = save_path
    
    return results

def save_training_report(results: Dict, analysis: Dict, output_file: str = 'training_report.json'):
    """
    Save comprehensive training report
    """
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_analysis': analysis,
        'training_results': results,
        'model_info': {
            'type': results.get('model_type', 'unknown'),
            'path': results.get('model_path', ''),
            'training_time': results.get('training_time', 0)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Training report saved: {output_file}")

def plot_training_results(results: Dict, features_list: List[Dict], labels: List[int]):
    """
    Create visualizations for training results
    """
    try:
        print("📈 Creating visualizations...")
        
        # Confusion Matrix
        cm = np.array(results['confusion_matrix'])
        plot_confusion_matrix(cm, ['Real', 'Fake'], 'training_confusion_matrix.png')
        
        # Feature Importance (if available)
        if results.get('model_type') == 'random_forest':
            try:
                classifier = DeepfakeClassifier(model_type='random_forest')
                classifier.load_model(results.get('model_path', 'models/classifier.pkl'))
                importance = classifier.get_feature_importance()
                if importance:
                    plot_feature_importance(importance, top_n=15, save_path='feature_importance.png')
            except Exception as e:
                print(f"Could not plot feature importance: {e}")
        
        # ROC Curve (if AUC available)
        if results.get('auc_score', 0) > 0:
            try:
                classifier = DeepfakeClassifier()
                classifier.load_model(results.get('model_path', 'models/classifier.pkl'))
                X = classifier.prepare_features(features_list)
                X_scaled = classifier.scaler.transform(X)
                y_pred_proba = classifier.model.predict_proba(X_scaled)[:, 1]
                
                fpr, tpr, _ = roc_curve(labels, y_pred_proba)
                plot_roc_curve(fpr, tpr, results['auc_score'], 'roc_curve.png')
            except Exception as e:
                print(f"Could not plot ROC curve: {e}")
        
        print("✅ Visualizations saved!")
        
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--real-dir', type=str, required=True,
                       help='Directory containing real videos')
    parser.add_argument('--fake-dir', type=str, required=True,
                       help='Directory containing fake videos')
    parser.add_argument('--model-path', type=str, default='models/classifier.pkl',
                       help='Path to save the trained model')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'logistic_regression'],
                       help='Type of classifier to use')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip creating visualizations')
    parser.add_argument('--output-report', type=str, default='training_report.json',
                       help='Path to save training report')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Deepfake Detection Model Training")
        print("=" * 60)
        
        # Step 1: Validate dataset
        real_videos, fake_videos = validate_dataset(args.real_dir, args.fake_dir)
        
        # Step 2: Prepare data
        all_videos = real_videos + fake_videos
        all_labels = [0] * len(real_videos) + [1] * len(fake_videos)  # 0=Real, 1=Fake
        
        # Step 3: Extract features
        features_list, valid_labels = extract_features_from_videos(all_videos, all_labels)
        
        if len(features_list) < 2:
            print("❌ Error: Need at least 2 videos for training")
            sys.exit(1)
        
        # Check if we have both classes
        unique_labels = set(valid_labels)
        if len(unique_labels) < 2:
            print("❌ Error: Need videos from both classes (real and fake)")
            sys.exit(1)
        
        # Step 4: Analyze features
        analysis = analyze_features(features_list, valid_labels)
        
        # Step 5: Train model
        results = train_and_evaluate_model(
            features_list, valid_labels, args.model_type, args.model_path
        )
        
        # Step 6: Save report
        save_training_report(results, analysis, args.output_report)
        
        # Step 7: Create visualizations
        if not args.no_visualizations:
            plot_training_results(results, features_list, valid_labels)
        
        # Final results
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"🎯 Model Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"📊 Cross-validation: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        if results.get('auc_score', 0) > 0:
            print(f"📈 AUC Score: {results['auc_score']:.4f}")
        print(f"⏱️  Training Time: {results['training_time']:.2f} seconds")
        print(f"💾 Model saved: {args.model_path}")
        print(f"📄 Report saved: {args.output_report}")
        
        print(f"\n🔥 Ready to predict! Use:")
        print(f"   python predict.py --video your_video.mp4 --model {args.model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()