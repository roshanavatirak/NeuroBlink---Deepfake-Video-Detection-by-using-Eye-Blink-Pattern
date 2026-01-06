#!/usr/bin/env python3
"""
Comprehensive evaluation script for deepfake detection model
Calculates F1, Precision, Recall, Accuracy and other advanced metrics
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extractor import FeatureExtractor
from src.classifier import DeepfakeClassifier

def load_ground_truth_labels(labels_file: str) -> dict:
    """
    Load ground truth labels from JSON file
    Expected format: {"video_name.mp4": 0/1, ...} where 0=real, 1=fake
    """
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Labels file {labels_file} not found")
        return {}

def evaluate_on_dataset(model_path: str, dataset_dir: str, labels_file: str = None,
                       save_results: bool = True, plot_results: bool = True) -> dict:
    """
    Comprehensive evaluation of the model on a dataset
    """
    print("=" * 60)
    print("DEEPFAKE DETECTION MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    classifier = DeepfakeClassifier()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    classifier.load_model(model_path)
    
    # Load ground truth labels if available
    ground_truth = {}
    if labels_file:
        ground_truth = load_ground_truth_labels(labels_file)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    if os.path.isdir(dataset_dir):
        for file in os.listdir(dataset_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(dataset_dir, file))
    else:
        # Single video file
        video_files = [dataset_dir]
    
    if not video_files:
        raise ValueError(f"No video files found in {dataset_dir}")
    
    print(f"Found {len(video_files)} video files for evaluation")
    
    # Process videos and make predictions
    predictions = []
    confidences = []
    true_labels = []
    video_names = []
    detailed_results = []
    
    print("\nProcessing videos...")
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"[{i}/{len(video_files)}] Processing: {video_name}")
        
        try:
            # Extract features
            features = extractor.extract_video_features(video_path)
            
            # Make prediction
            prediction, confidence, detailed_metrics = classifier.predict(features)
            
            predictions.append(prediction)
            confidences.append(confidence)
            video_names.append(video_name)
            
            # Get ground truth if available
            if video_name in ground_truth:
                true_labels.append(ground_truth[video_name])
            elif ground_truth:
                print(f"Warning: No ground truth label for {video_name}")
                continue
            
            # Store detailed results
            detailed_results.append({
                'video_name': video_name,
                'prediction': prediction,
                'confidence': confidence,
                'detailed_metrics': detailed_metrics,
                'features': features,
                'true_label': ground_truth.get(video_name, -1)
            })
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue
    
    # Calculate comprehensive evaluation metrics
    results = {}
    
    if true_labels and len(true_labels) == len(predictions):
        print(f"\nCalculating metrics for {len(true_labels)} samples...")
        
        y_true = np.array(true_labels)
        y_pred = np.array(predictions)
        y_conf = np.array(confidences)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # ROC AUC if we have both classes
        roc_auc = 0.5
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_conf)
            except:
                roc_auc = 0.5
        
        # Store results
        results = {
            'evaluation_summary': {
                'total_samples': len(y_true),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy,
                'roc_auc': roc_auc,
                'mcc': mcc,
                'npv': npv
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'class_distribution': {
                'real_videos': int(np.sum(y_true == 0)),
                'fake_videos': int(np.sum(y_true == 1))
            },
            'detailed_results': detailed_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Print comprehensive results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset: {dataset_dir}")
        print(f"Model: {model_path}")
        print(f"Total Samples: {len(y_true)}")
        print(f"Real Videos: {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)")
        print(f"Fake Videos: {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)")
        print("\nPERFORMANCE METRICS:")
        print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:         {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:            {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:          {f1:.4f} ({f1*100:.2f}%)")
        print(f"Specificity:       {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
        print(f"ROC-AUC:           {roc_auc:.4f}")
        print(f"Matthews CC:       {mcc:.4f}")
        print(f"NPV:               {npv:.4f} ({npv*100:.2f}%)")
        
        print("\nCONFUSION MATRIX:")
        print(f"True Negatives:  {tn:4d} (Real correctly identified)")
        print(f"False Positives: {fp:4d} (Real misclassified as Fake)")
        print(f"False Negatives: {fn:4d} (Fake misclassified as Real)")
        print(f"True Positives:  {tp:4d} (Fake correctly identified)")
        
        # Calculate and display error analysis
        if fp > 0 or fn > 0:
            print(f"\nERROR ANALYSIS:")
            print(f"Type I Error Rate (False Positive): {fp/(fp+tn)*100:.2f}%")
            print(f"Type II Error Rate (False Negative): {fn/(fn+tp)*100:.2f}%")
        
        # Plot results if requested
        if plot_results:
            create_evaluation_plots(y_true, y_pred, y_conf, results)
        
        # Cross-validation analysis
        if len(detailed_results) >= 10:
            print("\nCROSS-VALIDATION ANALYSIS:")
            perform_cross_validation_analysis(detailed_results, classifier)
    
    else:
        print("No ground truth labels available for evaluation")
        results = {
            'predictions_only': {
                'total_samples': len(predictions),
                'predicted_real': int(np.sum(np.array(predictions) == 0)),
                'predicted_fake': int(np.sum(np.array(predictions) == 1)),
                'average_confidence': float(np.mean(confidences)) if confidences else 0,
                'confidence_std': float(np.std(confidences)) if confidences else 0
            },
            'detailed_results': detailed_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        print(f"\nPREDICTION SUMMARY (No Ground Truth):")
        print(f"Total Videos: {len(predictions)}")
        print(f"Predicted Real: {np.sum(np.array(predictions) == 0)}")
        print(f"Predicted Fake: {np.sum(np.array(predictions) == 1)}")
        print(f"Average Confidence: {np.mean(confidences):.4f}")
    
    # Save results
    if save_results:
        save_evaluation_results(results, dataset_dir)
    
    return results

def create_evaluation_plots(y_true, y_pred, y_conf, results):
    """Create comprehensive evaluation plots"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Deepfake Detection Model Evaluation', fontsize=16, fontweight='bold')
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    axes[0,0].set_xticklabels(['Real', 'Fake'])
    axes[0,0].set_yticklabels(['Real', 'Fake'])
    
    # ROC Curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_conf)
        axes[0,1].plot(fpr, tpr, color='cyan', lw=2, label=f'ROC (AUC = {results["evaluation_summary"]["roc_auc"]:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'r--', lw=1)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    if len(np.unique(y_true)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_conf)
        axes[0,2].plot(recall_curve, precision_curve, color='lime', lw=2)
        axes[0,2].set_xlabel('Recall')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].set_title('Precision-Recall Curve')
        axes[0,2].grid(True, alpha=0.3)
    
    # Confidence Distribution
    axes[1,0].hist(y_conf[y_true == 0], bins=20, alpha=0.7, label='Real', color='green', density=True)
    axes[1,0].hist(y_conf[y_true == 1], bins=20, alpha=0.7, label='Fake', color='red', density=True)
    axes[1,0].set_xlabel('Confidence')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Confidence Distribution by Class')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Metrics Bar Plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    values = [
        results['evaluation_summary']['accuracy'],
        results['evaluation_summary']['precision'],
        results['evaluation_summary']['recall'],
        results['evaluation_summary']['f1_score'],
        results['evaluation_summary']['specificity']
    ]
    bars = axes[1,1].bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_title('Performance Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Prediction vs True Labels Scatter
    jitter = np.random.normal(0, 0.05, len(y_true))
    axes[1,2].scatter(y_true + jitter, y_pred + jitter, c=y_conf, cmap='viridis', alpha=0.6)
    axes[1,2].set_xlabel('True Labels')
    axes[1,2].set_ylabel('Predictions')
    axes[1,2].set_title('Predictions vs Truth (Color = Confidence)')
    axes[1,2].set_xticks([0, 1])
    axes[1,2].set_yticks([0, 1])
    axes[1,2].set_xticklabels(['Real', 'Fake'])
    axes[1,2].set_yticklabels(['Real', 'Fake'])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"evaluation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\nEvaluation plots saved to: {plot_path}")
    
    plt.show()

def perform_cross_validation_analysis(detailed_results, classifier):
    """Perform cross-validation analysis on the results"""
    try:
        # Extract features and labels
        features_list = []
        labels = []
        
        for result in detailed_results:
            if result['true_label'] != -1:  # Valid ground truth
                features_list.append(result['features'])
                labels.append(result['true_label'])
        
        if len(features_list) < 5:
            print("Insufficient data for cross-validation")
            return
        
        # Prepare features for sklearn
        X = classifier.prepare_features(features_list)
        y = np.array(labels)
        
        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': cross_val_score(classifier.model, X, y, cv=skf, scoring='accuracy'),
            'precision': cross_val_score(classifier.model, X, y, cv=skf, scoring='precision'),
            'recall': cross_val_score(classifier.model, X, y, cv=skf, scoring='recall'),
            'f1': cross_val_score(classifier.model, X, y, cv=skf, scoring='f1')
        }
        
        print("Cross-Validation Results:")
        for metric, scores in cv_scores.items():
            print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    except Exception as e:
        print(f"Cross-validation analysis failed: {e}")

def save_evaluation_results(results, dataset_dir):
    """Save evaluation results to JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = os.path.basename(dataset_dir.rstrip('/'))
    filename = f"evaluation_results_{dataset_name}_{timestamp}.json"
    
    # Create a clean version for JSON serialization
    clean_results = results.copy()
    if 'detailed_results' in clean_results:
        for result in clean_results['detailed_results']:
            # Remove large feature dictionaries to keep file size manageable
            if 'features' in result:
                result['features'] = {k: v for k, v in result['features'].items() 
                                   if k in ['avg_blink_rate', 'avg_avg_ear', 'video_quality_score', 'temporal_consistency']}
    
    with open(filename, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory or single video')
    parser.add_argument('--labels', type=str, help='Path to ground truth labels JSON file')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_on_dataset(
            model_path=args.model,
            dataset_dir=args.dataset,
            labels_file=args.labels,
            save_results=not args.no_save,
            plot_results=not args.no_plots
        )
        
        print("\nEvaluation completed successfully!")
        
        if 'evaluation_summary' in results:
            print(f"Final F1-Score: {results['evaluation_summary']['f1_score']:.4f}")
            print(f"Final Accuracy: {results['evaluation_summary']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()