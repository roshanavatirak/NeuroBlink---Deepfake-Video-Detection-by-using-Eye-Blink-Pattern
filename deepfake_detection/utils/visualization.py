# utils/visualization.py - COMPLETE FILE

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os
import time

def visualize_landmarks(frame: np.ndarray, landmarks: np.ndarray, 
                       eye_indices: List[int] = None) -> np.ndarray:
    """
    Visualize facial landmarks on frame
    """
    vis_frame = frame.copy()
    
    # Draw all landmarks
    for point in landmarks:
        cv2.circle(vis_frame, tuple(point.astype(int)), 1, (0, 255, 0), -1)
    
    # Highlight eye landmarks if specified
    if eye_indices:
        for idx in eye_indices:
            if idx < len(landmarks):
                point = landmarks[idx]
                cv2.circle(vis_frame, tuple(point.astype(int)), 2, (0, 0, 255), -1)
    
    return vis_frame

def plot_ear_sequence(ear_sequence: List[float], timestamps: List[float], 
                     title: str = "Eye Aspect Ratio Over Time", save_path: str = None):
    """Plot EAR sequence over time"""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, ear_sequence, 'b-', linewidth=1.5, alpha=0.8)
    plt.axhline(y=0.25, color='r', linestyle='--', alpha=0.7, label='Blink Threshold')
    plt.fill_between(timestamps, ear_sequence, 0.25, 
                     where=(np.array(ear_sequence) < 0.25), 
                     color='red', alpha=0.3, label='Blinks')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Eye Aspect Ratio (EAR)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_comparison(real_features: List[Dict], fake_features: List[Dict], 
                          feature_name: str, save_path: str = None):
    """Compare feature distributions between real and fake videos"""
    real_values = [f[feature_name] for f in real_features if feature_name in f and f[feature_name] is not None]
    fake_values = [f[feature_name] for f in fake_features if feature_name in f and f[feature_name] is not None]
    
    if not real_values or not fake_values:
        print(f"No data found for feature: {feature_name}")
        return
    
    plt.figure(figsize=(14, 6))
    
    # Distribution plot
    plt.subplot(1, 3, 1)
    plt.hist(real_values, bins=20, alpha=0.7, label='Real', color='blue', density=True)
    plt.hist(fake_values, bins=20, alpha=0.7, label='Fake', color='red', density=True)
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.title(f'{feature_name} Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 3, 2)
    data = [real_values, fake_values]
    labels = ['Real', 'Fake']
    box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel(feature_name)
    plt.title(f'{feature_name} Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Violin plot
    plt.subplot(1, 3, 3)
    try:
        import pandas as pd
        df_real = pd.DataFrame({'value': real_values, 'type': 'Real'})
        df_fake = pd.DataFrame({'value': fake_values, 'type': 'Fake'})
        df_combined = pd.concat([df_real, df_fake])
        
        sns.violinplot(data=df_combined, x='type', y='value', palette=['lightblue', 'lightcoral'])
        plt.title(f'{feature_name} Violin Plot')
        plt.ylabel(feature_name)
        plt.grid(True, alpha=0.3)
    except ImportError:
        # If pandas not available, create simple violin plot
        plt.hist([real_values, fake_values], bins=20, label=['Real', 'Fake'], 
                alpha=0.7, color=['blue', 'red'])
        plt.title(f'{feature_name} Histogram')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_detection_report(video_path: str, prediction: int, confidence: float,
                          features: Dict, output_dir: str = "reports") -> str:
    """Create a detailed detection report"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    report_path = os.path.join(output_dir, f"{video_name}_report.html")
    
    result = "FAKE" if prediction == 1 else "REAL"
    color = "red" if prediction == 1 else "green"
    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
    
    # Calculate risk assessment
    risk_level = "HIGH" if prediction == 1 and confidence > 0.8 else \
                "MEDIUM" if prediction == 1 and confidence > 0.6 else \
                "LOW" if prediction == 0 and confidence > 0.8 else "UNCERTAIN"
    
    risk_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green", "UNCERTAIN": "gray"}[risk_level]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepfake Detection Report - {video_name}</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px; 
                text-align: center;
            }}
            .result {{ 
                font-size: 36px; 
                font-weight: bold; 
                color: {color};
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .confidence {{ 
                font-size: 20px; 
                margin-top: 15px;
                color: {confidence_color};
            }}
            .risk-badge {{
                display: inline-block;
                padding: 8px 16px;
                background-color: {risk_color};
                color: white;
                border-radius: 20px;
                font-weight: bold;
                margin-top: 10px;
            }}
            .content {{
                padding: 30px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                border-left: 4px solid #667eea;
                background-color: #f8f9ff;
            }}
            .feature-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .feature-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .feature-name {{
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }}
            .feature-value {{
                font-size: 18px;
                color: #667eea;
                font-weight: bold;
            }}
            .analysis-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }}
            .analysis-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }}
            .indicator {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .good {{ background-color: #4CAF50; }}
            .warning {{ background-color: #FF9800; }}
            .bad {{ background-color: #F44336; }}
            h1, h2, h3 {{ margin-top: 0; }}
            .timestamp {{
                color: #888;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Deepfake Detection Report</h1>
                <p><strong>Video:</strong> {video_name}</p>
                <p class="result">{result}</p>
                <p class="confidence">Confidence: {confidence:.1%}</p>
                <div class="risk-badge">Risk Level: {risk_level}</div>
                <p class="timestamp">Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>🎯 Detection Summary</h2>
                    <div class="analysis-grid">
                        <div class="analysis-card">
                            <h3>Prediction Details</h3>
                            <p><span class="indicator {'good' if prediction == 0 else 'bad'}"></span>
                               Classification: <strong>{result}</strong></p>
                            <p><span class="indicator {confidence_color.replace('orange', 'warning').replace('red', 'bad').replace('green', 'good')}"></span>
                               Confidence: <strong>{confidence:.1%}</strong></p>
                            <p><span class="indicator {risk_color.replace('orange', 'warning').replace('red', 'bad').replace('green', 'good').replace('gray', 'warning')}"></span>
                               Risk Level: <strong>{risk_level}</strong></p>
                        </div>
                        <div class="analysis-card">
                            <h3>Video Information</h3>
                            <p>Duration: <strong>{features.get('video_duration', 0):.2f}s</strong></p>
                            <p>Total Frames: <strong>{features.get('total_frames', 0):,}</strong></p>
                            <p>Frame Rate: <strong>{features.get('fps', 0):.1f} FPS</strong></p>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>👁️ Eye Blink Analysis</h2>
                    <div class="feature-grid">
                        <div class="feature-card">
                            <div class="feature-name">Blink Rate</div>
                            <div class="feature-value">{features.get('avg_blink_rate', 0):.3f} blinks/sec</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">Average Blink Duration</div>
                            <div class="feature-value">{features.get('avg_avg_blink_duration', 0):.3f}s</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">Blink Cycle Time</div>
                            <div class="feature-value">{features.get('avg_avg_blink_cycle', 0):.3f}s</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">Average EAR</div>
                            <div class="feature-value">{features.get('avg_avg_ear', 0):.4f}</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">EAR Standard Deviation</div>
                            <div class="feature-value">{features.get('avg_std_ear', 0):.4f}</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">Blink Completeness</div>
                            <div class="feature-value">{features.get('avg_avg_blink_completeness', 0):.4f}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path

def visualize_blink_detection(frame: np.ndarray, landmarks: np.ndarray, 
                            ear_left: float, ear_right: float, 
                            blink_threshold: float = 0.25) -> np.ndarray:
    """
    Visualize blink detection on frame
    """
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Draw eye landmarks
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Draw left eye contour
    left_points = []
    for idx in left_eye_indices[:12]:
        if idx < len(landmarks):
            point = tuple(landmarks[idx].astype(int))
            left_points.append(point)
            cv2.circle(vis_frame, point, 2, (0, 255, 0), -1)
    
    # Draw right eye contour
    right_points = []
    for idx in right_eye_indices[:12]:
        if idx < len(landmarks):
            point = tuple(landmarks[idx].astype(int))
            right_points.append(point)
            cv2.circle(vis_frame, point, 2, (0, 255, 0), -1)
    
    # Draw eye contours
    if len(left_points) > 3:
        left_contour = np.array(left_points)
        cv2.polylines(vis_frame, [left_contour], True, (0, 255, 0), 2)
    
    if len(right_points) > 3:
        right_contour = np.array(right_points)
        cv2.polylines(vis_frame, [right_contour], True, (0, 255, 0), 2)
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Background rectangle for text
    cv2.rectangle(vis_frame, (10, 10), (400, 100), (0, 0, 0), -1)
    cv2.rectangle(vis_frame, (10, 10), (400, 100), (255, 255, 255), 2)
    
    # Left eye status
    left_status = "BLINK" if ear_left < blink_threshold else "OPEN"
    left_color = (0, 0, 255) if ear_left < blink_threshold else (0, 255, 0)
    cv2.putText(vis_frame, f"Left Eye:  {ear_left:.3f} ({left_status})", 
                (20, 35), font, font_scale, left_color, thickness)
    
    # Right eye status
    right_status = "BLINK" if ear_right < blink_threshold else "OPEN"
    right_color = (0, 0, 255) if ear_right < blink_threshold else (0, 255, 0)
    cv2.putText(vis_frame, f"Right Eye: {ear_right:.3f} ({right_status})", 
                (20, 65), font, font_scale, right_color, thickness)
    
    # Average EAR
    avg_ear = (ear_left + ear_right) / 2
    avg_status = "BLINK" if avg_ear < blink_threshold else "OPEN"
    avg_color = (0, 0, 255) if avg_ear < blink_threshold else (0, 255, 0)
    cv2.putText(vis_frame, f"Avg EAR:   {avg_ear:.3f} ({avg_status})", 
                (20, 95), font, font_scale, avg_color, thickness)
    
    return vis_frame

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None, save_path: str = None):
    """Plot confusion matrix with improved styling"""
    if class_names is None:
        class_names = ['Real', 'Fake']
    
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation text
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Predictions'},
                square=True, linewidths=0.5)
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(importance_dict: Dict, top_n: int = 15, save_path: str = None):
    """Plot feature importance with improved styling"""
    if not importance_dict:
        print("No feature importance data available")
        return
    
    # Get top N features
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_features)
    
    # Clean up feature names for display
    display_features = []
    for f in features:
        clean_name = f.replace('avg_', '').replace('left_', 'L_').replace('right_', 'R_')
        clean_name = clean_name.replace('_', ' ').title()
        display_features.append(clean_name)
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = plt.barh(y_pos, scores, color=colors)
    
    plt.yticks(y_pos, display_features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance in Deepfake Detection', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score, save_path: str = None):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()