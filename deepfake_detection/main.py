#!/usr/bin/env python3
"""
Simple FastAPI backend for Deepfake Detection - Fixed Version
"""

import os
import sys
import uuid
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Global variables
classifier = None
feature_extractor = None
model_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global classifier, feature_extractor, model_loaded
    try:
        print("Loading ML components...")
        
        # Import here to avoid module loading issues
        from feature_extractor import FeatureExtractor
        from classifier import DeepfakeClassifier
        
        feature_extractor = FeatureExtractor()
        classifier = DeepfakeClassifier()
        
        if os.path.exists("models/classifier.pkl"):
            classifier.load_model("models/classifier.pkl")
            model_loaded = True
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found: models/classifier.pkl")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure your ML files are in the src/ folder with correct imports")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    yield
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(title="Deepfake Detection API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    processing_time: float
    video_info: dict
    features: dict = None
    error: str = None

@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection API", 
        "model_loaded": model_loaded,
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_loaded else "unhealthy", 
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_path = None
    
    try:
        # Save file
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(temp_dir, f"video{file_ext}")
        
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Process video
        import time
        start_time = time.time()
        
        features = feature_extractor.extract_video_features(temp_path)
        prediction, confidence = classifier.predict(features)
        
        processing_time = time.time() - start_time
        result = "FAKE" if prediction == 1 else "REAL"
        
        return PredictionResponse(
            success=True,
            prediction=result,
            confidence=float(confidence),
            processing_time=processing_time,
            video_info={
                "filename": file.filename,
                "duration": features.get('video_duration', 0),
                "total_frames": features.get('total_frames', 0),
                "fps": features.get('fps', 0)
            },
            features={
                "blink_rate": features.get('avg_blink_rate', 0),
                "avg_blink_duration": features.get('avg_avg_blink_duration', 0),
                "avg_ear": features.get('avg_avg_ear', 0),
                "blink_completeness": features.get('avg_avg_blink_completeness', 0)
            }
        )
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return PredictionResponse(
            success=False,
            prediction="ERROR",
            confidence=0.0,
            processing_time=0.0,
            video_info={},
            error=str(e)
        )
    finally:
        # Cleanup
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Deepfake Detection API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)