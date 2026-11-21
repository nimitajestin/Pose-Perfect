"""
FastAPI Server for Pose Perfect
Provides REST API for pose detection and scoring with pose-specific models
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from PIL import Image
import base64
from typing import List, Dict, Optional
import json
import uvicorn
from pydantic import BaseModel
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our pose detection modules
from pose_model.pose_detector import PoseDetector, TargetPoseManager
from pose_model.model_manager import PoseSpecificModelManager, get_model_manager
from pose_specific_trainer import PoseSpecificTrainer

app = FastAPI(title="Pose Perfect API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://localhost:3000",  # React dev server (alternative)
        "http://127.0.0.1:3000",  # Alternative localhost
    ],
    # Any localhost port, so the UI still reaches the API when 5173 is taken
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global instances
pose_detector = None
target_pose_manager = None
model_manager = None

class PoseScore(BaseModel):
    person_id: int
    score: float
    error: float
    visible_keypoints: int
    confidence: float
    keypoints: List[List[float]]
    bbox: Optional[List[float]] = None

class InferenceResponse(BaseModel):
    poses: List[PoseScore]
    target_pose: str
    processing_time_ms: float

class TargetPose(BaseModel):
    name: str
    keypoints: List[List[float]]
    visibility: Optional[List[float]] = None

class Base64ScoreRequest(BaseModel):
    image_base64: str
    pose_name: str
    conf_threshold: Optional[float] = 0.5

class ScoreResponse(BaseModel):
    pose_name: str
    similarity_score: float
    processing_time_ms: float
    status: str = "success"

def adjust_score_for_overfitting(score: float) -> float:
    """
    Adjust low scores to compensate for dataset overfitting.
    Adds 50 to scores between 1-30 to make them more reasonable for users.
    
    Args:
        score: Original similarity score (0-100)
        
    Returns:
        Adjusted score (0-100, capped at 100)
    """
    if 1 <= score <= 30:
        adjusted_score = score + 50
        # Cap at 100 to maintain valid score range
        return min(adjusted_score, 100.0)
    else:
        # Keep scores above 30 as they are (including 0 which indicates no detection)
        return score

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global pose_detector, target_pose_manager, model_manager
    
    print("Initializing Pose Perfect API...")
    
    # Initialize pose-specific model manager
    model_manager = get_model_manager()
    
    # Initialize fallback pose detector for general use
    model_path = "yolo11m-pose.pt"  # Will download if not present
    pose_detector = PoseDetector(model_path=model_path)
    
    # Initialize target pose manager for legacy compatibility
    target_pose_manager = TargetPoseManager("target_poses")
    
    available_poses = model_manager.get_available_poses()
    print(f"API initialized successfully! Available pose models: {available_poses}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Pose Perfect API is running!", "status": "healthy"}

@app.get("/poses")
async def list_target_poses():
    """List all available target poses (both legacy and pose-specific models)"""
    try:
        # Get legacy poses
        legacy_poses = target_pose_manager.list_poses()
        
        # Get pose-specific models
        specific_poses = model_manager.get_available_poses()
        
        return {
            "legacy_poses": legacy_poses,
            "pose_specific_models": specific_poses,
            "total_count": len(legacy_poses) + len(specific_poses)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing poses: {str(e)}")

@app.get("/poses/models")
async def list_pose_models():
    """List all available pose-specific models with details"""
    try:
        available_poses = model_manager.get_available_poses()
        models_info = {}
        
        for pose_name in available_poses:
            models_info[pose_name] = model_manager.get_model_info(pose_name)
        
        return {
            "pose_models": models_info,
            "count": len(available_poses)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing pose models: {str(e)}")

@app.post("/poses")
async def add_target_pose(pose: TargetPose):
    """Add a new target pose"""
    try:
        keypoints = np.array(pose.keypoints)
        visibility = np.array(pose.visibility) if pose.visibility else None
        
        target_pose_manager.add_target_pose(
            pose_name=pose.name,
            keypoints=keypoints,
            visibility=visibility
        )
        
        return {"message": f"Target pose '{pose.name}' added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding pose: {str(e)}")

@app.delete("/poses/{pose_name}")
async def delete_target_pose(pose_name: str):
    """Delete a target pose"""
    try:
        if pose_name not in target_pose_manager.list_poses():
            raise HTTPException(status_code=404, detail=f"Pose '{pose_name}' not found")
        
        target_pose_manager.remove_pose(pose_name)
        return {"message": f"Target pose '{pose_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting pose: {str(e)}")

@app.post("/score")
async def score_pose(
    pose_name: str = Form(...),
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5)
):
    """
    Score a pose using pose-specific model (NEW MAIN ENDPOINT)
    
    Args:
        pose_name: Name of the target pose to score against
        file: Image file containing the pose to score
        conf_threshold: Confidence threshold for pose detection
        
    Returns:
        JSON with pose name and score (0-100)
    """
    import time
    start_time = time.time()
    
    try:
        # Check if pose-specific model exists
        available_poses = model_manager.get_available_poses()
        if pose_name not in available_poses:
            raise HTTPException(
                status_code=404,
                detail=f"Pose model '{pose_name}' not found. Available poses: {available_poses}"
            )
        
        # Read and decode image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Score the pose using pose-specific model
        result = model_manager.get_best_pose_score(image_cv, pose_name, conf_threshold)
        
        # Apply score adjustment for overfitting compensation
        if 'score' in result:
            result['score'] = adjust_score_for_overfitting(result['score'])
        
        processing_time = (time.time() - start_time) * 1000
        result['processing_time_ms'] = round(processing_time, 2)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during pose scoring: {str(e)}")

@app.post("/score/base64", response_model=ScoreResponse)
async def score_pose_base64(request: Base64ScoreRequest):
    """
    Score a pose using base64 encoded image (for webcam integration)
    
    Args:
        request: JSON payload with image_base64, pose_name, and optional conf_threshold
        
    Returns:
        JSON with pose name, similarity score, and processing time
    """
    import time
    start_time = time.time()
    
    try:
        # Check if pose-specific model exists
        available_poses = model_manager.get_available_poses()
        if request.pose_name not in available_poses:
            raise HTTPException(
                status_code=404,
                detail=f"Pose model '{request.pose_name}' not found. Available poses: {available_poses}"
            )
        
        # Decode base64 image
        try:
            # Handle data URL format (data:image/jpeg;base64,...)
            if ',' in request.image_base64:
                base64_data = request.image_base64.split(',')[1]
            else:
                base64_data = request.image_base64
                
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
        
        # Score the pose using pose-specific model
        result = model_manager.get_best_pose_score(image_cv, request.pose_name, request.conf_threshold)
        
        # Apply score adjustment for overfitting compensation
        raw_score = result.get('score', 0.0)
        adjusted_score = adjust_score_for_overfitting(raw_score)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ScoreResponse(
            pose_name=request.pose_name,
            similarity_score=adjusted_score,
            processing_time_ms=round(processing_time, 2),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during pose scoring: {str(e)}")

@app.post("/score/batch")
async def score_pose_batch(
    pose_name: str = Form(...),
    files: List[UploadFile] = File(...),
    conf_threshold: float = Form(0.5)
):
    """
    Score multiple images against a pose using pose-specific model
    
    Args:
        pose_name: Name of the target pose to score against
        files: List of image files
        conf_threshold: Confidence threshold for pose detection
        
    Returns:
        JSON with results for each image
    """
    import time
    start_time = time.time()
    
    try:
        # Check if pose-specific model exists
        available_poses = model_manager.get_available_poses()
        if pose_name not in available_poses:
            raise HTTPException(
                status_code=404,
                detail=f"Pose model '{pose_name}' not found. Available poses: {available_poses}"
            )
        
        # Process all images
        images = []
        filenames = []
        
        for file in files:
            try:
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes))
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                images.append(image_cv)
                filenames.append(file.filename)
            except Exception as e:
                # Add error placeholder
                images.append(None)
                filenames.append(file.filename)
        
        # Score all images
        results = []
        for i, (image, filename) in enumerate(zip(images, filenames)):
            if image is not None:
                try:
                    score_result = model_manager.get_best_pose_score(image, pose_name, conf_threshold)
                    
                    # Apply score adjustment for overfitting compensation
                    if 'score' in score_result:
                        score_result['score'] = adjust_score_for_overfitting(score_result['score'])
                    
                    score_result['filename'] = filename
                    score_result['image_index'] = i
                    results.append(score_result)
                except Exception as e:
                    results.append({
                        'filename': filename,
                        'image_index': i,
                        'pose': pose_name,
                        'score': 0,
                        'error': str(e)
                    })
            else:
                results.append({
                    'filename': filename,
                    'image_index': i,
                    'pose': pose_name,
                    'score': 0,
                    'error': 'Failed to decode image'
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'pose_name': pose_name,
            'total_images': len(files),
            'processing_time_ms': round(processing_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch pose scoring: {str(e)}")

@app.post("/infer", response_model=InferenceResponse)
async def infer_pose(
    target_pose_name: str = Form(...),
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5)
):
    """
    Detect poses in an image and compare with target pose
    
    Args:
        target_pose_name: Name of the target pose to compare against
        file: Image file
        conf_threshold: Confidence threshold for pose detection
    """
    import time
    start_time = time.time()
    
    try:
        # Validate target pose
        target_pose_data = target_pose_manager.get_target_pose(target_pose_name)
        if not target_pose_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Target pose '{target_pose_name}' not found"
            )
        
        # Read and decode image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect poses
        detected_poses = pose_detector.detect_poses(image_cv, conf_threshold=conf_threshold)
        
        # Process each detected pose
        pose_scores = []
        target_keypoints = np.array(target_pose_data["keypoints"])
        target_visibility = None
        if target_pose_data.get("visibility"):
            target_visibility = np.array(target_pose_data["visibility"])
        
        for pose in detected_poses:
            # Normalize detected pose
            normalized_pose = pose_detector.normalize_pose(pose["keypoints"])
            
            if normalized_pose is not None:
                # Calculate similarity
                visibility = pose["keypoints"][:, 2] if pose["keypoints"].shape[1] > 2 else None
                
                similarity = pose_detector.calculate_pose_similarity(
                    normalized_pose, target_keypoints, visibility, target_visibility
                )
                
                # Apply score adjustment for overfitting compensation
                adjusted_score = adjust_score_for_overfitting(similarity["score"])
                
                pose_score = PoseScore(
                    person_id=pose["person_id"],
                    score=adjusted_score,
                    error=similarity["error"],
                    visible_keypoints=similarity["visible_keypoints"],
                    confidence=pose["confidence"],
                    keypoints=pose["keypoints"].tolist(),
                    bbox=pose["bbox"].tolist() if pose["bbox"] is not None else None
                )
                pose_scores.append(pose_score)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return InferenceResponse(
            poses=pose_scores,
            target_pose=target_pose_name,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

@app.post("/infer/batch")
async def infer_pose_batch(
    target_pose_name: str = Form(...),
    files: List[UploadFile] = File(...),
    conf_threshold: float = Form(0.5)
):
    """
    Batch inference for multiple images
    """
    import time
    start_time = time.time()
    
    try:
        # Validate target pose
        target_pose_data = target_pose_manager.get_target_pose(target_pose_name)
        if not target_pose_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Target pose '{target_pose_name}' not found"
            )
        
        results = []
        target_keypoints = np.array(target_pose_data["keypoints"])
        target_visibility = None
        if target_pose_data.get("visibility"):
            target_visibility = np.array(target_pose_data["visibility"])
        
        for i, file in enumerate(files):
            try:
                # Read and decode image
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes))
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect poses
                detected_poses = pose_detector.detect_poses(image_cv, conf_threshold=conf_threshold)
                
                # Process poses
                pose_scores = []
                for pose in detected_poses:
                    normalized_pose = pose_detector.normalize_pose(pose["keypoints"])
                    
                    if normalized_pose is not None:
                        visibility = pose["keypoints"][:, 2] if pose["keypoints"].shape[1] > 2 else None
                        
                        similarity = pose_detector.calculate_pose_similarity(
                            normalized_pose, target_keypoints, visibility, target_visibility
                        )
                        
                        # Apply score adjustment for overfitting compensation
                        adjusted_score = adjust_score_for_overfitting(similarity["score"])
                        
                        pose_score = PoseScore(
                            person_id=pose["person_id"],
                            score=adjusted_score,
                            error=similarity["error"],
                            visible_keypoints=similarity["visible_keypoints"],
                            confidence=pose["confidence"],
                            keypoints=pose["keypoints"].tolist(),
                            bbox=pose["bbox"].tolist() if pose["bbox"] is not None else None
                        )
                        pose_scores.append(pose_score)
                
                results.append({
                    "image_index": i,
                    "filename": file.filename,
                    "poses": [pose.dict() for pose in pose_scores]
                })
                
            except Exception as e:
                results.append({
                    "image_index": i,
                    "filename": file.filename,
                    "error": str(e),
                    "poses": []
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "target_pose": target_pose_name,
            "total_images": len(files),
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch inference: {str(e)}")

@app.post("/capture_target_pose")
async def capture_target_pose(
    pose_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Capture a new target pose from an image
    """
    try:
        # Read and decode image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect poses
        detected_poses = pose_detector.detect_poses(image_cv, conf_threshold=0.3)
        
        if not detected_poses:
            raise HTTPException(status_code=400, detail="No poses detected in the image")
        
        if len(detected_poses) > 1:
            # Use the pose with highest confidence
            best_pose = max(detected_poses, key=lambda x: x["confidence"])
        else:
            best_pose = detected_poses[0]
        
        # Add as target pose
        keypoints = best_pose["keypoints"]
        visibility = keypoints[:, 2] if keypoints.shape[1] > 2 else None
        
        target_pose_manager.add_target_pose(
            pose_name=pose_name,
            keypoints=keypoints,
            visibility=visibility
        )
        
        return {
            "message": f"Target pose '{pose_name}' captured successfully",
            "confidence": best_pose["confidence"],
            "keypoints_count": len(keypoints)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error capturing target pose: {str(e)}")

@app.post("/train")
async def train_model(
    epochs: int = Form(80),
    batch_size: int = Form(16),
    image_size: int = Form(640)
):
    """
    Start training the legacy combined pose model
    """
    try:
        dataset_path = "dataset/dataset.yaml"
        
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=400, 
                detail="Dataset not found. Please prepare training data first."
            )
        
        # Start training (this will run in background)
        results = pose_detector.train_model(
            dataset_path=dataset_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size
        )
        
        return {
            "message": "Training completed successfully",
            "results_path": str(results.save_dir)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/train/pose-specific")
async def train_pose_specific_models(
    dataset_root: str = Form(...),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    train_ratio: float = Form(0.8),
    base_model: str = Form("yolo11m-pose.pt")
):
    """
    Train pose-specific models from folder-based dataset
    
    Args:
        dataset_root: Root directory containing pose folders (e.g., "dataset" containing "Tpose/", "WarriorPose/" etc.)
        epochs: Number of training epochs per pose
        batch_size: Training batch size
        train_ratio: Train/validation split ratio
        base_model: Base YOLO model to fine-tune
    """
    try:
        if not os.path.exists(dataset_root):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset root directory not found: {dataset_root}"
            )
        
        # Initialize trainer
        trainer = PoseSpecificTrainer(base_model)
        
        # Start training (this will take a while)
        results = trainer.train_all_poses(
            dataset_root=dataset_root,
            epochs=epochs,
            batch_size=batch_size,
            train_ratio=train_ratio
        )
        
        # Reload model manager to pick up new models
        model_manager.unload_all_models()
        model_manager.load_available_poses()
        
        # Count successful trainings
        successful = sum(1 for r in results.values() if r.get('success', False))
        failed = len(results) - successful
        
        return {
            "message": f"Pose-specific training completed",
            "total_poses": len(results),
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during pose-specific training: {str(e)}")

@app.post("/train/single-pose")
async def train_single_pose(
    pose_name: str = Form(...),
    pose_folder_path: str = Form(...),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    train_ratio: float = Form(0.8),
    base_model: str = Form("yolo11m-pose.pt")
):
    """
    Train a model for a single specific pose
    
    Args:
        pose_name: Name of the pose
        pose_folder_path: Path to folder containing images for this pose
        epochs: Number of training epochs
        batch_size: Training batch size
        train_ratio: Train/validation split ratio
        base_model: Base YOLO model to fine-tune
    """
    try:
        if not os.path.exists(pose_folder_path):
            raise HTTPException(
                status_code=400,
                detail=f"Pose folder not found: {pose_folder_path}"
            )
        
        # Initialize trainer
        trainer = PoseSpecificTrainer(base_model)
        
        # Prepare dataset
        dataset_yaml = trainer.prepare_pose_dataset(pose_name, pose_folder_path, train_ratio)
        
        # Train model
        train_result = trainer.train_pose_model(pose_name, dataset_yaml, epochs, batch_size)
        
        # Extract reference pose
        reference_data = trainer.extract_reference_pose(
            pose_name, train_result['model_path'], pose_folder_path
        )
        
        # Reload model manager
        model_manager.unload_all_models()
        model_manager.load_available_poses()
        
        return {
            "message": f"Training completed for pose: {pose_name}",
            "model_path": train_result['model_path'],
            "metadata_path": train_result['metadata_path'],
            "has_reference_data": reference_data is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training pose '{pose_name}': {str(e)}")

@app.post("/load_model")
async def load_custom_model(model_path: str = Form(...)):
    """
    Load a custom trained model
    """
    try:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        pose_detector.load_custom_model(model_path)
        
        return {"message": f"Model loaded successfully from {model_path}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/models/reload")
async def reload_models():
    """
    Reload all pose-specific models (useful after training new models)
    """
    try:
        # Reload model manager
        model_manager.unload_all_models()
        model_manager.load_available_poses()
        
        available_poses = model_manager.get_available_poses()
        
        return {
            "message": "Models reloaded successfully",
            "available_poses": available_poses,
            "count": len(available_poses)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

@app.get("/models/{pose_name}/info")
async def get_model_info(pose_name: str):
    """
    Get detailed information about a specific pose model
    """
    try:
        info = model_manager.get_model_info(pose_name)
        
        if info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Pose model '{pose_name}' not found"
            )
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.delete("/models/{pose_name}")
async def delete_pose_model(pose_name: str):
    """
    Delete a pose-specific model and its associated files
    """
    try:
        import shutil
        
        # Check if model exists
        if pose_name not in model_manager.get_available_poses():
            raise HTTPException(
                status_code=404,
                detail=f"Pose model '{pose_name}' not found"
            )
        
        # Unload model if loaded
        model_manager.unload_model(pose_name)
        
        # Delete model directory
        model_dir = Path("trained_models") / pose_name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Reload model manager
        model_manager.load_available_poses()
        
        return {"message": f"Pose model '{pose_name}' deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
