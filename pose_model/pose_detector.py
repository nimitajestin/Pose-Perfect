"""
YOLO11 Pose Detection Model
Handles model loading, inference, and pose scoring
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional
import json
import os
from pathlib import Path

class PoseDetector:
    def __init__(self, model_path: str = "yolo11m-pose.pt", device: str = "auto"):
        """
        Initialize the pose detector
        
        Args:
            model_path: Path to YOLO11 pose model
            device: Device to run inference on ('auto', 'cpu', 'cuda:0', 'mps', etc.)
        """
        self.device = device
        if device == "auto":
            self.device = self._detect_best_device()
        
        print(f"🔧 PoseDetector using device: {self.device}")
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # COCO-17 keypoint names
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Keypoint weights for scoring (prioritize core body parts)
        self.keypoint_weights = np.array([
            1.5,  # nose
            1.0, 1.0,  # eyes
            0.8, 0.8,  # ears
            2.0, 2.0,  # shoulders
            1.5, 1.5,  # elbows
            1.2, 1.2,  # wrists
            2.5, 2.5,  # hips
            2.0, 2.0,  # knees
            1.5, 1.5   # ankles
        ], dtype=np.float32)
    
    def _detect_best_device(self) -> str:
        """
        Detect and return the best available device for inference
        Priority: CUDA > CPU (MPS disabled for pose models due to known bug)
        
        Returns:
            Device string ('cuda', or 'cpu')
        """
        try:
            # Check for CUDA first
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Test if CUDA device is accessible
                test_tensor = torch.zeros(1).to('cuda:0')
                print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
                return 'cuda'
        except Exception as e:
            print(f"⚠️ CUDA available but not accessible: {e}")
        
        # Note: MPS disabled for pose models due to known YOLO11 pose bug
        # See: https://github.com/ultralytics/ultralytics/issues/4031
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("⚠️ MPS available but disabled for pose models due to known YOLO11 bug")
            print("   Falling back to CPU for stability")
        
        print("ℹ️ Using CPU for pose detection")
        return 'cpu'
        
    def detect_poses(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect poses in an image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detected poses with keypoints and metadata
        """
        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        poses = []
        if results and len(results) > 0:
            result = results[0]
            
            # Extract keypoints and boxes
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()  # Shape: (n_persons, 17, 3)
                boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else None
                
                for i in range(keypoints.shape[0]):
                    kpts = keypoints[i]  # Shape: (17, 3) - x, y, confidence
                    
                    # Get bounding box if available
                    bbox = None
                    if boxes is not None and i < len(boxes):
                        bbox = boxes[i][:4]  # x1, y1, x2, y2
                    
                    pose_data = {
                        "keypoints": kpts,
                        "bbox": bbox,
                        "person_id": i,
                        "confidence": float(np.mean(kpts[:, 2]))  # Average keypoint confidence
                    }
                    poses.append(pose_data)
        
        return poses
    
    def normalize_pose(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize pose to be scale and translation invariant
        
        Args:
            keypoints: Raw keypoints array (17, 3) or (17, 2)
            
        Returns:
            Normalized keypoints (17, 2) or None if normalization fails
        """
        if keypoints.shape[0] != 17:
            return None
            
        # Extract x, y coordinates
        kpts = keypoints[:, :2].copy()
        
        # Check visibility (if available)
        if keypoints.shape[1] > 2:
            visibility = keypoints[:, 2]
            # Only use visible keypoints for normalization
            visible_mask = visibility > 0.5
            if np.sum(visible_mask) < 5:  # Need at least 5 visible keypoints
                return None
        else:
            visible_mask = np.ones(17, dtype=bool)
        
        # Calculate root point (midpoint between hips)
        left_hip, right_hip = 11, 12
        if visible_mask[left_hip] and visible_mask[right_hip]:
            root = (kpts[left_hip] + kpts[right_hip]) / 2
        else:
            # Fallback to center of visible keypoints
            visible_kpts = kpts[visible_mask]
            if len(visible_kpts) == 0:
                return None
            root = np.mean(visible_kpts, axis=0)
        
        # Translate to root
        kpts_centered = kpts - root
        
        # Calculate scale factor (distance between shoulders or torso height)
        left_shoulder, right_shoulder = 5, 6
        scale_factor = 1.0
        
        if visible_mask[left_shoulder] and visible_mask[right_shoulder]:
            # Use shoulder distance
            scale_factor = np.linalg.norm(kpts[left_shoulder] - kpts[right_shoulder])
        else:
            # Fallback to bounding box of visible keypoints
            visible_kpts_centered = kpts_centered[visible_mask]
            if len(visible_kpts_centered) > 1:
                scale_factor = np.max(np.ptp(visible_kpts_centered, axis=0))
        
        if scale_factor < 1e-6:  # Avoid division by zero
            return None
            
        # Normalize scale
        kpts_normalized = kpts_centered / scale_factor
        
        return kpts_normalized.astype(np.float32)
    
    def calculate_pose_similarity(self, pose1: np.ndarray, pose2: np.ndarray, 
                                visibility1: Optional[np.ndarray] = None,
                                visibility2: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate similarity between two poses
        
        Args:
            pose1, pose2: Normalized keypoints (17, 2)
            visibility1, visibility2: Visibility flags (17,)
            
        Returns:
            Dictionary with similarity metrics
        """
        if pose1 is None or pose2 is None:
            return {"score": 0.0, "error": float('inf'), "visible_keypoints": 0}
        
        # Determine which keypoints to compare
        if visibility1 is not None and visibility2 is not None:
            valid_mask = (visibility1 > 0.5) & (visibility2 > 0.5)
        else:
            valid_mask = np.ones(17, dtype=bool)
        
        if np.sum(valid_mask) == 0:
            return {"score": 0.0, "error": float('inf'), "visible_keypoints": 0}
        
        # Calculate per-keypoint distances
        distances = np.linalg.norm(pose1 - pose2, axis=1)
        
        # Ensure keypoint_weights exists (fallback for backward compatibility)
        if not hasattr(self, 'keypoint_weights'):
            self.keypoint_weights = np.array([
                1.5,  # nose
                1.0, 1.0,  # eyes
                0.8, 0.8,  # ears
                2.0, 2.0,  # shoulders
                1.5, 1.5,  # elbows
                1.2, 1.2,  # wrists
                2.5, 2.5,  # hips
                2.0, 2.0,  # knees
                1.5, 1.5   # ankles
            ], dtype=np.float32)
        
        # Apply weights and visibility mask
        weighted_distances = distances * self.keypoint_weights * valid_mask
        valid_weights = self.keypoint_weights * valid_mask
        
        # Calculate weighted average error
        total_error = np.sum(weighted_distances) / np.sum(valid_weights)
        
        # Convert to score (0-100)
        # Tunable threshold - adjust based on validation data
        error_threshold = 0.3  # This needs tuning with real data
        score = max(0.0, 100.0 * (1.0 - min(total_error / error_threshold, 1.0)))
        
        return {
            "score": float(score),
            "error": float(total_error),
            "visible_keypoints": int(np.sum(valid_mask)),
            "per_keypoint_errors": distances.tolist()
        }
    
    def train_model(self, dataset_path: str, epochs: int = 80, imgsz: int = 640, 
                   batch: int = 16, save_dir: str = "runs/pose/train"):
        """
        Fine-tune the YOLO11 pose model
        
        Args:
            dataset_path: Path to dataset.yaml
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            save_dir: Directory to save results
        """
        print(f"Starting training with dataset: {dataset_path}")
        
        # Training arguments
        args = {
            'data': dataset_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': self.device,
            'project': save_dir,
            'name': 'pose_perfect',
            'save': True,
            'save_period': 10,
            'cache': False,
            'augment': True,
            'lr0': 0.01,
            'warmup_epochs': 3,
            'patience': 15,
            'amp': True,  # Automatic Mixed Precision
            'verbose': True
        }
        
        # Start training
        results = self.model.train(**args)
        
        print(f"Training completed. Results saved to: {results.save_dir}")
        return results
    
    def load_custom_model(self, model_path: str):
        """Load a custom trained model"""
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"🔧 Loaded custom model from: {model_path} on device: {self.device}")


class TargetPoseManager:
    """Manages target poses for comparison"""
    
    def __init__(self, poses_dir: str = "target_poses"):
        self.poses_dir = Path(poses_dir)
        self.poses_dir.mkdir(exist_ok=True)
        self.target_poses = {}
        self.load_target_poses()
    
    def add_target_pose(self, pose_name: str, keypoints: np.ndarray, 
                       visibility: Optional[np.ndarray] = None):
        """Add a new target pose"""
        # Normalize the pose
        detector = PoseDetector()
        normalized_kpts = detector.normalize_pose(keypoints)
        
        if normalized_kpts is not None:
            pose_data = {
                "keypoints": normalized_kpts.tolist(),
                "visibility": visibility.tolist() if visibility is not None else None,
                "name": pose_name
            }
            
            # Save to file
            pose_file = self.poses_dir / f"{pose_name}.json"
            with open(pose_file, 'w') as f:
                json.dump(pose_data, f, indent=2)
            
            self.target_poses[pose_name] = pose_data
            print(f"Added target pose: {pose_name}")
        else:
            print(f"Failed to normalize pose: {pose_name}")
    
    def load_target_poses(self):
        """Load all target poses from files"""
        self.target_poses = {}
        
        for pose_file in self.poses_dir.glob("*.json"):
            try:
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                pose_name = pose_file.stem
                self.target_poses[pose_name] = pose_data
            except Exception as e:
                print(f"Error loading pose {pose_file}: {e}")
    
    def get_target_pose(self, pose_name: str) -> Optional[Dict]:
        """Get a target pose by name"""
        return self.target_poses.get(pose_name)
    
    def list_poses(self) -> List[str]:
        """List all available target poses"""
        return list(self.target_poses.keys())
    
    def remove_pose(self, pose_name: str):
        """Remove a target pose"""
        if pose_name in self.target_poses:
            pose_file = self.poses_dir / f"{pose_name}.json"
            if pose_file.exists():
                pose_file.unlink()
            del self.target_poses[pose_name]
            print(f"Removed target pose: {pose_name}")
