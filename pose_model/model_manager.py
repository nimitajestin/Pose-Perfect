"""
Pose-Specific Model Manager for ARVR Pose Perfect
Manages loading and inference with multiple pose-specific trained models
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2

from pose_model.pose_detector import PoseDetector


class PoseSpecificModelManager:
    """Manages multiple pose-specific YOLO models"""
    
    def __init__(self, trained_models_dir: str = "trained_models",
                 base_model_path: str = "yolo11m-pose.pt"):
        """
        Initialize the model manager
        
        Args:
            trained_models_dir: Directory containing trained pose models
            base_model_path: General pose model used when a pose-specific one fails
        """
        self.trained_models_dir = Path(trained_models_dir)
        self.base_model_path = base_model_path
        self.loaded_models = {}  # Cache for loaded models
        self.pose_metadata = {}  # Metadata for each pose
        self.reference_poses = {}  # Reference pose data for scoring
        self.fallback_detector = None  # Base detector, loaded on first need
        self.unusable_models = set()  # Poses whose own model detects nothing
        
        # Load available poses
        self.load_available_poses()

    def get_fallback_detector(self) -> Optional[PoseDetector]:
        """
        Base pose detector, shared by poses whose own model cannot detect anyone.

        Some shipped pose-specific models return no detections on any image. The
        scoring only needs human keypoints - the reference comparison happens
        afterwards - so the base model keeps those poses playable.
        """
        if self.fallback_detector is None:
            try:
                self.fallback_detector = PoseDetector(self.base_model_path)
            except Exception as e:
                print(f"Could not load fallback pose model: {e}")
                return None
        return self.fallback_detector
    
    def load_available_poses(self):
        """Load metadata for all available pose models"""
        self.pose_metadata = {}
        self.reference_poses = {}
        
        if not self.trained_models_dir.exists():
            print(f"Trained models directory not found: {self.trained_models_dir}")
            return
        
        for pose_dir in self.trained_models_dir.iterdir():
            if pose_dir.is_dir():
                metadata_file = pose_dir / "metadata.json"
                model_file = pose_dir / "model.pt"
                reference_file = pose_dir / "reference_pose.json"
                
                if metadata_file.exists() and model_file.exists():
                    try:
                        # Load metadata
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        self.pose_metadata[pose_dir.name] = {
                            **metadata,
                            'model_path': str(model_file),
                            'metadata_path': str(metadata_file)
                        }
                        
                        # Load reference pose if available
                        if reference_file.exists():
                            with open(reference_file, 'r') as f:
                                reference_data = json.load(f)
                            self.reference_poses[pose_dir.name] = reference_data
                        
                        print(f"Loaded pose model: {pose_dir.name}")
                        
                    except Exception as e:
                        print(f"Error loading pose {pose_dir.name}: {e}")
        
        print(f"Found {len(self.pose_metadata)} trained pose models")
    
    def get_available_poses(self) -> List[str]:
        """Get list of available pose names"""
        return list(self.pose_metadata.keys())
    
    def get_model(self, pose_name: str) -> Optional[PoseDetector]:
        """
        Get or load a pose-specific model
        
        Args:
            pose_name: Name of the pose
            
        Returns:
            PoseDetector instance or None if not found
        """
        if pose_name not in self.pose_metadata:
            print(f"Pose '{pose_name}' not found in available models")
            return None
        
        # Check if model is already loaded
        if pose_name in self.loaded_models:
            return self.loaded_models[pose_name]
        
        # Load the model
        try:
            model_path = self.pose_metadata[pose_name]['model_path']
            detector = PoseDetector(model_path)
            self.loaded_models[pose_name] = detector
            print(f"📦 Loaded model for pose: {pose_name} on device: {detector.device}")
            return detector
            
        except Exception as e:
            print(f"Error loading model for pose '{pose_name}': {e}")
            return None
    
    def get_reference_pose(self, pose_name: str) -> Optional[Dict]:
        """
        Get reference pose data for scoring
        
        Args:
            pose_name: Name of the pose
            
        Returns:
            Reference pose data or None if not found
        """
        return self.reference_poses.get(pose_name)
    
    def detect_and_score_pose(self, image: np.ndarray, pose_name: str, 
                             conf_threshold: float = 0.5) -> Dict:
        """
        Detect poses in image using pose-specific model and score against reference
        
        Args:
            image: Input image (BGR format)
            pose_name: Name of the target pose
            conf_threshold: Detection confidence threshold
            
        Returns:
            Detection and scoring results
        """
        # Get the pose-specific model
        detector = self.get_model(pose_name)
        if detector is None:
            return {
                'error': f"Model for pose '{pose_name}' not available",
                'poses': [],
                'scores': []
            }
        
        # Get reference pose data
        reference_data = self.get_reference_pose(pose_name)
        if reference_data is None:
            return {
                'error': f"Reference data for pose '{pose_name}' not available",
                'poses': [],
                'scores': []
            }
        
        # Detect poses. A pose whose own model has already proven unusable goes
        # straight to the base model, so it costs one inference per frame, not two.
        if pose_name in self.unusable_models:
            detector = self.get_fallback_detector() or detector
            detected_poses = detector.detect_poses(image, conf_threshold=conf_threshold)
        else:
            detected_poses = detector.detect_poses(image, conf_threshold=conf_threshold)

            if not detected_poses:
                # Could just be an empty frame - but some shipped models detect
                # nobody in any image at all. Retry with the base model and, if
                # that finds someone, use it for this pose from now on.
                fallback = self.get_fallback_detector()
                if fallback is not None and fallback is not detector:
                    fallback_poses = fallback.detect_poses(image, conf_threshold=conf_threshold)
                    if fallback_poses:
                        print(f"Pose model for '{pose_name}' detected nothing; "
                              f"using the base pose model for it from now on")
                        self.unusable_models.add(pose_name)
                        detector = fallback
                        detected_poses = fallback_poses

        if not detected_poses:
            return {
                'pose_name': pose_name,
                'poses': [],
                'scores': [],
                'message': 'No poses detected'
            }
        
        # Score each detected pose
        reference_keypoints = np.array(reference_data['keypoints'])
        reference_visibility = None
        if 'visibility' in reference_data:
            reference_visibility = np.array(reference_data['visibility'])
        
        scored_poses = []
        for pose in detected_poses:
            # Normalize detected pose
            normalized_pose = detector.normalize_pose(pose['keypoints'])
            
            if normalized_pose is not None:
                # Calculate similarity
                visibility = pose['keypoints'][:, 2] if pose['keypoints'].shape[1] > 2 else None
                
                similarity = detector.calculate_pose_similarity(
                    normalized_pose, reference_keypoints, visibility, reference_visibility
                )
                
                scored_pose = {
                    'person_id': pose['person_id'],
                    'detection_confidence': pose['confidence'],
                    'pose_score': similarity['score'],
                    'pose_error': similarity['error'],
                    'visible_keypoints': similarity['visible_keypoints'],
                    'keypoints': pose['keypoints'].tolist(),
                    'bbox': pose['bbox'].tolist() if pose['bbox'] is not None else None,
                    'normalized_keypoints': normalized_pose.tolist()
                }
                scored_poses.append(scored_pose)
        
        return {
            'pose_name': pose_name,
            'poses': scored_poses,
            'num_detected': len(scored_poses),
            'reference_info': {
                'model_confidence': reference_data.get('avg_confidence', 0.0),
                'num_reference_images': reference_data.get('num_reference_images', 0)
            }
        }
    
    def get_best_pose_score(self, image: np.ndarray, pose_name: str, 
                           conf_threshold: float = 0.5) -> Dict:
        """
        Get the best pose score from an image
        
        Args:
            image: Input image (BGR format)
            pose_name: Name of the target pose
            conf_threshold: Detection confidence threshold
            
        Returns:
            Best pose score result
        """
        results = self.detect_and_score_pose(image, pose_name, conf_threshold)
        
        if 'error' in results:
            return {'pose': pose_name, 'score': 0, 'error': results['error']}
        
        if not results['poses']:
            return {'pose': pose_name, 'score': 0, 'message': 'No poses detected'}
        
        # Find the best score
        best_pose = max(results['poses'], key=lambda x: x['pose_score'])
        
        return {
            'pose': pose_name,
            'score': round(best_pose['pose_score'], 1),
            'detection_confidence': round(best_pose['detection_confidence'], 3),
            'visible_keypoints': best_pose['visible_keypoints'],
            'person_id': best_pose['person_id']
        }
    
    def unload_model(self, pose_name: str):
        """Unload a specific model to free memory"""
        if pose_name in self.loaded_models:
            del self.loaded_models[pose_name]
            print(f"Unloaded model for pose: {pose_name}")
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        self.loaded_models.clear()
        print("Unloaded all models")
    
    def force_reload_model(self, pose_name: str):
        """Force reload a specific model (useful after code updates)"""
        if pose_name in self.loaded_models:
            del self.loaded_models[pose_name]
            print(f"Force unloaded model for pose: {pose_name}")
        # The next call to get_model will create a fresh instance
        return self.get_model(pose_name)
    
    def get_model_info(self, pose_name: str) -> Optional[Dict]:
        """
        Get information about a specific pose model
        
        Args:
            pose_name: Name of the pose
            
        Returns:
            Model information or None if not found
        """
        if pose_name not in self.pose_metadata:
            return None
        
        metadata = self.pose_metadata[pose_name].copy()
        reference_data = self.reference_poses.get(pose_name)
        
        info = {
            'pose_name': pose_name,
            'model_path': metadata['model_path'],
            'training_completed': metadata.get('training_completed', False),
            'epochs_trained': metadata.get('epochs_trained', 0),
            'is_loaded': pose_name in self.loaded_models,
            'has_reference_data': reference_data is not None
        }
        
        if reference_data:
            info['reference_info'] = {
                'avg_confidence': reference_data.get('avg_confidence', 0.0),
                'num_reference_images': reference_data.get('num_reference_images', 0)
            }
        
        return info
    
    def batch_score_images(self, images: List[np.ndarray], pose_name: str, 
                          conf_threshold: float = 0.5) -> List[Dict]:
        """
        Score multiple images against a pose
        
        Args:
            images: List of input images (BGR format)
            pose_name: Name of the target pose
            conf_threshold: Detection confidence threshold
            
        Returns:
            List of scoring results for each image
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.get_best_pose_score(image, pose_name, conf_threshold)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'image_index': i,
                    'pose': pose_name,
                    'score': 0,
                    'error': str(e)
                })
        
        return results


# Global instance for easy access
_global_model_manager = None

def get_model_manager() -> PoseSpecificModelManager:
    """Get global model manager instance"""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = PoseSpecificModelManager()
    return _global_model_manager

def reload_models():
    """Reload all models (useful after training new models)"""
    global _global_model_manager
    if _global_model_manager is not None:
        _global_model_manager.unload_all_models()
        _global_model_manager.load_available_poses()
    else:
        _global_model_manager = PoseSpecificModelManager()
