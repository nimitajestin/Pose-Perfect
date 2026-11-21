"""
Folder-based training script for ARVR Pose Perfect
Trains separate YOLO models for each pose using pose-specific folders
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import cv2
import torch
from ultralytics import YOLO

from pose_model.pose_detector import PoseDetector, TargetPoseManager

try:
    from data_collection.data_collector import AnnotationHelper
except ImportError:
    # Optional: only auto-annotation needs it. Scoring and the API server do not,
    # so a checkout without the data_collection package must still be importable.
    AnnotationHelper = None


class PoseSpecificTrainer:
    """Trains separate YOLO models for each pose from folder-based datasets"""
    
    def __init__(self, base_model_path: str = "yolo11m-pose.pt"):
        """
        Initialize the pose-specific trainer
        
        Args:
            base_model_path: Path to the base YOLO11 pose model
        """
        self.base_model_path = base_model_path
        self.trained_models_dir = Path("trained_models")
        self.trained_models_dir.mkdir(exist_ok=True)
        
        # Created lazily - see the annotation_helper property below
        self._annotation_helper = None

    @property
    def annotation_helper(self):
        """Auto-annotation helper, loaded on first use."""
        if self._annotation_helper is None:
            if AnnotationHelper is None:
                raise ImportError(
                    "Auto-annotation requires the 'data_collection' package, which is "
                    "missing from this checkout. Pose scoring and the API server do not "
                    "need it - only dataset preparation does."
                )
            self._annotation_helper = AnnotationHelper()
        return self._annotation_helper

    def get_device(self) -> str:
        """
        Detect and return the appropriate device for training
        Priority: CUDA > CPU (MPS disabled for pose models due to known bug)
        
        Returns:
            Device string ('cuda:0', or 'cpu')
        """
        try:
            # Check for CUDA first
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Test if CUDA device is actually accessible
                device = torch.device('cuda:0')
                torch.zeros(1).to(device)  # Test tensor creation on GPU
                print(f"✅ Training will use CUDA: {torch.cuda.get_device_name(0)}")
                return 'cuda:0'
        except Exception as e:
            print(f"⚠️ CUDA available but not accessible: {e}")
        
        # Note: MPS disabled for pose training due to known YOLO11 pose bug
        # See: https://github.com/ultralytics/ultralytics/issues/4031
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("⚠️ MPS available but disabled for pose training due to known YOLO11 bug")
            print("   Reference: https://github.com/ultralytics/ultralytics/issues/4031")
            print("   Falling back to CPU for stability")
        
        print("ℹ️ Training will use CPU")
        return 'cpu'
        
    def discover_pose_folders(self, dataset_root: str) -> List[str]:
        """
        Discover pose folders in the dataset directory
        
        Args:
            dataset_root: Root directory containing pose folders
            
        Returns:
            List of pose folder names
        """
        dataset_path = Path(dataset_root)
        pose_folders = []
        
        if not dataset_path.exists():
            print(f"Dataset root {dataset_root} does not exist!")
            return pose_folders
            
        # Look for folders containing images
        for item in dataset_path.iterdir():
            if item.is_dir():
                # Check if folder contains images
                image_files = list(item.glob("*.jpg")) + list(item.glob("*.png")) + list(item.glob("*.jpeg"))
                if len(image_files) > 0:
                    pose_folders.append(item.name)
                    print(f"Found pose folder: {item.name} with {len(image_files)} images")
        
        return pose_folders
    
    def prepare_pose_dataset(self, pose_name: str, pose_folder_path: str, 
                           train_ratio: float = 0.8) -> str:
        """
        Prepare dataset for a specific pose
        
        Args:
            pose_name: Name of the pose
            pose_folder_path: Path to folder containing pose images
            train_ratio: Ratio of images to use for training
            
        Returns:
            Path to the prepared dataset yaml file
        """
        pose_folder = Path(pose_folder_path)
        
        # Create pose-specific dataset directory
        pose_dataset_dir = self.trained_models_dir / pose_name / "dataset"
        pose_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        train_images_dir = pose_dataset_dir / "images" / "train"
        val_images_dir = pose_dataset_dir / "images" / "val"
        train_labels_dir = pose_dataset_dir / "labels" / "train"
        val_labels_dir = pose_dataset_dir / "labels" / "val"
        
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(list(pose_folder.glob(ext)))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {pose_folder_path}")
        
        print(f"Processing {len(image_files)} images for pose: {pose_name}")
        
        # Auto-annotate images if annotations don't exist
        annotations_dir = pose_folder / "annotations"
        if not annotations_dir.exists():
            print(f"Auto-annotating images for {pose_name}...")
            self.annotation_helper.auto_annotate_directory(str(pose_folder), str(annotations_dir))
        
        # Split into train/val
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files to appropriate directories
        for image_file in train_files:
            # Copy image
            shutil.copy2(image_file, train_images_dir / image_file.name)
            
            # Copy annotation if exists
            annotation_file = annotations_dir / f"{image_file.stem}.txt"
            if annotation_file.exists():
                shutil.copy2(annotation_file, train_labels_dir / f"{image_file.stem}.txt")
        
        for image_file in val_files:
            # Copy image
            shutil.copy2(image_file, val_images_dir / image_file.name)
            
            # Copy annotation if exists
            annotation_file = annotations_dir / f"{image_file.stem}.txt"
            if annotation_file.exists():
                shutil.copy2(annotation_file, val_labels_dir / f"{image_file.stem}.txt")
        
        # Create dataset.yaml
        dataset_yaml = {
            'path': str(pose_dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'kpt_shape': [17, 3],
            'flip_idx': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
            'nc': 1,
            'names': ['person'],
            'skeleton': [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
        }
        
        yaml_path = pose_dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"Dataset prepared for {pose_name}: {len(train_files)} train, {len(val_files)} val images")
        return str(yaml_path)
    
    def train_pose_model(self, pose_name: str, dataset_yaml_path: str, 
                        epochs: int = 100, batch_size: int = 16, 
                        image_size: int = 640) -> Dict:
        """
        Train a model for a specific pose
        
        Args:
            pose_name: Name of the pose
            dataset_yaml_path: Path to dataset yaml file
            epochs: Number of training epochs
            batch_size: Training batch size
            image_size: Input image size
            
        Returns:
            Training results dictionary
        """
        print(f"Starting training for pose: {pose_name}")
        
        # Create model save directory
        model_save_dir = self.trained_models_dir / pose_name
        model_save_dir.mkdir(exist_ok=True)
        
        # Initialize model
        model = YOLO(self.base_model_path)
        
        # Detect appropriate device
        device = self.get_device()
        
        # Training arguments
        train_args = {
            'data': dataset_yaml_path,
            'epochs': epochs,
            'imgsz': image_size,
            'batch': batch_size,
            'device': device,
            'project': str(model_save_dir),
            'name': 'training',
            'save': True,
            'save_period': 10,
            'cache': False,
            'augment': True,
            'lr0': 0.01,
            'warmup_epochs': 3,
            'patience': 20,
            'amp': False,  # Disable AMP to prevent ultralytics from downloading yolo11n.pt during AMP validation
            'verbose': True,
            'exist_ok': True
        }
        
        # Train the model
        results = model.train(**train_args)
        
        # Clean up any yolo11n.pt that might have been downloaded during AMP checks
        if Path("yolo11n.pt").exists():
            Path("yolo11n.pt").unlink()
            print("🧹 Cleaned up yolo11n.pt downloaded during training")
        
        # Find the best model weights
        best_weights = model_save_dir / "training" / "weights" / "best.pt"
        final_model_path = model_save_dir / "model.pt"
        
        if best_weights.exists():
            shutil.copy2(best_weights, final_model_path)
            print(f"Model saved to: {final_model_path}")
        else:
            print(f"Warning: Best weights not found at {best_weights}")
        
        # Save training metadata
        metadata = {
            'pose_name': pose_name,
            'model_path': str(final_model_path),
            'dataset_path': dataset_yaml_path,
            'training_args': train_args,
            'training_completed': True,
            'epochs_trained': epochs,
            'best_weights_path': str(best_weights) if best_weights.exists() else None
        }
        
        metadata_path = model_save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training completed for {pose_name}")
        return {
            'pose_name': pose_name,
            'model_path': str(final_model_path),
            'metadata_path': str(metadata_path),
            'results': results
        }
    
    def extract_reference_pose(self, pose_name: str, model_path: str, 
                             sample_images_dir: str) -> Dict:
        """
        Extract reference pose data from trained model and sample images
        
        Args:
            pose_name: Name of the pose
            model_path: Path to trained model
            sample_images_dir: Directory with sample images for this pose
            
        Returns:
            Reference pose data
        """
        # Load the trained model
        detector = PoseDetector(model_path)
        
        # Process sample images to get reference pose
        sample_dir = Path(sample_images_dir)
        image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        
        if not image_files:
            print(f"No sample images found for {pose_name}")
            return None
        
        # Take first few images as reference
        reference_poses = []
        for img_file in image_files[:5]:  # Use first 5 images
            image = cv2.imread(str(img_file))
            if image is not None:
                poses = detector.detect_poses(image, conf_threshold=0.3)
                if poses:
                    # Take the best pose
                    best_pose = max(poses, key=lambda x: x['confidence'])
                    normalized_pose = detector.normalize_pose(best_pose['keypoints'])
                    if normalized_pose is not None:
                        reference_poses.append({
                            'keypoints': normalized_pose.tolist(),
                            'confidence': best_pose['confidence'],
                            'source_image': img_file.name
                        })
        
        if not reference_poses:
            print(f"Could not extract reference pose for {pose_name}")
            return None
        
        # Average the reference poses
        avg_keypoints = np.mean([np.array(pose['keypoints'], dtype=np.float32) for pose in reference_poses], axis=0)
        
        reference_data = {
            'pose_name': pose_name,
            'keypoints': avg_keypoints.astype(np.float32).tolist(),
            'visibility': [1.0] * 17,  # Assume all keypoints are visible
            'num_reference_images': len(reference_poses),
            'avg_confidence': np.mean([pose['confidence'] for pose in reference_poses]),
            'model_path': model_path
        }
        
        # Save reference pose
        model_dir = Path(model_path).parent
        reference_path = model_dir / "reference_pose.json"
        with open(reference_path, 'w') as f:
            json.dump(reference_data, f, indent=2)
        
        print(f"Reference pose extracted for {pose_name}: {reference_path}")
        return reference_data
    
    def train_all_poses(self, dataset_root: str, epochs: int = 100, 
                       batch_size: int = 16, train_ratio: float = 0.8) -> Dict:
        """
        Train models for all poses found in the dataset root
        
        Args:
            dataset_root: Root directory containing pose folders
            epochs: Number of training epochs per pose
            batch_size: Training batch size
            train_ratio: Train/validation split ratio
            
        Returns:
            Dictionary with training results for all poses
        """
        print(f"Starting folder-based training from: {dataset_root}")
        
        # Discover pose folders
        pose_folders = self.discover_pose_folders(dataset_root)
        
        if not pose_folders:
            print("No pose folders found!")
            return {}
        
        print(f"Found {len(pose_folders)} pose folders: {pose_folders}")
        
        results = {}
        
        for pose_name in pose_folders:
            try:
                print(f"\\n{'='*50}")
                print(f"Training pose: {pose_name}")
                print(f"{'='*50}")
                
                pose_folder_path = os.path.join(dataset_root, pose_name)
                
                # Prepare dataset
                dataset_yaml = self.prepare_pose_dataset(
                    pose_name, pose_folder_path, train_ratio
                )
                
                # Train model
                train_result = self.train_pose_model(
                    pose_name, dataset_yaml, epochs, batch_size
                )
                
                # Extract reference pose
                reference_data = self.extract_reference_pose(
                    pose_name, train_result['model_path'], pose_folder_path
                )
                
                results[pose_name] = {
                    'success': True,
                    'model_path': train_result['model_path'],
                    'metadata_path': train_result['metadata_path'],
                    'reference_data': reference_data
                }
                
                print(f"✓ Successfully trained {pose_name}")
                
            except Exception as e:
                print(f"✗ Error training {pose_name}: {str(e)}")
                results[pose_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Save overall training summary
        summary_path = self.trained_models_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n{'='*50}")
        print("TRAINING SUMMARY")
        print(f"{'='*50}")
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"Total poses: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Summary saved to: {summary_path}")
        
        return results


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train pose-specific YOLO models")
    parser.add_argument('dataset_root', help='Root directory containing pose folders')
    parser.add_argument('--base-model', default='yolo11m-pose.pt', help='Base YOLO model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per pose')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/val split ratio')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = PoseSpecificTrainer(args.base_model)
    
    # Train all poses
    results = trainer.train_all_poses(
        args.dataset_root, 
        args.epochs, 
        args.batch_size, 
        args.train_ratio
    )
    
    return results


if __name__ == "__main__":
    main()
