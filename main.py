"""
Main entry point for Pose Perfect
Provides command-line interface for all operations
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def collect_data(args):
    """Start data collection"""
    from data_collection.data_collector import DataCollector
    
    collector = DataCollector(args.output_dir)
    
    if args.video:
        collector.extract_frames_from_video(
            args.video, args.pose_name, args.frame_interval
        )
    else:
        collector.start_camera_collection(args.pose_name, args.camera_id)

def annotate_data(args):
    """Auto-annotate collected data"""
    from data_collection.data_collector import AnnotationHelper
    
    annotator = AnnotationHelper()
    annotator.auto_annotate_directory(args.image_dir, args.output_dir)

def split_dataset(args):
    """Split data into train/val sets"""
    from data_collection.data_collector import DatasetSplitter
    
    splitter = DatasetSplitter(args.dataset_root)
    splitter.split_data(args.source_dir, args.train_ratio, args.copy_files)

def train_model(args):
    """Train the pose model (legacy combined model)"""
    from pose_model.pose_detector import PoseDetector
    
    detector = PoseDetector(args.base_model)
    results = detector.train_model(
        dataset_path=args.dataset_config,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size
    )
    
    print(f"Training completed. Results saved to: {results.save_dir}")

def train_pose_camera(args):
    """Train pose-specific model using camera capture"""
    try:
        from pose_camera_trainer import PoseCameraTrainer
    except ImportError as e:
        print(f"Error importing pose_camera_trainer: {e}")
        return
    
    trainer = PoseCameraTrainer(args.base_model)
    
    # Run camera-based training
    result = trainer.train_pose_from_camera(
        pose_name=args.pose_name,
        camera_id=args.camera_id,
        frames_needed=args.frames_needed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio
    )
    
    # Cleanup temporary files
    trainer.cleanup_temp_files()
    
    if result['success']:
        print(f"✓ Camera training completed successfully for {args.pose_name}!")
        print(f"Model saved to: {result['model_path']}")
    else:
        print(f"✗ Camera training failed: {result.get('error', 'Unknown error')}")

def train_pose_specific(args):
    """Train pose-specific models from folder-based dataset"""
    try:
        from pose_specific_trainer import PoseSpecificTrainer
    except ImportError as e:
        print(f"Error importing pose_specific_trainer: {e}")
        return
    
    trainer = PoseSpecificTrainer(args.base_model)
    
    if args.single_pose:
        # Train single pose
        if not args.pose_name or not args.pose_folder:
            print("Error: --pose-name and --pose-folder required for single pose training")
            return
        
        try:
            # Prepare dataset
            dataset_yaml = trainer.prepare_pose_dataset(
                args.pose_name, args.pose_folder, args.train_ratio
            )
            
            # Train model
            train_result = trainer.train_pose_model(
                args.pose_name, dataset_yaml, args.epochs, args.batch_size
            )
            
            # Extract reference pose
            reference_data = trainer.extract_reference_pose(
                args.pose_name, train_result['model_path'], args.pose_folder
            )
            
            print(f"Single pose training completed for: {args.pose_name}")
            print(f"Model saved to: {train_result['model_path']}")
            
        except Exception as e:
            print(f"Error training single pose: {e}")
    else:
        # Train all poses in dataset root
        results = trainer.train_all_poses(
            args.dataset_root, 
            args.epochs, 
            args.batch_size, 
            args.train_ratio
        )
        
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"Pose-specific training completed: {successful}/{len(results)} successful")

def evaluate_model(args):
    """Evaluate model performance"""
    from utils.pose_utils import evaluate_model_performance
    
    results = evaluate_model_performance(
        args.model_path, args.test_images, args.target_poses
    )
    
    if args.output_file:
        results.to_csv(args.output_file, index=False)
        print(f"Results saved to: {args.output_file}")

def create_demo(args):
    """Create demo video"""
    from utils.pose_utils import create_demo_video
    
    create_demo_video(
        args.input_video, args.output_video, 
        args.target_pose, args.model_path
    )

def start_api(args):
    """Start the FastAPI server"""
    import uvicorn
    from api.api_server import app
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

def test_inference(args):
    """Test inference on a single image (legacy method)"""
    import cv2
    import numpy as np
    from pose_model.pose_detector import PoseDetector, TargetPoseManager
    from utils.pose_utils import visualize_pose
    
    # Initialize components
    detector = PoseDetector(args.model_path)
    pose_manager = TargetPoseManager(args.target_poses_dir)
    
    # Load image
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image {args.image_path}")
        return
    
    # Detect poses
    poses = detector.detect_poses(image, conf_threshold=args.conf_threshold)
    
    if not poses:
        print("No poses detected in the image")
        return
    
    print(f"Detected {len(poses)} pose(s)")
    
    # Compare with target pose if specified
    if args.target_pose:
        target_pose_data = pose_manager.get_target_pose(args.target_pose)
        if target_pose_data:
            target_keypoints = np.array(target_pose_data["keypoints"])
            
            for i, pose in enumerate(poses):
                normalized_pose = detector.normalize_pose(pose["keypoints"])
                if normalized_pose is not None:
                    similarity = detector.calculate_pose_similarity(
                        normalized_pose, target_keypoints
                    )
                    print(f"Pose {i+1} similarity to '{args.target_pose}': {similarity['score']:.1f}%")
        else:
            print(f"Target pose '{args.target_pose}' not found")
    
    # Visualize results
    if args.visualize:
        for i, pose in enumerate(poses):
            vis_image = visualize_pose(
                image, pose["keypoints"], pose["bbox"], 
                score=similarity.get("score") if 'similarity' in locals() else None
            )
            
            cv2.imshow(f'Pose {i+1}', vis_image)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save results if specified
    if args.output_path:
        best_pose = max(poses, key=lambda x: x["confidence"])
        vis_image = visualize_pose(
            image, best_pose["keypoints"], best_pose["bbox"]
        )
        cv2.imwrite(args.output_path, vis_image)
        print(f"Result saved to: {args.output_path}")

def test_pose_scoring(args):
    """Test pose scoring using pose-specific models"""
    import cv2
    from pose_model.model_manager import get_model_manager
    from utils.pose_utils import visualize_pose
    
    # Initialize model manager
    model_manager = get_model_manager()
    available_poses = model_manager.get_available_poses()
    
    if not available_poses:
        print("No pose-specific models found. Please train some models first.")
        return
    
    if args.pose_name and args.pose_name not in available_poses:
        print(f"Pose '{args.pose_name}' not found. Available poses: {available_poses}")
        return
    
    # Load image
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image {args.image_path}")
        return
    
    # Test specific pose or all poses
    poses_to_test = [args.pose_name] if args.pose_name else available_poses
    
    print(f"Testing image against {len(poses_to_test)} pose model(s)...")
    print("-" * 50)
    
    results = []
    for pose_name in poses_to_test:
        try:
            result = model_manager.get_best_pose_score(
                image, pose_name, args.conf_threshold
            )
            results.append(result)
            
            status = "✓" if result['score'] > 0 else "✗"
            print(f"{status} {pose_name}: {result['score']:.1f}")
            
            if 'detection_confidence' in result:
                print(f"   Detection confidence: {result['detection_confidence']:.3f}")
            if 'visible_keypoints' in result:
                print(f"   Visible keypoints: {result['visible_keypoints']}/17")
                
        except Exception as e:
            print(f"✗ {pose_name}: Error - {e}")
    
    # Find best match
    if results:
        best_result = max(results, key=lambda x: x.get('score', 0))
        print("-" * 50)
        print(f"Best match: {best_result['pose']} ({best_result['score']:.1f})")
    
    # Visualize if requested
    if args.visualize and results:
        # Use the original detector for visualization
        from pose_model.pose_detector import PoseDetector
        detector = PoseDetector()
        poses = detector.detect_poses(image, args.conf_threshold)
        
        if poses:
            best_pose = max(poses, key=lambda x: x["confidence"])
            vis_image = visualize_pose(
                image, best_pose["keypoints"], best_pose["bbox"],
                score=best_result.get('score')
            )
            
            cv2.imshow(f'Best Match: {best_result.get("pose", "Unknown")}', vis_image)
            print("Press any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Save results if specified
    if args.output_path and results:
        with open(args.output_path, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Pose Perfect - YOLO11 Pose Detection and Scoring")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Collect pose data')
    collect_parser.add_argument('pose_name', help='Name of the pose to collect')
    collect_parser.add_argument('--output-dir', default='collected_data', help='Output directory')
    collect_parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    collect_parser.add_argument('--video', help='Video file to extract frames from')
    collect_parser.add_argument('--frame-interval', type=int, default=30, help='Frame extraction interval')
    collect_parser.set_defaults(func=collect_data)
    
    # Annotation command
    annotate_parser = subparsers.add_parser('annotate', help='Auto-annotate collected data')
    annotate_parser.add_argument('image_dir', help='Directory containing images')
    annotate_parser.add_argument('--output-dir', help='Output directory for annotations')
    annotate_parser.set_defaults(func=annotate_data)
    
    # Dataset splitting command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val')
    split_parser.add_argument('source_dir', help='Source directory with collected data')
    split_parser.add_argument('--dataset-root', default='dataset', help='Dataset root directory')
    split_parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    split_parser.add_argument('--copy-files', action='store_true', help='Copy files instead of moving')
    split_parser.set_defaults(func=split_dataset)
    
    # Training command (legacy)
    train_parser = subparsers.add_parser('train', help='Train the combined pose model (legacy)')
    train_parser.add_argument('--dataset-config', default='dataset/dataset.yaml', help='Dataset config file')
    train_parser.add_argument('--base-model', default='yolo11m-pose.pt', help='Base model to fine-tune')
    train_parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--image-size', type=int, default=640, help='Input image size')
    train_parser.set_defaults(func=train_model)
    
    # Pose-specific training command
    pose_train_parser = subparsers.add_parser('train-poses', help='Train pose-specific models from folders')
    pose_train_parser.add_argument('dataset_root', nargs='?', help='Root directory containing pose folders')
    pose_train_parser.add_argument('--base-model', default='yolo11m-pose.pt', help='Base model to fine-tune')
    pose_train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs per pose')
    pose_train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    pose_train_parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/validation split ratio')
    pose_train_parser.add_argument('--single-pose', action='store_true', help='Train single pose instead of all poses')
    pose_train_parser.add_argument('--pose-name', help='Name of pose to train (for single pose mode)')
    pose_train_parser.add_argument('--pose-folder', help='Folder containing pose images (for single pose mode)')
    pose_train_parser.set_defaults(func=train_pose_specific)
    
    # Camera-based pose training command
    camera_train_parser = subparsers.add_parser('train-camera', help='Train pose-specific model using camera capture')
    camera_train_parser.add_argument('pose_name', help='Name of pose to train')
    camera_train_parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    camera_train_parser.add_argument('--frames-needed', type=int, default=100, help='Number of frames to capture')
    camera_train_parser.add_argument('--base-model', default='yolo11m-pose.pt', help='Base model to fine-tune')
    camera_train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    camera_train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    camera_train_parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/validation split ratio')
    camera_train_parser.set_defaults(func=train_pose_camera)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('model_path', help='Path to trained model')
    eval_parser.add_argument('test_images', help='Directory with test images')
    eval_parser.add_argument('--target-poses', default='target_poses', help='Target poses directory')
    eval_parser.add_argument('--output-file', help='Output CSV file for results')
    eval_parser.set_defaults(func=evaluate_model)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Create demo video')
    demo_parser.add_argument('input_video', help='Input video file')
    demo_parser.add_argument('output_video', help='Output video file')
    demo_parser.add_argument('target_pose', help='Target pose name')
    demo_parser.add_argument('--model-path', default='yolo11m-pose.pt', help='Model to use')
    demo_parser.set_defaults(func=create_demo)
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start FastAPI server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host address')
    api_parser.add_argument('--port', type=int, default=8000, help='Port number')
    api_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    api_parser.add_argument('--log-level', default='info', help='Log level')
    api_parser.set_defaults(func=start_api)
    
    # Test inference command (legacy)
    test_parser = subparsers.add_parser('test', help='Test inference on image (legacy)')
    test_parser.add_argument('image_path', help='Path to test image')
    test_parser.add_argument('--model-path', default='yolo11m-pose.pt', help='Model to use')
    test_parser.add_argument('--target-pose', help='Target pose to compare against')
    test_parser.add_argument('--target-poses-dir', default='target_poses', help='Target poses directory')
    test_parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    test_parser.add_argument('--visualize', action='store_true', help='Show visualization')
    test_parser.add_argument('--output-path', help='Save result image')
    test_parser.set_defaults(func=test_inference)
    
    # Test pose scoring command
    score_parser = subparsers.add_parser('score', help='Test pose scoring with pose-specific models')
    score_parser.add_argument('image_path', help='Path to test image')
    score_parser.add_argument('--pose-name', help='Specific pose to test (if not provided, tests all)')
    score_parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    score_parser.add_argument('--visualize', action='store_true', help='Show visualization')
    score_parser.add_argument('--output-path', help='Save results as JSON')
    score_parser.set_defaults(func=test_pose_scoring)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Call the appropriate function
    args.func(args)

if __name__ == "__main__":
    main()
