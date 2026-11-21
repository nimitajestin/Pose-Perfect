"""
Utility functions for Pose Perfect
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Tuple
import os

def visualize_pose(image: np.ndarray, keypoints: np.ndarray, 
                  bbox: np.ndarray = None, score: float = None) -> np.ndarray:
    """
    Visualize pose keypoints on an image
    
    Args:
        image: Input image (BGR format)
        keypoints: Keypoints array (17, 3) - x, y, confidence
        bbox: Bounding box [x1, y1, x2, y2]
        score: Pose similarity score
        
    Returns:
        Image with pose visualization
    """
    vis_image = image.copy()
    
    # COCO-17 skeleton connections
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # Convert to 0-based indexing
    skeleton = [[s[0]-1, s[1]-1] for s in skeleton]
    
    # Define colors for different body parts
    colors = [
        (255, 0, 0),    # Head (red)
        (255, 0, 0),    # Head
        (255, 0, 0),    # Head
        (255, 0, 0),    # Head
        (255, 0, 0),    # Head
        (0, 255, 0),    # Arms (green)
        (0, 255, 0),    # Arms
        (0, 255, 0),    # Arms
        (0, 255, 0),    # Arms
        (0, 255, 0),    # Arms
        (0, 255, 0),    # Arms
        (0, 0, 255),    # Legs (blue)
        (0, 0, 255),    # Legs
        (0, 0, 255),    # Legs
        (0, 0, 255),    # Legs
        (0, 0, 255),    # Legs
        (0, 0, 255),    # Legs
    ]
    
    # Draw skeleton
    for connection in skeleton:
        kpt1_idx, kpt2_idx = connection
        if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and
            keypoints[kpt1_idx][2] > 0.5 and keypoints[kpt2_idx][2] > 0.5):
            
            pt1 = (int(keypoints[kpt1_idx][0]), int(keypoints[kpt1_idx][1]))
            pt2 = (int(keypoints[kpt2_idx][0]), int(keypoints[kpt2_idx][1]))
            cv2.line(vis_image, pt1, pt2, (0, 255, 255), 2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            center = (int(x), int(y))
            cv2.circle(vis_image, center, 4, colors[i], -1)
            cv2.circle(vis_image, center, 6, (255, 255, 255), 2)
    
    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    # Draw score
    if score is not None:
        text = f"Score: {score:.1f}"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
    
    return vis_image

def compare_poses_visual(image1: np.ndarray, keypoints1: np.ndarray,
                        image2: np.ndarray, keypoints2: np.ndarray,
                        score: float) -> np.ndarray:
    """
    Create a side-by-side comparison of two poses
    """
    # Resize images to same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    target_height = min(h1, h2)
    scale1 = target_height / h1
    scale2 = target_height / h2
    
    img1_resized = cv2.resize(image1, (int(w1 * scale1), target_height))
    img2_resized = cv2.resize(image2, (int(w2 * scale2), target_height))
    
    # Scale keypoints
    kpts1_scaled = keypoints1.copy()
    kpts1_scaled[:, 0] *= scale1
    kpts1_scaled[:, 1] *= scale1
    
    kpts2_scaled = keypoints2.copy()
    kpts2_scaled[:, 0] *= scale2
    kpts2_scaled[:, 1] *= scale2
    kpts2_scaled[:, 0] += img1_resized.shape[1]  # Offset for concatenation
    
    # Visualize poses
    vis1 = visualize_pose(img1_resized, kpts1_scaled)
    vis2 = visualize_pose(img2_resized, kpts2_scaled)
    
    # Concatenate images
    combined = np.hstack([vis1, vis2])
    
    # Add score text
    text = f"Similarity Score: {score:.1f}%"
    cv2.putText(combined, text, (10, combined.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return combined

def plot_training_metrics(results_dir: str):
    """
    Plot training metrics from YOLO results
    """
    results_dir = Path(results_dir)
    results_file = results_dir / "results.csv"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    # Load results
    import pandas as pd
    df = pd.read_csv(results_file)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot losses
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    axes[0, 0].plot(df['epoch'], df['train/pose_loss'], label='Pose Loss')
    axes[0, 0].plot(df['epoch'], df['train/kobj_loss'], label='Keypoint Obj Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot mAP
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(pose)'], label='mAP@0.5')
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(pose)'], label='mAP@0.5:0.95')
    axes[0, 1].set_title('Pose mAP')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate
    axes[1, 0].plot(df['epoch'], df['lr/pg0'], label='LR Group 0')
    axes[1, 0].plot(df['epoch'], df['lr/pg1'], label='LR Group 1')
    axes[1, 0].plot(df['epoch'], df['lr/pg2'], label='LR Group 2')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Plot validation metrics
    if 'val/box_loss' in df.columns:
        axes[1, 1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        axes[1, 1].plot(df['epoch'], df['val/pose_loss'], label='Val Pose Loss')
        axes[1, 1].plot(df['epoch'], df['val/kobj_loss'], label='Val Keypoint Loss')
        axes[1, 1].set_title('Validation Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_performance(model_path: str, test_images_dir: str, 
                             target_poses_dir: str):
    """
    Evaluate model performance on test set
    """
    from pose_model.pose_detector import PoseDetector, TargetPoseManager
    
    # Initialize model and pose manager
    detector = PoseDetector(model_path)
    pose_manager = TargetPoseManager(target_poses_dir)
    
    # Get test images
    test_images = list(Path(test_images_dir).glob("*.jpg")) + \
                 list(Path(test_images_dir).glob("*.png"))
    
    results = []
    
    for img_path in test_images:
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Detect poses
        poses = detector.detect_poses(image)
        
        # For each target pose, calculate similarity
        for pose_name in pose_manager.list_poses():
            target_pose_data = pose_manager.get_target_pose(pose_name)
            target_keypoints = np.array(target_pose_data["keypoints"])
            
            for pose in poses:
                normalized_pose = detector.normalize_pose(pose["keypoints"])
                if normalized_pose is not None:
                    similarity = detector.calculate_pose_similarity(
                        normalized_pose, target_keypoints
                    )
                    
                    results.append({
                        "image": img_path.name,
                        "target_pose": pose_name,
                        "score": similarity["score"],
                        "error": similarity["error"],
                        "confidence": pose["confidence"]
                    })
    
    # Create performance report
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("Model Performance Summary:")
    print(f"Total evaluations: {len(results)}")
    print(f"Average score: {df['score'].mean():.2f}")
    print(f"Average error: {df['error'].mean():.4f}")
    print(f"Average confidence: {df['confidence'].mean():.3f}")
    
    # Per-pose performance
    print("\nPer-pose performance:")
    pose_stats = df.groupby('target_pose').agg({
        'score': ['mean', 'std', 'count'],
        'error': 'mean'
    }).round(3)
    print(pose_stats)
    
    return df

def create_demo_video(input_video: str, output_video: str, target_pose_name: str,
                     model_path: str = "yolo11m-pose.pt"):
    """
    Create a demo video with pose detection and scoring
    """
    from pose_model.pose_detector import PoseDetector, TargetPoseManager
    
    # Initialize components
    detector = PoseDetector(model_path)
    pose_manager = TargetPoseManager("target_poses")
    
    # Get target pose
    target_pose_data = pose_manager.get_target_pose(target_pose_name)
    if not target_pose_data:
        print(f"Target pose '{target_pose_name}' not found")
        return
    
    target_keypoints = np.array(target_pose_data["keypoints"])
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video: {input_video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    scores_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect poses
        poses = detector.detect_poses(frame)
        
        # Process best pose
        best_score = 0
        best_pose = None
        
        for pose in poses:
            normalized_pose = detector.normalize_pose(pose["keypoints"])
            if normalized_pose is not None:
                similarity = detector.calculate_pose_similarity(
                    normalized_pose, target_keypoints
                )
                if similarity["score"] > best_score:
                    best_score = similarity["score"]
                    best_pose = pose
        
        # Visualize
        if best_pose is not None:
            frame = visualize_pose(frame, best_pose["keypoints"], 
                                 best_pose["bbox"], best_score)
        
        # Add target pose name
        cv2.putText(frame, f"Target: {target_pose_name}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Track scores
        scores_history.append(best_score)
        if len(scores_history) > 30:  # Keep last 30 frames
            scores_history.pop(0)
        
        # Add stability indicator
        if len(scores_history) > 10:
            stability = 1 - np.std(scores_history[-10:]) / 100
            cv2.putText(frame, f"Stability: {stability:.2f}", (10, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Demo video created: {output_video}")

def convert_coco_to_yolo(coco_annotation_file: str, output_dir: str, 
                        images_dir: str):
    """
    Convert COCO pose annotations to YOLO format
    """
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process each image
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']
        
        # Find annotations for this image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        if annotations:
            yolo_lines = []
            
            for ann in annotations:
                if 'keypoints' in ann and 'bbox' in ann:
                    # Get bounding box
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Convert to YOLO format (normalized center, width, height)
                    center_x = (x + w/2) / img_width
                    center_y = (y + h/2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Get keypoints
                    keypoints = ann['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]
                    
                    # Normalize keypoints
                    normalized_kpts = []
                    for i in range(0, len(keypoints), 3):
                        kx, ky, kv = keypoints[i], keypoints[i+1], keypoints[i+2]
                        norm_kx = kx / img_width
                        norm_ky = ky / img_height
                        normalized_kpts.extend([norm_kx, norm_ky, kv])
                    
                    # Create YOLO line
                    line = f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    for kpt_val in normalized_kpts:
                        line += f" {kpt_val:.6f}"
                    
                    yolo_lines.append(line)
            
            # Save YOLO annotation file
            if yolo_lines:
                output_file = output_dir / f"{Path(img_filename).stem}.txt"
                with open(output_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
    
    print(f"Converted {len(coco_data['images'])} images to YOLO format")
