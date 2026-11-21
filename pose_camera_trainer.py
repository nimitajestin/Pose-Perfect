"""
Camera-based training for ARVR Pose Perfect
Captures frames from camera and trains pose-specific models
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import time
import shutil

from pose_specific_trainer import PoseSpecificTrainer

# ===================== GLOBAL CONFIGURATION VARIABLES =====================
# Easily configurable parameters for camera training

# Pose Configuration
DEFAULT_POSE_NAME = "WarriorPose"  # Default pose name if not specified
FRAMES_NEEDED = 100  # Target number of frames to capture

# Camera Configuration  
DEFAULT_CAMERA_ID = 0  # Default camera device ID
CAMERA_WIDTH = 640   # Camera capture width
CAMERA_HEIGHT = 480  # Camera capture height
CAMERA_FPS = 30      # Camera FPS

# Training Configuration
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_BASE_MODEL = "yolo11m-pose.pt"

# Capture Configuration
CAPTURE_COOLDOWN = 0.5  # Minimum seconds between captures (prevent spam)
AUTO_SAVE_INTERVAL = 10  # Save progress every N frames

# Display Configuration
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # Green
PROGRESS_COLOR = (255, 0, 0)  # Red
INSTRUCTION_COLOR = (0, 255, 255)  # Yellow

# Directory Configuration
CAMERA_DATA_DIR = "camera_captured_data"  # Base directory for camera captures
TEMP_CAPTURE_DIR = "temp_captures"        # Temporary directory during capture

# ===================== END GLOBAL CONFIGURATION =====================


class PoseCameraTrainer:
    """Camera-based pose training that integrates with existing PoseSpecificTrainer"""
    
    def __init__(self, base_model_path: str = DEFAULT_BASE_MODEL):
        """
        Initialize the camera trainer
        
        Args:
            base_model_path: Path to the base YOLO11 pose model
        """
        self.base_model_path = base_model_path
        self.base_trainer = PoseSpecificTrainer(base_model_path)
        
        # Create directories
        self.camera_data_dir = Path(CAMERA_DATA_DIR)
        self.camera_data_dir.mkdir(exist_ok=True)
        
        self.temp_capture_dir = Path(TEMP_CAPTURE_DIR)
        self.temp_capture_dir.mkdir(exist_ok=True)
        
        # Capture state
        self.captured_frames = []
        self.last_capture_time = 0
        
    def capture_frames_for_pose(self, pose_name: str, camera_id: int = DEFAULT_CAMERA_ID, 
                               frames_needed: int = FRAMES_NEEDED) -> str:
        """
        Capture frames for a specific pose using camera
        
        Args:
            pose_name: Name of the pose to capture
            camera_id: Camera device ID
            frames_needed: Target number of frames to capture
            
        Returns:
            Path to the folder containing captured images
        """
        print(f"Starting camera capture for pose: {pose_name}")
        print(f"Target frames: {frames_needed}")
        print("\\n" + "="*60)
        print("CAMERA CONTROLS:")
        print("  SPACE     - Capture frame")
        print("  'r'       - Review captured frames (delete bad ones)")
        print("  'c'       - Change camera source")
        print("  'q'       - Finish capturing and proceed to training")
        print("  ESC       - Cancel and exit")
        print("="*60 + "\\n")
        
        # Create pose-specific temp directory
        pose_temp_dir = self.temp_capture_dir / pose_name
        pose_temp_dir.mkdir(exist_ok=True)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        captured_count = 0
        
        try:
            while captured_count < frames_needed:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Create display frame
                display_frame = self._create_display_frame(
                    frame, pose_name, captured_count, frames_needed, camera_id
                )
                
                cv2.imshow('Pose Capture', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - capture frame
                    if time.time() - self.last_capture_time > CAPTURE_COOLDOWN:
                        self._save_captured_frame(frame, pose_temp_dir, captured_count)
                        captured_count += 1
                        self.last_capture_time = time.time()
                        
                        print(f"Captured frame {captured_count}/{frames_needed}")
                        
                        # Auto-save progress
                        if captured_count % AUTO_SAVE_INTERVAL == 0:
                            print(f"Progress saved: {captured_count} frames")
                
                elif key == ord('r'):  # Review frames
                    if captured_count > 0:
                        deleted_count = self._review_frames(pose_temp_dir)
                        captured_count -= deleted_count
                        print(f"Frames after review: {captured_count}")
                
                elif key == ord('c'):  # Change camera
                    print("\\nChanging camera source...")
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Get new camera ID from user
                    new_camera_id = self._get_camera_selection(camera_id)
                    if new_camera_id is not None:
                        camera_id = new_camera_id
                        print(f"Switched to camera {camera_id}")
                        
                        # Reinitialize camera
                        cap = cv2.VideoCapture(camera_id)
                        if not cap.isOpened():
                            print(f"Could not open camera {camera_id}, reverting to previous")
                            cap = cv2.VideoCapture(camera_id)
                        else:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                    else:
                        print("Camera change cancelled")
                        # Reopen current camera
                        cap = cv2.VideoCapture(camera_id)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                
                elif key == ord('q'):  # Finish capturing
                    if captured_count >= 10:  # Minimum frames check
                        print(f"\\nFinishing capture with {captured_count} frames")
                        break
                    else:
                        print(f"\\nNeed at least 10 frames (current: {captured_count})")
                
                elif key == 27:  # ESC - cancel
                    print("\\nCapture cancelled")
                    return None
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if captured_count < 10:
            print(f"Insufficient frames captured: {captured_count}")
            return None
        
        # Move frames to final directory
        final_pose_dir = self.camera_data_dir / pose_name
        if final_pose_dir.exists():
            shutil.rmtree(final_pose_dir)
        
        shutil.move(str(pose_temp_dir), str(final_pose_dir))
        
        print(f"\\n✓ Successfully captured {captured_count} frames for {pose_name}")
        print(f"Images saved to: {final_pose_dir}")
        
        return str(final_pose_dir)
    
    def _create_display_frame(self, frame: np.ndarray, pose_name: str, 
                             captured: int, needed: int, camera_id: int = 0) -> np.ndarray:
        """Create the display frame with overlays"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Title
        cv2.putText(display_frame, f"Capturing: {pose_name} (Camera {camera_id})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # Progress
        progress_text = f"Frames: {captured}/{needed} ({captured/needed*100:.1f}%)"
        cv2.putText(display_frame, progress_text, 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, PROGRESS_COLOR, FONT_THICKNESS)
        
        # Instructions
        instructions = [
            "SPACE: Capture",
            "'r': Review", 
            "'c': Change Camera",
            "'q': Finish (min 10)",
            "ESC: Cancel"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 100 + (i * 20)
            cv2.putText(display_frame, instruction, 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INSTRUCTION_COLOR, 1)
        
        # Progress bar
        bar_width = w - 20
        bar_height = 10
        bar_x, bar_y = 10, 90
        
        # Background bar
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        if needed > 0:
            progress_width = int(bar_width * (captured / needed))
            cv2.rectangle(display_frame, (bar_x, bar_y), 
                         (bar_x + progress_width, bar_y + bar_height), PROGRESS_COLOR, -1)
        
        return display_frame
    
    def _save_captured_frame(self, frame: np.ndarray, save_dir: Path, frame_number: int):
        """Save a captured frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{frame_number:04d}_{timestamp}.jpg"
        filepath = save_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        self.captured_frames.append(str(filepath))
    
    def _review_frames(self, pose_dir: Path) -> int:
        """
        Review captured frames and allow deletion of bad ones
        Returns number of deleted frames
        """
        image_files = list(pose_dir.glob("*.jpg"))
        if not image_files:
            print("No frames to review")
            return 0
        
        print(f"\\nReviewing {len(image_files)} captured frames...")
        print("Controls: 'd' = delete, SPACE = keep, 'q' = finish review")
        
        deleted_count = 0
        current_idx = 0
        
        while current_idx < len(image_files):
            if current_idx >= len(image_files):
                break
                
            img_path = image_files[current_idx]
            if not img_path.exists():  # Skip if already deleted
                current_idx += 1
                continue
            
            # Load and display image
            img = cv2.imread(str(img_path))
            if img is None:
                current_idx += 1
                continue
            
            # Create review display
            display_img = img.copy()
            h, w = display_img.shape[:2]
            
            # Add review overlay
            cv2.putText(display_img, f"Review: {current_idx + 1}/{len(image_files)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, "d=DELETE  SPACE=KEEP  q=FINISH", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display_img, f"File: {img_path.name}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Frame Review', display_img)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('d'):  # Delete frame
                img_path.unlink()
                print(f"Deleted: {img_path.name}")
                deleted_count += 1
                image_files.remove(img_path)
            elif key == ord(' '):  # Keep frame
                current_idx += 1
            elif key == ord('q'):  # Finish review
                break
            else:
                current_idx += 1
        
        cv2.destroyWindow('Frame Review')
        print(f"Review completed. Deleted {deleted_count} frames")
        return deleted_count
    
    def _get_camera_selection(self, current_camera_id: int) -> Optional[int]:
        """
        Interactive camera selection interface
        
        Args:
            current_camera_id: Currently active camera ID
            
        Returns:
            New camera ID or None if cancelled
        """
        print(f"\\nCurrent camera: {current_camera_id}")
        print("Testing available cameras...")
        
        # Test cameras 0-5 for availability
        available_cameras = []
        for i in range(6):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    available_cameras.append(i)
                    status = "✓ CURRENT" if i == current_camera_id else "✓ Available"
                    print(f"Camera {i}: {status}")
                else:
                    print(f"Camera {i}: ✗ Opens but no frames")
            else:
                print(f"Camera {i}: ✗ Not available")
            test_cap.release()
        
        if not available_cameras:
            print("No cameras found!")
            return None
        
        print(f"\\nAvailable cameras: {available_cameras}")
        
        while True:
            try:
                user_input = input(f"Enter camera ID (current: {current_camera_id}) or 'c' to cancel: ").strip()
                
                if user_input.lower() == 'c':
                    return None
                
                new_camera_id = int(user_input)
                
                if new_camera_id == current_camera_id:
                    print("Same camera selected, no change needed")
                    return None
                
                if new_camera_id in available_cameras:
                    # Test the selected camera
                    test_cap = cv2.VideoCapture(new_camera_id)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            print(f"✓ Camera {new_camera_id} selected successfully")
                            test_cap.release()
                            return new_camera_id
                        else:
                            print(f"✗ Camera {new_camera_id} failed to capture frames")
                    test_cap.release()
                else:
                    print(f"Camera {new_camera_id} is not available. Choose from: {available_cameras}")
                    
            except ValueError:
                print("Invalid input. Enter a number or 'c' to cancel.")
            except KeyboardInterrupt:
                print("\\nCamera selection cancelled")
                return None
    
    def train_pose_from_camera(self, pose_name: str = DEFAULT_POSE_NAME, 
                              camera_id: int = DEFAULT_CAMERA_ID,
                              frames_needed: int = FRAMES_NEEDED,
                              epochs: int = DEFAULT_EPOCHS, 
                              batch_size: int = DEFAULT_BATCH_SIZE,
                              train_ratio: float = DEFAULT_TRAIN_RATIO) -> Dict:
        """
        Complete pipeline: capture frames from camera and train pose model
        
        Args:
            pose_name: Name of the pose to train
            camera_id: Camera device ID
            frames_needed: Target number of frames to capture
            epochs: Training epochs
            batch_size: Training batch size
            train_ratio: Train/validation split ratio
            
        Returns:
            Training results dictionary
        """
        print(f"Starting camera-based training for pose: {pose_name}")
        
        # Step 1: Capture frames
        pose_folder_path = self.capture_frames_for_pose(pose_name, camera_id, frames_needed)
        
        if pose_folder_path is None:
            return {"success": False, "error": "Frame capture failed or cancelled"}
        
        print(f"\\n" + "="*60)
        print("STARTING TRAINING PIPELINE")
        print(f"{'='*60}")
        
        try:
            # Step 2: Prepare dataset (same as existing trainer)
            dataset_yaml = self.base_trainer.prepare_pose_dataset(
                pose_name, pose_folder_path, train_ratio
            )
            
            # Step 3: Train model (same as existing trainer)
            train_result = self.base_trainer.train_pose_model(
                pose_name, dataset_yaml, epochs, batch_size
            )
            
            # Step 4: Extract reference pose (same as existing trainer)
            reference_data = self.base_trainer.extract_reference_pose(
                pose_name, train_result['model_path'], pose_folder_path
            )
            
            result = {
                'success': True,
                'pose_name': pose_name,
                'captured_frames': len(list(Path(pose_folder_path).glob("*.jpg"))),
                'model_path': train_result['model_path'],
                'metadata_path': train_result['metadata_path'],
                'reference_data': reference_data,
                'camera_source': f"camera_{camera_id}",
                'training_params': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'train_ratio': train_ratio
                }
            }
            
            print(f"\\n✓ Camera-based training completed successfully!")
            print(f"Model saved to: {train_result['model_path']}")
            
            return result
            
        except Exception as e:
            print(f"\\n✗ Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pose_name': pose_name
            }
    
    def cleanup_temp_files(self):
        """Clean up temporary capture files"""
        if self.temp_capture_dir.exists():
            shutil.rmtree(self.temp_capture_dir)
            self.temp_capture_dir.mkdir(exist_ok=True)
        print("Temporary files cleaned up")


def main():
    """Main function for standalone camera training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera-based Pose Training")
    parser.add_argument('--pose-name', default=DEFAULT_POSE_NAME, help='Name of pose to train')
    parser.add_argument('--camera-id', type=int, default=DEFAULT_CAMERA_ID, help='Camera device ID')
    parser.add_argument('--frames-needed', type=int, default=FRAMES_NEEDED, help='Number of frames to capture')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--train-ratio', type=float, default=DEFAULT_TRAIN_RATIO, help='Train/val split')
    parser.add_argument('--base-model', default=DEFAULT_BASE_MODEL, help='Base YOLO model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PoseCameraTrainer(args.base_model)
    
    # Run training
    result = trainer.train_pose_from_camera(
        pose_name=args.pose_name,
        camera_id=args.camera_id, 
        frames_needed=args.frames_needed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio
    )
    
    # Cleanup
    trainer.cleanup_temp_files()
    
    if result['success']:
        print(f"\\n🎉 Training completed successfully for {args.pose_name}!")
    else:
        print(f"\\n❌ Training failed: {result.get('error', 'Unknown error')}")
    
    return result


if __name__ == "__main__":
    main()