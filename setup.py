"""
Setup script for Pose Perfect
Initializes the project and downloads required models
"""

import os
import sys
from pathlib import Path
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    
    return True

def download_base_model():
    """Download YOLO11 pose model"""
    print("Downloading YOLO11 pose model...")
    
    try:
        from ultralytics import YOLO
        import os
        
        # This will download the model if not present
        model = YOLO("yolo11m-pose.pt")
        
        # Validate that it's a pose model, not detection model
        if hasattr(model, 'task') and model.task == 'pose':
            print("✓ YOLO11m-pose model ready (pose detection)")
        else:
            print("⚠️ Warning: Model task is not 'pose', got:", getattr(model, 'task', 'unknown'))
            
        # Clean up any incorrect detection models that might have been downloaded
        # Note: yolo11n.pt may be downloaded by ultralytics during AMP checks even when using pose models
        incorrect_models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11l.pt', 'yolo11x.pt']
        for incorrect_model in incorrect_models:
            if os.path.exists(incorrect_model):
                print(f"🧹 Removing incorrect detection model: {incorrect_model}")
                os.remove(incorrect_model)
                
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

def create_sample_poses():
    """Create some sample target poses"""
    print("Creating sample target poses...")
    
    try:
        import numpy as np
        import json
        
        # Create target_poses directory
        poses_dir = Path("target_poses")
        poses_dir.mkdir(exist_ok=True)
        
        # Sample poses (normalized keypoints for common poses)
        sample_poses = {
            "arms_up": {
                "name": "Arms Up",
                "keypoints": [
                    [0.0, -0.8],    # nose
                    [-0.1, -0.85],  # left_eye
                    [0.1, -0.85],   # right_eye
                    [-0.2, -0.8],   # left_ear
                    [0.2, -0.8],    # right_ear
                    [-0.3, -0.4],   # left_shoulder
                    [0.3, -0.4],    # right_shoulder
                    [-0.6, -0.8],   # left_elbow (raised)
                    [0.6, -0.8],    # right_elbow (raised)
                    [-0.8, -1.0],   # left_wrist (high)
                    [0.8, -1.0],    # right_wrist (high)
                    [-0.15, 0.0],   # left_hip
                    [0.15, 0.0],    # right_hip
                    [-0.15, 0.6],   # left_knee
                    [0.15, 0.6],    # right_knee
                    [-0.15, 1.0],   # left_ankle
                    [0.15, 1.0]     # right_ankle
                ],
                "visibility": [1.0] * 17
            },
            
            "t_pose": {
                "name": "T-Pose",
                "keypoints": [
                    [0.0, -0.8],    # nose
                    [-0.1, -0.85],  # left_eye
                    [0.1, -0.85],   # right_eye
                    [-0.2, -0.8],   # left_ear
                    [0.2, -0.8],    # right_ear
                    [-0.3, -0.4],   # left_shoulder
                    [0.3, -0.4],    # right_shoulder
                    [-0.7, -0.4],   # left_elbow (horizontal)
                    [0.7, -0.4],    # right_elbow (horizontal)
                    [-1.0, -0.4],   # left_wrist (extended)
                    [1.0, -0.4],    # right_wrist (extended)
                    [-0.15, 0.0],   # left_hip
                    [0.15, 0.0],    # right_hip
                    [-0.15, 0.6],   # left_knee
                    [0.15, 0.6],    # right_knee
                    [-0.15, 1.0],   # left_ankle
                    [0.15, 1.0]     # right_ankle
                ],
                "visibility": [1.0] * 17
            },
            
            "warrior_pose": {
                "name": "Warrior Pose",
                "keypoints": [
                    [0.0, -0.8],    # nose
                    [-0.1, -0.85],  # left_eye
                    [0.1, -0.85],   # right_eye
                    [-0.2, -0.8],   # left_ear
                    [0.2, -0.8],    # right_ear
                    [-0.4, -0.4],   # left_shoulder
                    [0.4, -0.4],    # right_shoulder
                    [-0.8, -0.3],   # left_elbow
                    [0.8, -0.3],    # right_elbow
                    [-1.0, -0.4],   # left_wrist
                    [1.0, -0.4],    # right_wrist
                    [-0.2, 0.0],    # left_hip
                    [0.2, 0.0],     # right_hip
                    [-0.5, 0.4],    # left_knee (lunged)
                    [0.2, 0.8],     # right_knee
                    [-0.6, 0.8],    # left_ankle
                    [0.2, 1.0]      # right_ankle
                ],
                "visibility": [1.0] * 17
            }
        }
        
        # Save sample poses
        for pose_name, pose_data in sample_poses.items():
            pose_file = poses_dir / f"{pose_name}.json"
            with open(pose_file, 'w') as f:
                json.dump(pose_data, f, indent=2)
        
        print(f"✓ Created {len(sample_poses)} sample target poses")
        return True
        
    except Exception as e:
        print(f"✗ Error creating sample poses: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("Setting up project directories...")
    
    directories = [
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/labels/train",
        "dataset/labels/val",
        "target_poses",
        "collected_data/images",
        "collected_data/annotations",
        "collected_data/videos",
        "runs/pose",
        "api/logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Project directories created")

def check_gpu():
    """Check GPU/accelerator availability"""
    print("Checking accelerator availability...")
    
    try:
        import torch
        
        # Check for Apple MPS (Metal Performance Shaders) - Apple Silicon
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                test_tensor = torch.zeros(1).to('mps')
                print("⚠️ Apple MPS available but disabled for pose models due to YOLO11 bug")
                print("   Your Mac has hardware acceleration but will use CPU for pose training")
                print("   Reference: https://github.com/ultralytics/ultralytics/issues/4031")
            except Exception as e:
                print(f"⚠️ MPS available but not functional: {e}")
        
        # Check for CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU available: {gpu_name}")
        else:
            print("ℹ️ No GPU detected, will use CPU (training will be slower)")
            
    except ImportError:
        print("⚠️ PyTorch not installed, cannot check accelerator availability")

def main():
    print("=" * 50)
    print("Setting up Pose Perfect")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Setup steps
    steps = [
        ("Setting up directories", setup_directories),
        ("Installing requirements", install_requirements),
        ("Downloading base model", download_base_model), 
        ("Creating sample poses", create_sample_poses),
        ("Checking GPU", check_gpu)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            success = step_func()
            if success is False:
                print(f"✗ Failed: {step_name}")
                return
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            return
    
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Collect pose data: python main.py collect <pose_name>")
    print("2. Auto-annotate data: python main.py annotate <image_dir>") 
    print("3. Split dataset: python main.py split <source_dir>")
    print("4. Train model: python main.py train")
    print("5. Start API server: python main.py api")
    print("\nFor help: python main.py -h")

if __name__ == "__main__":
    main()
