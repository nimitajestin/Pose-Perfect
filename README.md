# Pose Perfect - YOLO11 Pose Detection & Scoring System

A comprehensive pose detection and scoring system built with YOLO11, designed for real-time pose comparison and evaluation. **Now featuring a complete 2-Player Competition Game with C# WPF Frontend!!**

## 🎯 Features

### **Backend (Python)**
- **Real-time Pose Detection**: Fast and accurate human pose detection using YOLO11
- **Pose Similarity Scoring**: Compare detected poses against target poses with 0-100 scoring
- **Custom Pose Training**: Fine-tune models on your specific poses and scenarios
- **Data Collection Tools**: Easy-to-use tools for capturing and annotating pose data
- **FastAPI Server**: RESTful API for integration with web/mobile applications
- **Visualization Tools**: Rich visualization and analysis capabilities
- **Multi-platform Support**: Works on Windows, Linux, and macOS

### **Frontend (C# WPF) - NEW! 🎮**
- **2-Player Competition Game**: Head-to-head pose matching gameplay
- **Dual Camera Support**: Independent camera selection for each player
- **Progressive Difficulty**: 3 rounds with decreasing time limits (10s → 5s → 3s)
- **Real-time Scoring**: Live pose comparison with visual feedback
- **Leaderboard System**: Track high scores and player statistics
- **Modern UI**: Dark theme with smooth animations and responsive design
- **Comprehensive Analytics**: Player stats, pose difficulty ratings, game history

## 🚀 Quick Start

### 1. Backend Setup (Python)

```bash
# Clone the repository
git clone https://github.com/your-username/ARVR-Pose-Perfect.git
cd ARVR-Pose-Perfect

# Run setup script
python setup.py
```

This will:
- Install required dependencies
- Download **YOLO11m-pose model** (pose estimation, not object detection)
- Create project directories  
- Set up sample target poses
- Validate model configuration

**⚠️ Important**: The system specifically requires `yolo11m-pose.pt` for pose estimation. Do not use detection models like `yolo11n.pt`, `yolo11s.pt`, etc.

### 2. Frontend Setup (C# WPF Game)

```bash
# Navigate to frontend directory
cd frontend

# Open in Visual Studio
start PosePerfect.sln
```

**Requirements:**
- Visual Studio 2022 with .NET 6 SDK
- Connected cameras (webcam/USB cameras)
- Python backend running on localhost:8000

### 3. Start the Game

1. **Start Backend**: Run `python main.py api` (from root directory)
2. **Start Frontend**: Build and run the WPF application
3. **Select Cameras**: Choose different cameras for each player
4. **Play**: Press START and compete in pose matching!
- Create project directories
- Set up sample target poses

### 2. Test the System

```bash
# Test pose detection on an image
python main.py test path/to/your/image.jpg --visualize

# Start the API server
python main.py api

# Visit http://localhost:8000 for API documentation

### 3. Verify Model Installation

To ensure correct model setup (especially on Windows):

```bash
python verify_models.py
```

This verification script will:
- ✅ Check for correct YOLO11m-pose model (42MB)
- ✅ Validate model task is 'pose' (not 'detect')
- ✅ Remove incorrect detection models (yolo11n.pt, yolo11s.pt, etc.)
- ✅ Display model information and validation results

**If you see yolo11n.pt or other detection models**, the script will automatically remove them and ensure only the correct pose model is present.

## 🎮 AR/VR Game Development
```

## 📊 Core Components

### 1. Pose Detection Model (`pose_model/`)
- **PoseDetector**: Main class for pose detection and scoring
- **TargetPoseManager**: Manages target poses for comparison
- Supports COCO-17 keypoint format
- Normalized pose representation for scale/translation invariance

### 2. Data Collection (`data_collection/`)
- **DataCollector**: Camera and video data collection
- **AnnotationHelper**: Auto-annotation using pretrained models
- **DatasetSplitter**: Train/validation split utilities

### 3. FastAPI Server (`api/`)
- RESTful endpoints for pose detection and scoring
- Batch processing support
- Target pose management
- Model training endpoints

### 4. Utilities (`utils/`)
- Pose visualization
- Performance evaluation
- Demo video creation
- Data format conversion

## 🎮 Usage Guide

### Data Collection

Collect pose data for training:

```bash
# Collect data from camera
python main.py collect "warrior_pose" --camera-id 0

# Extract frames from video
python main.py collect "dance_move" --video path/to/video.mp4
```

### Auto-Annotation

Automatically annotate collected images:

```bash
python main.py annotate collected_data/images/warrior_pose
```

### Dataset Preparation

Split annotated data for training:

```bash
python main.py split collected_data --train-ratio 0.8
```

### Model Training

Fine-tune YOLO11 on your poses:

```bash
python main.py train --epochs 100 --batch-size 16
```

### API Server

Start the FastAPI server:

```bash
python main.py api --port 8000
```

### API Endpoints

- `GET /`: Health check
- `GET /poses`: List target poses
- `POST /poses`: Add new target pose
- `POST /infer`: Single image inference
- `POST /infer/batch`: Batch inference
- `POST /capture_target_pose`: Capture pose from image
- `POST /train`: Start model training

### Example API Usage

```python
import requests
import json

# Upload image for pose detection
with open('pose_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/infer',
        data={'target_pose_name': 'warrior_pose'},
        files={'file': f}
    )

result = response.json()
print(f"Pose score: {result['poses'][0]['score']}")
```

## 🏗️ Architecture

```
ARVR-Pose-Perfect/
├── pose_model/           # Core pose detection & scoring
│   └── pose_detector.py
├── data_collection/      # Data collection & annotation
│   └── data_collector.py
├── api/                  # FastAPI server
│   └── api_server.py
├── utils/               # Utilities & visualization
│   └── pose_utils.py
├── dataset/             # Training dataset
│   ├── images/
│   ├── labels/
│   └── dataset.yaml
├── target_poses/        # Target pose definitions
├── collected_data/      # Raw collected data
├── main.py             # CLI interface
├── setup.py            # Setup script
└── requirements.txt    # Dependencies
```

## 🎯 Scoring Algorithm

The pose similarity scoring uses:

1. **Normalization**: Poses are normalized for scale and translation invariance
2. **Keypoint Weighting**: Different body parts have different importance weights
3. **Visibility Handling**: Only visible keypoints are compared
4. **Error Calculation**: Weighted euclidean distance between keypoints
5. **Score Conversion**: Error is converted to 0-100 score range

## 🔧 Configuration

### Model Configuration
- Base model: YOLO11m-pose (can be changed to yolo11n/s/l/x)
- Input size: 640x640 (configurable)
- Confidence threshold: 0.5 (adjustable)

### Scoring Configuration
```python
# Keypoint weights (can be adjusted)
keypoint_weights = [
    1.5,  # nose
    1.0, 1.0,  # eyes
    0.8, 0.8,  # ears
    2.0, 2.0,  # shoulders
    1.5, 1.5,  # elbows
    1.2, 1.2,  # wrists
    2.5, 2.5,  # hips (most important)
    2.0, 2.0,  # knees
    1.5, 1.5   # ankles
]
```

## 📈 Performance

### Hardware Recommendations
- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM
- **Optimal**: RTX 3060+ or equivalent

### Speed Benchmarks
- **YOLO11n**: ~100 FPS (RTX 3060)
- **YOLO11m**: ~60 FPS (RTX 3060)
- **YOLO11l**: ~40 FPS (RTX 3060)

### Accuracy
- Base COCO mAP: 65.6% (YOLO11m)
- Fine-tuned mAP: 70-80% (depends on dataset quality)

## 🛠️ Development

### Adding New Poses

1. **Manual Definition**:
```python
from pose_model.pose_detector import TargetPoseManager

manager = TargetPoseManager()
manager.add_target_pose("my_pose", keypoints, visibility)
```

2. **From Image**:
```bash
# Capture pose from image
curl -X POST "http://localhost:8000/capture_target_pose" \
  -F "pose_name=my_pose" \
  -F "file=@pose_image.jpg"
```

### Custom Training

1. Collect 200-1000 images per pose
2. Use auto-annotation or manual annotation
3. Fine-tune model with your data
4. Evaluate and iterate

### Integration Examples

**React/JavaScript**:
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('target_pose_name', 'warrior_pose');

fetch('http://localhost:8000/infer', {
  method: 'POST',
  body: formData
}).then(response => response.json())
  .then(data => console.log('Score:', data.poses[0].score));
```

**Python**:
```python
from pose_model.pose_detector import PoseDetector

detector = PoseDetector()
poses = detector.detect_poses(image)
score = detector.calculate_pose_similarity(pose1, pose2)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use smaller model (yolo11n)
   - Reduce image size

2. **Low Detection Accuracy**:
   - Check lighting conditions
   - Ensure full body is visible
   - Adjust confidence threshold

3. **Poor Scoring**:
   - Verify target pose quality
   - Check keypoint normalization
   - Adjust scoring weights

### Debug Commands

```bash
# Test with verbose output
python main.py test image.jpg --visualize --conf-threshold 0.3

# Check accelerator availability
python -c "import torch; print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False); print('CUDA:', torch.cuda.is_available())"

# Validate dataset with correct pose model
python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt').val(data='dataset/dataset.yaml')"
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- Create an issue for bug reports
- Check documentation for common questions
- Join our Discord for community support

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the base model
- [COCO Dataset](https://cocodataset.org/) for keypoint annotations
- FastAPI team for the excellent web framework

---

**Built with ❤️ for the pose estimation community**
