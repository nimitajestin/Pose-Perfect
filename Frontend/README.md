# AR/VR Pose Perfect - Frontend

A React-based web application for real-time pose comparison and scoring using dual webcam support.

## Features

- **Dual Webcam Support**: Compare poses between two players simultaneously
- **Real-time Scoring**: Live pose evaluation with percentage accuracy
- **Timer-based Games**: Configurable countdown timer for competitive sessions
- **Pose Selection**: Choose from available trained pose models
- **Score Tracking**: Average score calculation and winner determination
- **Responsive Design**: Works on desktop and tablet devices
- **Camera Controls**: Multiple camera selection and configuration

## Setup

### Prerequisites

- Node.js (v16 or higher)
- NPM or Yarn
- Backend API server running on `http://localhost:8000`
- At least one webcam (two cameras recommended for dual player mode)

### Installation

1. Navigate to the Frontend directory:
   ```bash
   cd Frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser to `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Usage

### Getting Started

1. **Start the Backend**: Ensure the API server is running on port 8000
2. **Camera Permissions**: Allow browser access to your cameras when prompted
3. **Select Pose**: Choose a pose from the dropdown menu
4. **Configure Cameras**: Select different cameras for Player 1 and Player 2 (optional)
5. **Start Game**: Click the timer's "Start" button to begin scoring
6. **Play**: Try to match the selected pose as accurately as possible
7. **View Results**: Check real-time scores and see the winner at the end

### Game Phases

- **Setup**: Select pose and configure cameras
- **Playing**: Timer active, poses being scored in real-time
- **Finished**: Game complete, winner displayed

### Scoring System

- **90-100%**: Excellent pose match (Green)
- **70-89%**: Good pose match (Yellow)  
- **50-69%**: Fair pose match (Orange)
- **0-49%**: Needs improvement (Red)

## API Integration

The frontend connects to the backend API with the following endpoints:

- `GET /poses` - List available poses
- `POST /score` - Score individual pose images
- `GET /` - Health check

### Error Handling

- Automatic retry on connection failures
- Clear error messages for API issues
- Graceful degradation when cameras are unavailable

## Components

### Core Components

- **WebcamCapture**: Real-time camera feed with auto-capture
- **PoseSelector**: Dropdown for choosing poses with model info
- **Timer**: Configurable countdown timer with controls
- **ScoreDisplay**: Real-time score visualization and comparison

### Features

- **Responsive Layout**: Adapts to different screen sizes
- **Real-time Updates**: Live scoring during gameplay
- **Camera Selection**: Support for multiple camera devices
- **Score Tracking**: Historical score averaging
- **Winner Determination**: Automatic winner calculation

## Configuration

### Timer Settings

- Default duration: 30 seconds
- Configurable from 10-300 seconds
- Real-time countdown display

### Camera Settings

- Automatic device enumeration
- Resolution: 640x480 optimized for pose detection
- 1 FPS capture rate for scoring

## Browser Compatibility

- **Chrome/Chromium**: Full support
- **Firefox**: Full support
- **Safari**: Full support (macOS/iOS)
- **Edge**: Full support

### Required Permissions

- Camera access for webcam functionality
- Microphone access may be requested but is not used

## Performance

### Optimization Features

- Efficient image capture (1 FPS)
- Lazy loading of camera streams
- Optimized API calls with error handling
- Smooth animations and transitions

### System Requirements

- Modern web browser with WebRTC support
- 4GB+ RAM recommended
- Stable internet connection for API communication
- USB webcam or built-in camera

## Troubleshooting

### Common Issues

1. **"Server is not running" Error**
   - Start the backend API server on port 8000
   - Check firewall settings

2. **Camera Not Working**
   - Grant camera permissions in browser
   - Check if camera is being used by another application
   - Try refreshing the page

3. **No Poses Available**
   - Ensure backend has trained pose models
   - Check API connectivity
   - Verify models exist in `trained_models/` directory

4. **Low Scoring Accuracy**
   - Ensure good lighting conditions
   - Position yourself clearly in the camera frame
   - Match the reference pose as closely as possible

### Debug Mode

Open browser developer tools (F12) to see:
- API request/response logs
- Camera initialization status
- Real-time scoring data
- Error details

## Development

### Project Structure

```
Frontend/
├── src/
│   ├── components/          # React components
│   │   ├── WebcamCapture.jsx
│   │   ├── PoseSelector.jsx
│   │   ├── Timer.jsx
│   │   └── ScoreDisplay.jsx
│   ├── services/           # API service layer
│   │   └── api.js
│   ├── App.jsx            # Main application component
│   ├── main.jsx           # React entry point
│   └── index.css          # Global styles
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

### Tech Stack

- **React 19**: UI framework
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first styling
- **Axios**: HTTP client for API calls
- **react-webcam**: Camera access and capture
- **Lucide React**: Icon library

### Adding New Features

1. Create new components in `src/components/`
2. Add API endpoints to `src/services/api.js`
3. Update main App.jsx as needed
4. Style with Tailwind CSS classes

## License

This project is part of the AR/VR Pose Perfect system. See the main project LICENSE file for details.