import React, { useRef, useCallback, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { Camera, CameraOff } from 'lucide-react';

const WebcamCapture = ({
  onCapture,
  isActive,
  playerName,
  deviceId,
  onDeviceChange,
  availableDevices = [],
  compact = false
}) => {
  const webcamRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(null);

  const capture = useCallback(() => {
    if (webcamRef.current && isActive && isReady) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc && onCapture) {
        onCapture(imageSrc, playerName);
      }
    }
  }, [isActive, isReady, onCapture, playerName]);

  const handleUserMedia = useCallback(() => {
    setIsReady(true);
    setError(null);
  }, []);

  const handleUserMediaError = useCallback((error) => {
    setError(error.message);
    setIsReady(false);
  }, []);

  // Auto-capture when active and ready
  useEffect(() => {
    if (isActive && isReady) {
      // Add offset for Player 2 to prevent simultaneous API calls
      const offset = playerName === 'Player 2' ? 1500 : 0; // 1.5 second offset for Player 2
      let interval;

      const timeout = setTimeout(() => {
        interval = setInterval(capture, 2600); // Capture every 2.6 seconds for 5 frames per 13-second pose
      }, offset);

      return () => {
        clearTimeout(timeout);
        if (interval) clearInterval(interval);
      };
    }
  }, [isActive, isReady, capture, playerName]);

  const videoConstraints = {
    width: 640,
    height: 480,
    deviceId: deviceId
  };

  if (compact) {
    return (
      <div className="relative bg-black rounded-lg overflow-hidden">
        <div className="aspect-video bg-black flex items-center justify-center border border-gray-600">
          {isActive ? (
            <>
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                onUserMedia={handleUserMedia}
                onUserMediaError={handleUserMediaError}
                className="w-full h-full object-cover"
              />
              {!isReady && (
                <div className="absolute inset-0 flex items-center justify-center bg-black">
                  <div className="text-white text-center">
                    <Camera className="mx-auto mb-2" size={32} />
                    <p className="text-sm">Initializing...</p>
                  </div>
                </div>
              )}
              {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-red-900">
                  <div className="text-white text-center">
                    <CameraOff className="mx-auto mb-2" size={32} />
                    <p className="text-xs">Camera Error: {error}</p>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-gray-400 text-center">
              <CameraOff className="mx-auto mb-2" size={32} />
              <p className="text-sm">Camera Off</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="relative bg-gray-900 rounded-lg overflow-hidden">
      <div className="aspect-video bg-gray-800 flex items-center justify-center">
        {isActive ? (
          <>
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              onUserMedia={handleUserMedia}
              onUserMediaError={handleUserMediaError}
              className="w-full h-full object-cover"
            />
            {!isReady && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                <div className="text-white text-center">
                  <Camera className="mx-auto mb-2" size={48} />
                  <p>Initializing camera...</p>
                </div>
              </div>
            )}
            {error && (
              <div className="absolute inset-0 flex items-center justify-center bg-red-900">
                <div className="text-white text-center">
                  <CameraOff className="mx-auto mb-2" size={48} />
                  <p className="text-sm">Camera Error: {error}</p>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-gray-400 text-center">
            <CameraOff className="mx-auto mb-2" size={48} />
            <p>Camera Disabled</p>
          </div>
        )}
      </div>

      {/* Player Info and Camera Controls */}
      <div className="p-4 bg-gray-800">
        <h3 className="text-white font-semibold mb-2">{playerName}</h3>

        {/* Camera Selection */}
        {availableDevices.length > 1 && (
          <div className="mb-2">
            <label className="text-gray-300 text-sm block mb-1">Camera:</label>
            <select
              value={deviceId || ''}
              onChange={(e) => onDeviceChange(e.target.value)}
              className="w-full p-1 text-sm bg-gray-700 text-white rounded border border-gray-600"
            >
              <option value="">Default Camera</option>
              {availableDevices.map((device, index) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${index + 1}`}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Status Indicators */}
        <div className="flex justify-between items-center text-sm">
          <span className={`px-2 py-1 rounded ${isReady ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
            }`}>
            {isReady ? 'Ready' : 'Not Ready'}
          </span>
          <span className={`px-2 py-1 rounded ${isActive ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
            }`}>
            {isActive ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default WebcamCapture;