import React, { useState, useEffect, useCallback } from 'react';
import { AlertCircle, Settings, Wifi, WifiOff, Play, Pause } from 'lucide-react';

// Component imports
import WebcamCapture from './components/WebcamCapture';
import PoseSelector from './components/PoseSelector';
import PoseDisplay from './components/PoseDisplay';
import Timer from './components/Timer';
import ScoreDisplay from './components/ScoreDisplay';

// API service
import { poseAPI, utils } from './services/api';

function App() {
  // Core state
  const [poses, setPoses] = useState([]);
  const [selectedPose, setSelectedPose] = useState(null);
  const [currentPoseIndex, setCurrentPoseIndex] = useState(0);
  const [isGameActive, setIsGameActive] = useState(false);
  const [gamePhase, setGamePhase] = useState('setup'); // 'setup', 'countdown', 'playing', 'finished'

  // API state
  const [isLoadingPoses, setIsLoadingPoses] = useState(true);
  const [apiError, setApiError] = useState(null);
  const [isApiConnected, setIsApiConnected] = useState(false);

  // Camera state
  const [availableCameras, setAvailableCameras] = useState([]);
  const [player1Camera, setPlayer1Camera] = useState('');
  const [player2Camera, setPlayer2Camera] = useState('');
  const [camerasActive, setCamerasActive] = useState(false);

  // Scoring state
  const [player1Score, setPlayer1Score] = useState(null);
  const [player2Score, setPlayer2Score] = useState(null);
  const [player1Scores, setPlayer1Scores] = useState([]);
  const [player2Scores, setPlayer2Scores] = useState([]);
  const [player1TotalScore, setPlayer1TotalScore] = useState(0);
  const [player2TotalScore, setPlayer2TotalScore] = useState(0);
  const [winner, setWinner] = useState(null);

  // Settings state
  const [timerDuration, setTimerDuration] = useState(13); // 13 seconds per pose (9 poses total)
  const [showSettings, setShowSettings] = useState(false);
  const [poseChangeInterval, setPoseChangeInterval] = useState(null);

  // Game timer state (9 poses × 13 seconds = 117 seconds)
  const [gameTimeLeft, setGameTimeLeft] = useState(117);

  // Player names state
  const [player1Name, setPlayer1Name] = useState('Player 1');
  const [player2Name, setPlayer2Name] = useState('Player 2');
  const [showNameModal, setShowNameModal] = useState(false);

  // UI state
  const [currentRound, setCurrentRound] = useState(1);
  const [showLeaderboard, setShowLeaderboard] = useState(false);
  const [showGameResultModal, setShowGameResultModal] = useState(false);
  const [timeLeft, setTimeLeft] = useState(timerDuration);

  // Initialize app
  useEffect(() => {
    const init = async () => {
      try {
        // Check API connection
        await poseAPI.healthCheck();
        setIsApiConnected(true);
        setApiError(null);

        // Load available poses
        const posesData = await poseAPI.getPoses();
        setPoses(posesData || []);

        // Get available cameras
        const cameras = await getCameraDevices();
        setAvailableCameras(cameras);

      } catch (error) {
        console.error('App initialization failed:', error);
        setApiError(utils.formatApiError(error));
        setIsApiConnected(false);
      } finally {
        setIsLoadingPoses(false);
      }
    };

    init();
  }, []);

  const initializeApp = async () => {
    try {
      // Check API connection
      await poseAPI.healthCheck();
      setIsApiConnected(true);
      setApiError(null);

      // Load available poses
      const posesData = await poseAPI.getPoses();
      setPoses(posesData || []);

      // Get available cameras
      const cameras = await getCameraDevices();
      setAvailableCameras(cameras);

    } catch (error) {
      console.error('App initialization failed:', error);
      setApiError(utils.formatApiError(error));
      setIsApiConnected(false);
    } finally {
      setIsLoadingPoses(false);
    }
  };

  // Get available camera devices
  const getCameraDevices = async () => {
    try {
      // Request camera permission first to get proper device labels
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        stream.getTracks().forEach(track => track.stop()); // Stop the stream immediately
      } catch {
        console.warn('Camera permission not granted, device labels may be limited');
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');

      console.log('Available cameras:', videoDevices.map(d => ({
        id: d.deviceId,
        label: d.label || 'Unknown Camera'
      })));

      return videoDevices;
    } catch (error) {
      console.error('Failed to get camera devices:', error);
      return [];
    }
  };

  // Game timer and pose progression
  useEffect(() => {
    if (isGameActive && poses.length > 0) {
      // Set initial pose and start 2-minute game timer
      setSelectedPose(poses[0]);
      setCurrentPoseIndex(0);
      setGameTimeLeft(117); // 9 poses × 13 seconds = 117 seconds

      // Clear any existing intervals
      if (poseChangeInterval) {
        clearInterval(poseChangeInterval);
      }

      // Main game timer - counts down from 120 seconds
      const gameTimer = setInterval(() => {
        setGameTimeLeft(prevTime => {
          const newTime = prevTime - 1;

          // Check if game should end
          if (newTime <= 0) {
            setIsGameActive(false);
            setGamePhase('finished');
            setCamerasActive(false);

            // Game will end, winner calculation will happen in a separate effect

            clearInterval(gameTimer);
            return 0;
          }

          return newTime;
        });
      }, 1000);

      // Pose change timer - switches pose every 10 seconds
      const poseTimer = setInterval(() => {
        setCurrentPoseIndex(prevIndex => {
          const nextIndex = prevIndex + 1;

          // If we've gone through all poses, cycle back to start
          if (nextIndex >= poses.length) {
            setSelectedPose(poses[0]);
            setPlayer1Score(null);
            setPlayer2Score(null);
            return 0;
          }

          // Move to next pose
          setSelectedPose(poses[nextIndex]);
          // Reset current pose scores for new pose
          setPlayer1Score(null);
          setPlayer2Score(null);

          return nextIndex;
        });
      }, 13000); // Change pose every 13 seconds (9 poses total)

      setPoseChangeInterval(poseTimer);

      return () => {
        clearInterval(gameTimer);
        clearInterval(poseTimer);
      };
    }
  }, [isGameActive, poses]);

  // Handle game end and winner calculation
  useEffect(() => {
    if (gamePhase === 'finished' && !winner) {
      // Calculate final scores and determine winner
      const finalPlayer1Score = player1Scores.reduce((sum, score) => sum + score, 0);
      const finalPlayer2Score = player2Scores.reduce((sum, score) => sum + score, 0);

      setPlayer1TotalScore(finalPlayer1Score);
      setPlayer2TotalScore(finalPlayer2Score);

      if (finalPlayer1Score > finalPlayer2Score) {
        setWinner(player1Name);
      } else if (finalPlayer2Score > finalPlayer1Score) {
        setWinner(player2Name);
      } else {
        setWinner('Tie');
      }

      // Show the result modal
      setShowGameResultModal(true);
    }
  }, [gamePhase, winner, player1Scores, player2Scores, player1Name, player2Name]);

  // Handle pose selection (for debugging/manual mode - now unused in auto mode)
  const handlePoseSelect = useCallback((pose) => {
    if (!isGameActive) {
      setSelectedPose(pose);
    }
  }, [isGameActive]);

  // Handle frame capture from webcams
  const handleFrameCapture = useCallback(async (imageBase64, playerName) => {
    if (!selectedPose || !isGameActive) return;

    try {
      const poseName = selectedPose?.name || selectedPose;
      const response = await poseAPI.scorePose(imageBase64, poseName);

      if (response && typeof response.similarity_score === 'number') {
        const score = response.similarity_score; // Use raw score without percentage conversion

        if (playerName === player1Name) {
          setPlayer1Scores(prev => {
            const newScores = [...prev];
            // For automatic game mode, we want the best score for current pose
            const currentPoseScores = newScores.slice(-5); // Last 5 scores for current pose
            const bestCurrentScore = Math.max(score, ...currentPoseScores);

            // Add score to total accumulation
            newScores.push(score);
            setPlayer1Score(bestCurrentScore); // Show best score for current pose
            return newScores;
          });
        } else if (playerName === player2Name) {
          setPlayer2Scores(prev => {
            const newScores = [...prev];
            // For automatic game mode, we want the best score for current pose
            const currentPoseScores = newScores.slice(-5); // Last 5 scores for current pose
            const bestCurrentScore = Math.max(score, ...currentPoseScores);

            // Add score to total accumulation
            newScores.push(score);
            setPlayer2Score(bestCurrentScore); // Show best score for current pose
            return newScores;
          });
        }
      }
    } catch (error) {
      console.error('Error scoring pose:', error);
    }
  }, [selectedPose, isGameActive, player1Name, player2Name]);

  // Timer handlers
  const handleTimerStart = () => {
    setShowNameModal(true);
  };

  const startGameWithNames = () => {
    setShowNameModal(false);
    setGamePhase('playing');
    setIsGameActive(true);
    setCamerasActive(true);
    setGameTimeLeft(117); // Start with 9 poses × 13 seconds

    // Reset scores for new game
    setCurrentPoseIndex(0);
    setPlayer1Score(null);
    setPlayer2Score(null);
    setPlayer1Scores([]);
    setPlayer2Scores([]);
    setPlayer1TotalScore(0);
    setPlayer2TotalScore(0);
    setWinner(null);

    // Set first pose if poses are available
    if (poses.length > 0) {
      setSelectedPose(poses[0]);
    }
  }; const handleTimerComplete = () => {
    // Timer complete is now handled by automatic pose progression
    // This function is kept for compatibility but logic moved to useEffect
  };

  const handleTimerTick = (newTimeLeft) => {
    setTimeLeft(newTimeLeft);
  };

  const handleGameReset = () => {
    setGamePhase('setup');
    setIsGameActive(false);
    setCamerasActive(false);
    setCurrentPoseIndex(0);
    setSelectedPose(null);
    setPlayer1Score(null);
    setPlayer2Score(null);
    setPlayer1Scores([]);
    setPlayer2Scores([]);
    setPlayer1TotalScore(0);
    setPlayer2TotalScore(0);
    setWinner(null);
    setGameTimeLeft(117);
    setCurrentRound(1);
    setShowGameResultModal(false);

    // Clear pose change interval
    if (poseChangeInterval) {
      clearInterval(poseChangeInterval);
      setPoseChangeInterval(null);
    }
  };  // Leaderboard handlers
  const handleShowLeaderboard = () => {
    setShowLeaderboard(true);
  };

  const handleCloseLeaderboard = () => {
    setShowLeaderboard(false);
  };

  // Calculate average scores
  const getAverageScore = (scores) => {
    if (scores.length === 0) return null;
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  };

  const canStartGame = poses.length > 0 && isApiConnected && availableCameras.length >= 1;

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-full mx-auto flex items-center justify-between">
          {/* Left spacing */}
          <div className="flex-1"></div>

          {/* Centered Title */}
          <h1 className="text-3xl font-bold text-center">POSE PERFECT</h1>

          {/* Right side - API Status and Settings */}
          <div className="flex-1 flex justify-end items-center space-x-3">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${isApiConnected
              ? 'bg-green-900 text-green-200'
              : 'bg-red-900 text-red-200'
              }`}>
              <div className={`w-2 h-2 rounded-full ${isApiConnected ? 'bg-green-400' : 'bg-red-400'
                }`}></div>
              <span>{isApiConnected ? 'API Connected' : 'API Disconnected'}</span>
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              title="Settings"
            >
              <Settings size={16} />
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 p-6">
        {/* Error Alert */}
        {apiError && (
          <div className="mb-4 bg-red-900 border border-red-600 rounded-lg p-4 flex items-start space-x-3">
            <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <h3 className="font-semibold text-red-200">Connection Error</h3>
              <p className="text-red-300">{apiError}</p>
              <button
                onClick={initializeApp}
                className="mt-2 px-3 py-1 bg-red-700 hover:bg-red-600 rounded text-sm transition-colors"
              >
                Retry Connection
              </button>
            </div>
          </div>
        )}

        {/* Main Game Layout - Three Panel Design */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          {/* Left Panel - Player 1 */}
          <div className="bg-gray-800 rounded-lg border border-gray-600 p-4">
            <div className="mb-4">
              <h2 className="text-xl font-bold text-cyan-400 mb-2">{player1Name.toUpperCase()}</h2>

              {/* Camera Selection */}
              <div className="mb-3">
                <label className="block text-sm font-medium text-gray-300 mb-1">Camera Source:</label>
                <select
                  value={player1Camera || ''}
                  onChange={(e) => setPlayer1Camera(e.target.value)}
                  className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                >
                  <option value="">Select Camera</option>
                  {availableCameras.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              </div>

              {/* Recording Status */}
              <div className="flex items-center space-x-2 mb-3">
                <div className={`w-3 h-3 rounded-full ${camerasActive && isGameActive ? 'bg-red-500 animate-pulse' : 'bg-gray-500'
                  }`}></div>
                <span className="text-sm text-gray-300">
                  {camerasActive && isGameActive ? 'Recording' : 'Standby'}
                </span>
              </div>
            </div>

            {/* Camera View */}
            <div className="mb-4">
              <WebcamCapture
                onCapture={handleFrameCapture}
                isActive={camerasActive}
                playerName={player1Name}
                deviceId={player1Camera}
                onDeviceChange={setPlayer1Camera}
                availableDevices={availableCameras}
                compact={true}
              />
            </div>

            {/* Score */}
            <div className="mb-4">
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">Current Pose</div>
                <div className="text-3xl font-bold text-cyan-400">
                  {player1Score ? player1Score.toFixed(1) : '0'}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Total: {player1Scores.reduce((sum, score) => sum + score, 0).toFixed(1)}
                </div>
              </div>
            </div>

            {/* Recent Scores */}
            <div>
              <div className="text-sm text-gray-400 mb-2">Recent Scores</div>
              <div className="bg-gray-700 rounded p-2 min-h-[80px] max-h-[120px] overflow-y-auto">
                {player1Scores.length > 0 ? (
                  <div className="space-y-1">
                    {player1Scores.slice(-5).map((score, index) => (
                      <div key={index} className="text-sm text-gray-300">
                        Round {player1Scores.length - 4 + index}: {score.toFixed(3)}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">No scores yet</div>
                )}
              </div>
            </div>
          </div>

          {/* Center Panel - Game Controls */}
          <div className="bg-gray-800 rounded-lg border border-gray-600 p-6">
            {/* Status Text */}
            <div className="text-center mb-6">
              <div className="text-lg font-medium text-white mb-2">
                {!selectedPose && "Ready to Play! Select cameras and press START."}
                {selectedPose && !isGameActive && "Ready to start! Press START GAME."}
                {isGameActive && "Game in progress! Match the pose!"}
                {gamePhase === 'finished' && winner && `Game Over! ${winner} wins!`}
              </div>
            </div>

            {/* Game Timer Display */}
            <div className="mb-6">
              <div className="text-center">
                <span className="text-sm text-gray-400 mb-2 block">Game Timer</span>
                <div className="relative inline-block">
                  <div className={`text-4xl font-mono font-bold ${isGameActive
                    ? gameTimeLeft > 60
                      ? 'text-green-400'
                      : gameTimeLeft > 30
                        ? 'text-yellow-400'
                        : 'text-red-400 animate-pulse'
                    : 'text-gray-500'
                    }`}>
                    {isGameActive
                      ? `${Math.floor(gameTimeLeft / 60)}:${String(gameTimeLeft % 60).padStart(2, '0')}`
                      : "1:57"
                    }
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {isGameActive ? 'Game Time Remaining' : 'Total Game Duration'}
                  </div>
                </div>
              </div>
            </div>

            {/* Game Control Buttons */}
            <div className="space-y-3 mb-6">
              <button
                onClick={handleTimerStart}
                disabled={!canStartGame || isGameActive}
                className={`w-full px-4 py-3 rounded-lg font-semibold transition-colors ${canStartGame && !isGameActive
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  }`}
              >
                🟦 START GAME
              </button>

              <button
                onClick={() => setIsGameActive(false)}
                disabled={!isGameActive}
                className={`w-full px-4 py-3 rounded-lg font-semibold transition-colors ${isGameActive
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  }`}
              >
                🟥 PAUSE
              </button>

              <button
                onClick={handleGameReset}
                className="w-full px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg font-semibold transition-colors"
              >
                🟧 RESET
              </button>

              <button
                onClick={handleShowLeaderboard}
                className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-colors"
              >
                🟩 LEADERBOARD
              </button>
            </div>

            {/* Pose Reference Display */}
            {/* key: remount per pose so a missing image never sticks to the next one */}
            <PoseDisplay pose={selectedPose} key={selectedPose?.name || selectedPose} />

            {/* Game Progress Indicator */}
            {isGameActive && (
              <div className="text-center mb-4">
                <div className="text-sm text-gray-400 mb-1">Pose Progress</div>
                <div className="text-lg font-bold text-white">
                  {currentPoseIndex + 1} / {poses.length}
                </div>
                <div className="text-xs text-gray-500">
                  ({poses.length - currentPoseIndex - 1} poses remaining)
                </div>
              </div>
            )}

            {/* Hidden Timer Component - maintains timing logic */}
            <div className="hidden">
              <Timer
                duration={timerDuration}
                onStart={handleTimerStart}
                onComplete={handleTimerComplete}
                onTick={handleTimerTick}
                isRunning={isGameActive}
                canStart={canStartGame}
              />
            </div>
          </div>

          {/* Right Panel - Player 2 */}
          <div className="bg-gray-800 rounded-lg border border-gray-600 p-4">
            <div className="mb-4">
              <h2 className="text-xl font-bold text-red-400 mb-2">{player2Name.toUpperCase()}</h2>

              {/* Camera Selection */}
              <div className="mb-3">
                <label className="block text-sm font-medium text-gray-300 mb-1">Camera Source:</label>
                <select
                  value={player2Camera || ''}
                  onChange={(e) => setPlayer2Camera(e.target.value)}
                  className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                >
                  <option value="">Select Camera</option>
                  {availableCameras.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              </div>

              {/* Recording Status */}
              <div className="flex items-center space-x-2 mb-3">
                <div className={`w-3 h-3 rounded-full ${camerasActive && isGameActive ? 'bg-red-500 animate-pulse' : 'bg-gray-500'
                  }`}></div>
                <span className="text-sm text-gray-300">
                  {camerasActive && isGameActive ? 'Recording' : 'Standby'}
                </span>
              </div>
            </div>

            {/* Camera View */}
            <div className="mb-4">
              <WebcamCapture
                onCapture={handleFrameCapture}
                isActive={camerasActive}
                playerName={player2Name}
                deviceId={player2Camera}
                onDeviceChange={setPlayer2Camera}
                availableDevices={availableCameras}
                compact={true}
              />
            </div>

            {/* Score */}
            <div className="mb-4">
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">Current Pose</div>
                <div className="text-3xl font-bold text-red-400">
                  {player2Score ? player2Score.toFixed(1) : '0'}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Total: {player2Scores.reduce((sum, score) => sum + score, 0).toFixed(1)}
                </div>
              </div>
            </div>

            {/* Recent Scores */}
            <div>
              <div className="text-sm text-gray-400 mb-2">Recent Scores</div>
              <div className="bg-gray-700 rounded p-2 min-h-[80px] max-h-[120px] overflow-y-auto">
                {player2Scores.length > 0 ? (
                  <div className="space-y-1">
                    {player2Scores.slice(-5).map((score, index) => (
                      <div key={index} className="text-sm text-gray-300">
                        Round {player2Scores.length - 4 + index}: {score.toFixed(3)}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">No scores yet</div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Instructions */}
        <div className="mt-6 text-center">
          <p className="text-lg font-medium text-yellow-400">
            {selectedPose ?
              `Match the ${selectedPose?.name || selectedPose} shown above! Higher accuracy = Higher score!` :
              'Select a pose to begin the challenge!'
            }
          </p>
        </div>

        {/* Leaderboard Modal */}
        {showLeaderboard && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 border border-gray-600">
              <h3 className="text-xl font-bold text-white mb-4">🏆 Leaderboard</h3>
              <div className="space-y-2">
                {/* Calculate and display leaderboard */}
                {(() => {
                  const player1Avg = getAverageScore(player1Scores) || 0;
                  const player2Avg = getAverageScore(player2Scores) || 0;
                  const players = [
                    { name: player1Name, score: player1Avg, color: 'text-cyan-400' },
                    { name: player2Name, score: player2Avg, color: 'text-red-400' }
                  ].sort((a, b) => b.score - a.score);

                  return players.map((player, index) => (
                    <div key={player.name} className="flex justify-between items-center p-2 bg-gray-700 rounded">
                      <span className={`font-semibold ${player.color}`}>
                        #{index + 1} {player.name}
                      </span>
                      <span className="text-white">{player.score.toFixed(3)}</span>
                    </div>
                  ));
                })()}
              </div>
              <button
                onClick={handleCloseLeaderboard}
                className="mt-4 w-full px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        )}

        {/* Settings Panel (hidden by default, can be toggled) */}
        {showSettings && (
          <div className="mt-6 bg-gray-800 border border-gray-600 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Game Settings</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Timer Duration (seconds)</label>
                <input
                  type="number"
                  min="10"
                  max="300"
                  value={timerDuration}
                  onChange={(e) => setTimerDuration(parseInt(e.target.value))}
                  className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white"
                />
              </div>
            </div>
          </div>
        )}

        {/* Player Name Modal */}
        {showNameModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-600 max-w-md w-full mx-4">
              <h3 className="text-xl font-bold text-white mb-4">Enter Player Names</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-cyan-400 mb-2">Player 1 Name:</label>
                  <input
                    type="text"
                    value={player1Name}
                    onChange={(e) => setPlayer1Name(e.target.value || 'Player 1')}
                    className="w-full p-3 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-400 focus:outline-none"
                    placeholder="Enter Player 1 name"
                    maxLength={20}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-red-400 mb-2">Player 2 Name:</label>
                  <input
                    type="text"
                    value={player2Name}
                    onChange={(e) => setPlayer2Name(e.target.value || 'Player 2')}
                    className="w-full p-3 bg-gray-700 text-white rounded border border-gray-600 focus:border-red-400 focus:outline-none"
                    placeholder="Enter Player 2 name"
                    maxLength={20}
                  />
                </div>
              </div>

              <div className="flex space-x-3 mt-6">
                <button
                  onClick={() => setShowNameModal(false)}
                  className="flex-1 px-4 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-semibold transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={startGameWithNames}
                  className="flex-1 px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-colors"
                >
                  Start Game
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Game Result Modal */}
        {showGameResultModal && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg border border-gray-600 p-8 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
              {/* Modal Header */}
              <div className="flex justify-between items-start mb-6">
                <div className="text-center flex-1">
                  <h2 className="text-3xl font-bold text-yellow-400 mb-2">🏆 GAME COMPLETE! 🏆</h2>
                  <div className="text-xl font-semibold text-white">
                    {winner === 'Tie' ? "It's a Tie!" : `🎉 ${winner} Wins! 🎉`}
                  </div>
                </div>
                <button
                  onClick={() => {
                    setShowGameResultModal(false);
                    // Reset game state for new game
                    setGamePhase('setup');
                    setWinner(null);
                  }}
                  className="text-gray-400 hover:text-white text-2xl font-bold ml-4"
                  title="Close"
                >
                  ×
                </button>
              </div>

              {/* Score Display */}
              <div className="grid grid-cols-2 gap-6 mb-6">
                <div className="text-center p-6 bg-cyan-900 rounded-lg border-2 border-cyan-600">
                  <div className="text-lg font-semibold text-gray-200 mb-2">{player1Name}</div>
                  <div className="text-4xl font-bold text-cyan-400 mb-1">{player1TotalScore.toFixed(1)}</div>
                  <div className="text-sm text-gray-400">Total Score</div>
                  <div className="mt-3 text-xs text-gray-500">
                    {player1Scores.length} poses completed
                  </div>
                </div>
                <div className="text-center p-6 bg-red-900 rounded-lg border-2 border-red-600">
                  <div className="text-lg font-semibold text-gray-200 mb-2">{player2Name}</div>
                  <div className="text-4xl font-bold text-red-400 mb-1">{player2TotalScore.toFixed(1)}</div>
                  <div className="text-sm text-gray-400">Total Score</div>
                  <div className="mt-3 text-xs text-gray-500">
                    {player2Scores.length} poses completed
                  </div>
                </div>
              </div>

              {/* Congratulations Message */}
              <div className="text-center mb-6 p-4 bg-gray-700 rounded-lg">
                <p className="text-green-400 font-semibold text-lg mb-2">
                  🎊 Congratulations {winner === 'Tie' ? 'to both players' : winner}! 🎊
                </p>
                <p className="text-gray-400">
                  Game completed all {poses.length} poses in ~2 minutes
                </p>
                <p className="text-gray-500 text-sm mt-2">
                  You both gave an amazing performance!
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-center space-x-4">
                <button
                  onClick={() => {
                    setShowGameResultModal(false);
                    handleGameReset();
                  }}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
                >
                  🔄 Play Again
                </button>
                <button
                  onClick={() => {
                    setShowGameResultModal(false);
                    setGamePhase('setup');
                    setWinner(null);
                  }}
                  className="px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-semibold transition-colors"
                >
                  ✅ Close
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
