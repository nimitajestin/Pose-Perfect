import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Timer as TimerIcon } from 'lucide-react';

const Timer = ({
  duration = 30,
  onStart,
  onStop,
  onComplete,
  onTick,
  isRunning: externalIsRunning,
  canStart = true
}) => {
  const [timeLeft, setTimeLeft] = useState(duration);
  const [isRunning, setIsRunning] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  // Sync with external running state
  useEffect(() => {
    if (typeof externalIsRunning === 'boolean') {
      setIsRunning(externalIsRunning);
    }
  }, [externalIsRunning]);

  useEffect(() => {
    let interval = null;

    if (isRunning && timeLeft > 0) {
      interval = setInterval(() => {
        setTimeLeft((prevTime) => {
          const newTime = prevTime - 1;

          // Call onTick callback
          if (onTick) {
            onTick(newTime, duration - newTime);
          }

          // Check if timer completed
          if (newTime <= 0) {
            setIsRunning(false);
            if (onComplete) {
              onComplete();
            }
            return 0;
          }

          return newTime;
        });
      }, 1000);
    } else if (interval) {
      clearInterval(interval);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isRunning, timeLeft, onTick, onComplete, duration]);

  const handleStart = () => {
    if (!canStart || timeLeft <= 0) return;

    setIsRunning(true);
    setHasStarted(true);
    if (onStart) {
      onStart();
    }
  };

  const handlePause = () => {
    setIsRunning(false);
    if (onStop) {
      onStop();
    }
  };

  const handleReset = () => {
    setIsRunning(false);
    setTimeLeft(duration);
    setHasStarted(false);
    if (onStop) {
      onStop();
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getProgressPercentage = () => {
    return ((duration - timeLeft) / duration) * 100;
  };

  const getTimerColor = () => {
    const percentage = (timeLeft / duration) * 100;
    if (percentage > 50) return 'text-green-400';
    if (percentage > 20) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getProgressColor = () => {
    const percentage = (timeLeft / duration) * 100;
    if (percentage > 50) return 'bg-green-400';
    if (percentage > 20) return 'bg-yellow-400';
    return 'bg-red-400';
  };

  return (
    <div className="w-full max-w-md bg-gray-800 rounded-lg p-6 border border-gray-600">
      <div className="text-center mb-4">
        <h3 className="text-white font-semibold mb-2 flex items-center justify-center">
          <TimerIcon className="mr-2" size={20} />
          Timer
        </h3>

        {/* Timer Display */}
        <div className={`text-6xl font-mono font-bold mb-4 ${getTimerColor()}`}>
          {formatTime(timeLeft)}
        </div>

        {/* Circular Timer Display */}
        <div className="relative w-32 h-32 mx-auto mb-4">
          <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
            {/* Background circle */}
            <circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke="rgb(75, 85, 99)"
              strokeWidth="8"
            />
            {/* Progress circle */}
            <circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke={getProgressColor().replace('bg-', '').replace('green-400', 'rgb(74, 222, 128)').replace('yellow-400', 'rgb(250, 204, 21)').replace('red-400', 'rgb(248, 113, 113)')}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 50}`}
              strokeDashoffset={`${2 * Math.PI * 50 * (1 - ((duration - timeLeft) / duration))}`}
              className="transition-all duration-1000"
            />
          </svg>
          {/* Time remaining in center */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`text-xl font-bold ${getTimerColor()}`}>
              {timeLeft}s
            </span>
          </div>
        </div>

        {/* Status */}
        <div className="text-sm text-gray-400 mb-4">
          {timeLeft <= 0 ? (
            <span className="text-green-400 font-semibold">Time's Up!</span>
          ) : isRunning ? (
            <span className="text-blue-400">Running...</span>
          ) : hasStarted ? (
            <span className="text-yellow-400">Paused</span>
          ) : (
            <span>Ready to start</span>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center space-x-3">
        {!isRunning ? (
          <button
            onClick={handleStart}
            disabled={!canStart || timeLeft <= 0}
            className={`px-4 py-2 rounded-lg flex items-center space-x-2 font-semibold transition-colors ${canStart && timeLeft > 0
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
              }`}
          >
            <Play size={16} />
            <span>{hasStarted ? 'Resume' : 'Start'}</span>
          </button>
        ) : (
          <button
            onClick={handlePause}
            className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg flex items-center space-x-2 font-semibold transition-colors"
          >
            <Pause size={16} />
            <span>Pause</span>
          </button>
        )}

        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg flex items-center space-x-2 font-semibold transition-colors"
        >
          <RotateCcw size={16} />
          <span>Reset</span>
        </button>
      </div>

      {/* Additional Info */}
      {hasStarted && (
        <div className="mt-4 text-center text-sm text-gray-400">
          <p>Elapsed: {formatTime(duration - timeLeft)}</p>
        </div>
      )}
    </div>
  );
};

export default Timer;