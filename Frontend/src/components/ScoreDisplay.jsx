import React from 'react';
import { TrendingUp, TrendingDown, Minus, Trophy, Target } from 'lucide-react';

const ScoreDisplay = ({ 
  player1Score, 
  player2Score, 
  averageScore1, 
  averageScore2,
  isActive,
  winner = null 
}) => {
  const getScoreColor = (score) => {
    if (score >= 0.9) return 'text-green-400';
    if (score >= 0.7) return 'text-yellow-400';
    if (score >= 0.5) return 'text-orange-400';
    return 'text-red-400';
  };

  const getScoreBackground = (score) => {
    if (score >= 0.9) return 'bg-green-900 border-green-600';
    if (score >= 0.7) return 'bg-yellow-900 border-yellow-600';
    if (score >= 0.5) return 'bg-orange-900 border-orange-600';
    return 'bg-red-900 border-red-600';
  };

  const getTrendIcon = (current, average) => {
    if (!average || !current) return <Minus size={16} className="text-gray-400" />;
    
    const diff = current - average;
    if (diff > 5) return <TrendingUp size={16} className="text-green-400" />;
    if (diff < -5) return <TrendingDown size={16} className="text-red-400" />;
    return <Minus size={16} className="text-gray-400" />;
  };

  const formatScore = (score) => {
    if (typeof score !== 'number') return '--';
    return score.toFixed(1);
  };

  return (
    <div className="w-full max-w-4xl bg-gray-800 rounded-lg p-6 border border-gray-600">
      {/* Header */}
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2 flex items-center justify-center">
          <Target className="mr-2" size={24} />
          Pose Scoring
        </h2>
        {winner && (
          <div className="flex items-center justify-center text-yellow-400 font-semibold">
            <Trophy className="mr-2" size={20} />
            Winner: {winner}!
          </div>
        )}
      </div>

      {/* Score Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Player 1 Score */}
        <div className={`p-4 rounded-lg border-2 transition-all ${
          isActive ? 'border-blue-500 bg-blue-900/20' : 'border-gray-600 bg-gray-700'
        }`}>
          <div className="text-center">
            <h3 className="text-lg font-semibold text-white mb-2">Player 1</h3>
            <div className={`text-4xl font-bold mb-2 ${getScoreColor(player1Score)}`}>
              {formatScore(player1Score)}
            </div>
            
            {/* Current Score Indicator */}
            <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium border ${
              getScoreBackground(player1Score)
            }`}>
              Current Score
            </div>

            {/* Average and Trend */}
            {averageScore1 && (
              <div className="mt-3 flex items-center justify-center space-x-2 text-sm">
                <span className="text-gray-300">Avg: {formatScore(averageScore1)}</span>
                {getTrendIcon(player1Score, averageScore1)}
              </div>
            )}
          </div>
        </div>

        {/* Player 2 Score */}
        <div className={`p-4 rounded-lg border-2 transition-all ${
          isActive ? 'border-blue-500 bg-blue-900/20' : 'border-gray-600 bg-gray-700'
        }`}>
          <div className="text-center">
            <h3 className="text-lg font-semibold text-white mb-2">Player 2</h3>
            <div className={`text-4xl font-bold mb-2 ${getScoreColor(player2Score)}`}>
              {formatScore(player2Score)}
            </div>
            
            {/* Current Score Indicator */}
            <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium border ${
              getScoreBackground(player2Score)
            }`}>
              Current Score
            </div>

            {/* Average and Trend */}
            {averageScore2 && (
              <div className="mt-3 flex items-center justify-center space-x-2 text-sm">
                <span className="text-gray-300">Avg: {formatScore(averageScore2)}</span>
                {getTrendIcon(player2Score, averageScore2)}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Comparison Section */}
      {(player1Score || player1Score === 0) && (player2Score || player2Score === 0) && (
        <div className="bg-gray-700 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-3 text-center">Comparison</h4>
          
          {/* Score Difference */}
          <div className="flex justify-center items-center mb-3">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">
                {Math.abs(player1Score - player2Score).toFixed(1)}
              </div>
              <div className="text-sm text-gray-300">Score Difference</div>
            </div>
          </div>

          {/* Visual Comparison Bar */}
          <div className="relative bg-gray-600 rounded-full h-4 overflow-hidden">
            <div 
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500"
              style={{ width: `${player1Score || 0}%` }}
            ></div>
            <div 
              className="absolute top-0 right-0 h-full bg-gradient-to-l from-green-500 to-green-600 transition-all duration-500"
              style={{ width: `${player2Score || 0}%` }}
            ></div>
          </div>
          
          <div className="flex justify-between mt-2 text-xs text-gray-400">
            <span>Player 1: {formatScore(player1Score)}</span>
            <span>Player 2: {formatScore(player2Score)}</span>
          </div>
        </div>
      )}

      {/* Status */}
      <div className="mt-4 text-center">
        <div className={`inline-block px-4 py-2 rounded-full text-sm font-medium ${
          isActive 
            ? 'bg-green-900 text-green-200 border border-green-600' 
            : 'bg-gray-600 text-gray-300'
        }`}>
          {isActive ? '🟢 Live Scoring' : '⭕ Scoring Inactive'}
        </div>
      </div>

      {/* Score Legend */}
      <div className="mt-4 bg-gray-700 rounded-lg p-3">
        <h5 className="text-white font-medium mb-2 text-sm">Score Guide:</h5>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-400 rounded mr-2"></div>
            <span className="text-gray-300">90-100 Excellent</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-yellow-400 rounded mr-2"></div>
            <span className="text-gray-300">70-89 Good</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-orange-400 rounded mr-2"></div>
            <span className="text-gray-300">50-69 Fair</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-400 rounded mr-2"></div>
            <span className="text-gray-300">0-49 Needs Work</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScoreDisplay;