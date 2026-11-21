import React, { useState, useEffect } from 'react';
import { ChevronDown, Check, Loader2 } from 'lucide-react';

const PoseSelector = ({ poses, selectedPose, onPoseSelect, isLoading, error }) => {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Auto-select first pose if none selected
    if (!selectedPose && poses && poses.length > 0) {
      onPoseSelect(poses[0]);
    }
  }, [poses, selectedPose, onPoseSelect]);

  if (isLoading) {
    return (
      <div className="w-full max-w-md">
        <label className="block text-white text-sm font-medium mb-2">
          Select Pose
        </label>
        <div className="flex items-center justify-center p-4 bg-gray-800 rounded-lg border border-gray-600">
          <Loader2 className="animate-spin mr-2" size={20} />
          <span className="text-gray-300">Loading poses...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full max-w-md">
        <label className="block text-white text-sm font-medium mb-2">
          Select Pose
        </label>
        <div className="p-4 bg-red-900 rounded-lg border border-red-600">
          <span className="text-red-200">Error: {error}</span>
        </div>
      </div>
    );
  }

  if (!poses || poses.length === 0) {
    return (
      <div className="w-full max-w-md">
        <label className="block text-white text-sm font-medium mb-2">
          Select Pose
        </label>
        <div className="p-4 bg-yellow-900 rounded-lg border border-yellow-600">
          <span className="text-yellow-200">No poses available</span>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md relative">
      <label className="block text-white text-sm font-medium mb-2">
        Select Pose
      </label>
      
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full p-3 bg-gray-800 text-white rounded-lg border border-gray-600 hover:border-blue-500 focus:border-blue-500 focus:outline-none transition-colors flex items-center justify-between"
        >
          <span className="flex items-center">
            {selectedPose ? (
              <>
                <Check className="mr-2 text-green-400" size={16} />
                {selectedPose.name || selectedPose}
              </>
            ) : (
              <span className="text-gray-400">Choose a pose...</span>
            )}
          </span>
          <ChevronDown 
            className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} 
            size={20} 
          />
        </button>

        {isOpen && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-gray-800 border border-gray-600 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
            {poses.map((pose) => (
              <button
                key={pose.name || pose}
                onClick={() => {
                  onPoseSelect(pose);
                  setIsOpen(false);
                }}
                className={`w-full p-3 text-left hover:bg-gray-700 transition-colors border-b border-gray-600 last:border-b-0 flex items-center ${
                  (selectedPose?.name || selectedPose) === (pose.name || pose)
                    ? 'bg-blue-900 text-blue-200'
                    : 'text-white'
                }`}
              >
                {(selectedPose?.name || selectedPose) === (pose.name || pose) && (
                  <Check className="mr-2 text-green-400" size={16} />
                )}
                <div>
                  <div className="font-medium">{pose.name || pose}</div>
                  {pose.description && (
                    <div className="text-sm text-gray-400 mt-1">
                      {pose.description}
                    </div>
                  )}
                  {typeof pose === 'object' && pose.accuracy && (
                    <div className="text-xs text-blue-400 mt-1">
                      Accuracy: {pose.accuracy}%
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {selectedPose && typeof selectedPose === 'object' && selectedPose.description && (
        <div className="mt-2 p-2 bg-gray-700 rounded text-sm text-gray-300">
          <strong>Description:</strong> {selectedPose.description}
        </div>
      )}
    </div>
  );
};

export default PoseSelector;