import React, { useState } from 'react';

import { utils } from '../services/api';

/**
 * Shows the reference image for the pose the players have to match.
 *
 * The fallback is rendered by React rather than by overwriting the container's
 * innerHTML: mutating the DOM behind React's back detaches the <img> node, so
 * every later pose would keep showing the placeholder and unmounting the block
 * (RESET) would fail on a node React no longer owns.
 */
function PoseDisplay({ pose }) {
  const poseName = pose?.name || pose;
  const imagePath = utils.getPoseImagePath(poseName);
  const [hasImageError, setHasImageError] = useState(false);

  if (!poseName) return null;

  const showFallback = !imagePath || hasImageError;

  return (
    <div className="mb-6">
      <div className="text-center mb-3">
        <div className="text-sm text-gray-400 mb-1">Target Pose</div>
        <div className="text-lg font-semibold text-white">{poseName}</div>
      </div>
      <div className="flex justify-center">
        <div className="relative bg-gray-700 rounded-lg p-4 border-2 border-yellow-500">
          {showFallback ? (
            <div className="flex flex-col items-center justify-center bg-gray-600 rounded p-4 min-h-[320px] min-w-[320px] text-center">
              <div className="text-4xl mb-2">🤸‍♂️</div>
              <div className="text-white font-semibold text-sm">{utils.formatPoseName(poseName)}</div>
              <div className="text-gray-300 text-xs mt-1">Pose Reference</div>
            </div>
          ) : (
            <img
              src={imagePath}
              alt={`${poseName} Reference Pose`}
              className="max-w-[400px] max-h-[400px] object-contain rounded"
              style={{
                backgroundColor: 'rgba(0,0,0,0.1)',
                minHeight: '320px',
                minWidth: '320px'
              }}
              onError={() => setHasImageError(true)}
            />
          )}
          <div className="absolute -top-2 -right-2 bg-yellow-500 text-black text-xs px-2 py-1 rounded-full font-semibold">
            Reference
          </div>
        </div>
      </div>
      <div className="text-center mt-2">
        <p className="text-sm text-yellow-400">
          Match this pose for maximum score!
        </p>
      </div>
    </div>
  );
}

export default PoseDisplay;
