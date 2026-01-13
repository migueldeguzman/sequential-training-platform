'use client';

import React, { useState, useEffect, useRef } from 'react';
import { TokenMetrics } from '@/types';

interface TokenSliderProps {
  tokens: TokenMetrics[];
  currentTokenIndex: number;
  onTokenChange: (index: number) => void;
}

export default function TokenSlider({
  tokens,
  currentTokenIndex,
  onTokenChange
}: TokenSliderProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1); // 1x, 2x, 4x
  const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, []);

  // Handle play/pause logic
  useEffect(() => {
    if (isPlaying) {
      playbackIntervalRef.current = setInterval(() => {
        const nextIndex = currentTokenIndex + 1;
        if (nextIndex >= tokens.length) {
          setIsPlaying(false);
        } else {
          onTokenChange(nextIndex);
        }
      }, 500 / playbackSpeed); // Base 500ms per token
    } else {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
        playbackIntervalRef.current = null;
      }
    }

    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, currentTokenIndex, tokens.length, onTokenChange]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newIndex = parseInt(e.target.value, 10);
    onTokenChange(newIndex);
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const cycleSpeed = () => {
    setPlaybackSpeed((prev) => {
      if (prev === 1) return 2;
      if (prev === 2) return 4;
      return 1;
    });
  };

  const jumpToStart = () => {
    onTokenChange(0);
    setIsPlaying(false);
  };

  const jumpToEnd = () => {
    onTokenChange(tokens.length - 1);
    setIsPlaying(false);
  };

  if (tokens.length === 0) {
    return (
      <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
        <p className="text-gray-400 text-sm">No tokens available</p>
      </div>
    );
  }

  const currentToken = tokens[currentTokenIndex];

  return (
    <div className="bg-gray-900 p-4 rounded-lg border border-gray-700 space-y-4">
      {/* Token Information Display */}
      <div className="bg-gray-800 p-3 rounded border border-gray-700">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-gray-400 mb-1">Token Position</div>
            <div className="text-lg font-mono text-white">
              {currentTokenIndex + 1} / {tokens.length}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Token Text</div>
            <div className="text-lg font-mono text-blue-400 truncate">
              &quot;{currentToken.token_text}&quot;
            </div>
          </div>
        </div>

        {/* Token Metrics */}
        <div className="grid grid-cols-4 gap-3 mt-3 pt-3 border-t border-gray-700">
          <div>
            <div className="text-xs text-gray-400 mb-1">Duration</div>
            <div className="text-sm font-mono text-green-400">
              {currentToken.duration_ms.toFixed(2)} ms
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Energy</div>
            <div className="text-sm font-mono text-yellow-400">
              {currentToken.energy_mj.toFixed(2)} mJ
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Power</div>
            <div className="text-sm font-mono text-orange-400">
              {currentToken.power_snapshot_mw.toFixed(1)} mW
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Phase</div>
            <div className="text-sm font-mono text-blue-400 capitalize">
              {currentToken.phase}
            </div>
          </div>
        </div>
      </div>

      {/* Slider Control */}
      <div className="space-y-2">
        <input
          type="range"
          min="0"
          max={tokens.length - 1}
          value={currentTokenIndex}
          onChange={handleSliderChange}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          style={{
            background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
              (currentTokenIndex / (tokens.length - 1)) * 100
            }%, #374151 ${
              (currentTokenIndex / (tokens.length - 1)) * 100
            }%, #374151 100%)`
          }}
        />

        {/* Position Markers */}
        <div className="flex justify-between text-xs text-gray-500 px-1">
          <span>0</span>
          <span>{Math.floor(tokens.length / 2)}</span>
          <span>{tokens.length - 1}</span>
        </div>
      </div>

      {/* Playback Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <button
            onClick={jumpToStart}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded border border-gray-600 transition-colors"
            title="Jump to start"
          >
            <svg
              className="w-5 h-5 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
              />
            </svg>
          </button>

          <button
            onClick={togglePlayPause}
            className="p-3 bg-blue-600 hover:bg-blue-700 rounded border border-blue-500 transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <svg
                className="w-6 h-6 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg
                className="w-6 h-6 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          <button
            onClick={jumpToEnd}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded border border-gray-600 transition-colors"
            title="Jump to end"
          >
            <svg
              className="w-5 h-5 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 5l7 7-7 7M5 5l7 7-7 7"
              />
            </svg>
          </button>
        </div>

        <button
          onClick={cycleSpeed}
          className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded border border-gray-600 transition-colors text-sm font-mono text-gray-300"
          title="Change playback speed"
        >
          {playbackSpeed}x
        </button>
      </div>

      {/* Progress Indicator */}
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>
          Progress: {Math.round((currentTokenIndex / (tokens.length - 1)) * 100)}%
        </span>
        <span>
          {isPlaying ? (
            <span className="flex items-center text-green-400">
              <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse" />
              Playing
            </span>
          ) : (
            <span className="text-gray-500">Paused</span>
          )}
        </span>
      </div>
    </div>
  );
}
