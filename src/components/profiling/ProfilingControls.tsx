'use client';

import React, { useState } from 'react';
import { useProfilingContext } from './ProfilingContext';
import type { ProfiledGenerateRequest } from '@/types';

export function ProfilingControls() {
  const { isRunning, isProfiling, connectionState, startProfiling, stopProfiling } = useProfilingContext();

  // Form state
  const [modelPath, setModelPath] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('');
  const [profilingDepth, setProfilingDepth] = useState<'module' | 'deep'>('module');
  const [tags, setTags] = useState<string>('');
  const [experimentName, setExperimentName] = useState<string>('');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxLength, setMaxLength] = useState<number>(100);

  // Handle start profiling
  const handleStartProfiling = async () => {
    if (!prompt.trim()) {
      alert('Please enter a prompt');
      return;
    }

    const request: ProfiledGenerateRequest = {
      prompt: prompt.trim(),
      profiling_depth: profilingDepth,
      temperature,
      max_length: maxLength,
    };

    // Add optional fields if provided
    if (modelPath.trim()) {
      request.model_path = modelPath.trim();
    }

    if (experimentName.trim()) {
      request.experiment_name = experimentName.trim();
    }

    if (tags.trim()) {
      request.tags = tags.split(',').map((t) => t.trim()).filter((t) => t.length > 0);
    }

    await startProfiling(request);
  };

  // Handle stop profiling
  const handleStopProfiling = () => {
    stopProfiling();
  };

  // Determine status text and color
  const getStatusDisplay = () => {
    if (isProfiling) {
      return { text: 'Profiling...', color: 'text-green-600 dark:text-green-400' };
    }
    if (connectionState === 'connecting') {
      return { text: 'Connecting...', color: 'text-yellow-600 dark:text-yellow-400' };
    }
    if (connectionState === 'connected') {
      return { text: 'Connected', color: 'text-blue-600 dark:text-blue-400' };
    }
    return { text: 'Ready', color: 'text-gray-600 dark:text-gray-400' };
  };

  const status = getStatusDisplay();

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
      <div className="space-y-4">
        {/* Status Indicator */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Profiling Controls
          </h3>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isProfiling ? 'bg-green-500 animate-pulse' : 'bg-gray-300 dark:bg-gray-600'}`} />
            <span className={`text-sm font-medium ${status.color}`}>
              {status.text}
            </span>
          </div>
        </div>

        {/* Model Selector */}
        <div>
          <label htmlFor="model-path" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Model Path (optional)
          </label>
          <input
            id="model-path"
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            disabled={isRunning}
            placeholder="e.g., /path/to/model or huggingface-model-id"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Leave empty to use default model
          </p>
        </div>

        {/* Prompt Input */}
        <div>
          <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Prompt <span className="text-red-500">*</span>
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={isRunning}
            rows={3}
            placeholder="Enter the text prompt for inference..."
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          />
        </div>

        {/* Profiling Depth Selector */}
        <div>
          <label htmlFor="profiling-depth" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Profiling Depth
          </label>
          <select
            id="profiling-depth"
            value={profilingDepth}
            onChange={(e) => setProfilingDepth(e.target.value as 'module' | 'deep')}
            disabled={isRunning}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          >
            <option value="module">Module (faster, less overhead)</option>
            <option value="deep">Deep (detailed operation profiling)</option>
          </select>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Module: Layer-level profiling only. Deep: Includes attention, MLP, and LayerNorm internals.
          </p>
        </div>

        {/* Advanced Settings - Collapsible */}
        <details className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100">
            Advanced Settings
          </summary>
          <div className="mt-4 space-y-4">
            {/* Experiment Name */}
            <div>
              <label htmlFor="experiment-name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Experiment Name
              </label>
              <input
                id="experiment-name"
                type="text"
                value={experimentName}
                onChange={(e) => setExperimentName(e.target.value)}
                disabled={isRunning}
                placeholder="e.g., baseline-test"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>

            {/* Tags */}
            <div>
              <label htmlFor="tags" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Tags
              </label>
              <input
                id="tags"
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                disabled={isRunning}
                placeholder="e.g., benchmark, optimization, baseline (comma-separated)"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>

            {/* Temperature */}
            <div>
              <label htmlFor="temperature" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Temperature: {temperature}
              </label>
              <input
                id="temperature"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                disabled={isRunning}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>0 (deterministic)</span>
                <span>2 (creative)</span>
              </div>
            </div>

            {/* Max Length */}
            <div>
              <label htmlFor="max-length" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Max Tokens
              </label>
              <input
                id="max-length"
                type="number"
                min="1"
                max="4096"
                value={maxLength}
                onChange={(e) => setMaxLength(parseInt(e.target.value, 10) || 100)}
                disabled={isRunning}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
          </div>
        </details>

        {/* Action Buttons */}
        <div className="flex space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          {!isRunning ? (
            <button
              onClick={handleStartProfiling}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md shadow transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Start Profiling
            </button>
          ) : (
            <button
              onClick={handleStopProfiling}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-md shadow transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
            >
              Stop Profiling
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
