'use client';

import React from 'react';
import { ProfilingControls } from './ProfilingControls';
import { PowerTimeSeriesChart } from './charts/PowerTimeSeriesChart';
import { LiveLayerHeatmap } from './charts/LiveLayerHeatmap';
import { TokenGenerationStream } from './TokenGenerationStream';
import CurrentOperationIndicator from './CurrentOperationIndicator';
import { EnergyPredictionCard } from './EnergyPredictionCard';
import { useProfilingContext } from './ProfilingContext';

/**
 * RealTimeView Container
 *
 * Composes all real-time profiling components into a single view
 * with responsive layout:
 * - Controls at top
 * - Power time series and layer heatmap in main area
 * - Token generation stream in sidebar
 * - Current operation indicator at footer
 */
export function RealTimeView() {
  const { isProfiling, powerSamples, tokens } = useProfilingContext();

  return (
    <div className="space-y-6">
      {/* Controls Section - Top */}
      <ProfilingControls />

      {/* Main Content Area - Grid Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Left Column - Charts (3/4 width on xl screens) */}
        <div className="xl:col-span-3 space-y-6">
          {/* Power Time Series Chart - Main Area */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Power Consumption Over Time
            </h3>
            {powerSamples.length > 0 ? (
              <PowerTimeSeriesChart samples={powerSamples} autoScroll={true} />
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
                {isProfiling ? (
                  <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-2"></div>
                    <p>Waiting for power samples...</p>
                  </div>
                ) : (
                  <p>Start profiling to see real-time power consumption</p>
                )}
              </div>
            )}
          </div>

          {/* Layer Heatmap - Below Charts */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Layer Activity Heatmap
            </h3>
            {tokens.length > 0 ? (
              <LiveLayerHeatmap latestToken={tokens[tokens.length - 1]} metric="energy" />
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
                {isProfiling ? (
                  <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500 mb-2"></div>
                    <p>Waiting for token generation...</p>
                  </div>
                ) : (
                  <p>Start profiling to see layer-by-layer activity</p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Token Stream + Prediction Sidebar (1/4 width on xl screens) */}
        <div className="xl:col-span-1 space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 sticky top-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Token Generation
            </h3>
            {tokens.length > 0 || isProfiling ? (
              <TokenGenerationStream />
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400 text-center px-4">
                <p>Tokens will appear here as they are generated during inference</p>
              </div>
            )}
          </div>

          {/* Energy Prediction - Shows when not profiling */}
          {!isProfiling && (
            <EnergyPredictionCard />
          )}
        </div>
      </div>

      {/* Current Operation Indicator - Footer */}
      <div className="sticky bottom-0 z-10">
        <CurrentOperationIndicator />
      </div>

      {/* Info Panel - Shows when not profiling */}
      {!isProfiling && powerSamples.length === 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg
                className="h-6 w-6 text-blue-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300">
                Real-Time Energy Profiling
              </h3>
              <div className="mt-2 text-sm text-blue-700 dark:text-blue-400">
                <p className="mb-2">
                  This view shows live profiling data as inference runs. You&apos;ll see:
                </p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>
                    <strong>Power Timeline:</strong> Real-time CPU, GPU, ANE, and DRAM power consumption
                  </li>
                  <li>
                    <strong>Layer Heatmap:</strong> Component-level activity updating with each token
                  </li>
                  <li>
                    <strong>Token Stream:</strong> Generated tokens color-coded by energy consumption
                  </li>
                  <li>
                    <strong>Operation Indicator:</strong> Current inference phase and section being profiled
                  </li>
                </ul>
                <p className="mt-3">
                  Configure your settings above and click <strong>Start Profiling</strong> to begin.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Stats - Shows during/after profiling */}
      {(isProfiling || powerSamples.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Tokens Generated
            </div>
            <div className="mt-1 text-2xl font-semibold text-gray-900 dark:text-gray-100">
              {tokens.length}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Power Samples
            </div>
            <div className="mt-1 text-2xl font-semibold text-gray-900 dark:text-gray-100">
              {powerSamples.length}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Avg Power
            </div>
            <div className="mt-1 text-2xl font-semibold text-gray-900 dark:text-gray-100">
              {powerSamples.length > 0
                ? `${(powerSamples.reduce((sum, s) => sum + s.total_power_mw, 0) / powerSamples.length / 1000).toFixed(2)}W`
                : '0W'}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Duration
            </div>
            <div className="mt-1 text-2xl font-semibold text-gray-900 dark:text-gray-100">
              {powerSamples.length > 0
                ? `${((powerSamples[powerSamples.length - 1].timestamp - powerSamples[0].timestamp) / 1000).toFixed(1)}s`
                : '0s'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
