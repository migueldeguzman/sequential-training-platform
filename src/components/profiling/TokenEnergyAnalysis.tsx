'use client';

import React, { useState, useMemo } from 'react';
import { ProfilingRun } from '@/types';
import TokenEnergyHeatmap from './charts/TokenEnergyHeatmap';
import TokenEnergyDistribution from './charts/TokenEnergyDistribution';

interface TokenEnergyAnalysisProps {
  run: ProfilingRun;
}

export default function TokenEnergyAnalysis({ run }: TokenEnergyAnalysisProps) {
  const [selectedMetric, setSelectedMetric] = useState<'energy' | 'duration' | 'activation_mean' | 'sparsity'>('energy');
  const [energyThreshold, setEnergyThreshold] = useState<number | undefined>(undefined);
  const [showHighEnergy, setShowHighEnergy] = useState(false);
  const [showLowEnergy, setShowLowEnergy] = useState(false);

  const tokens = run.tokens || [];

  // Calculate high and low energy thresholds
  const { highEnergyThreshold, lowEnergyThreshold, highEnergyTokens, lowEnergyTokens } = useMemo(() => {
    if (!tokens.length) {
      return {
        highEnergyThreshold: 0,
        lowEnergyThreshold: 0,
        highEnergyTokens: [],
        lowEnergyTokens: [],
      };
    }

    const energies = tokens.map((t) => t.energy_mj);
    const mean = energies.reduce((sum, e) => sum + e, 0) / energies.length;
    const variance = energies.reduce((sum, e) => sum + Math.pow(e - mean, 2), 0) / energies.length;
    const stdDev = Math.sqrt(variance);

    // High energy: > mean + 1 std dev
    const highThreshold = mean + stdDev;
    // Low energy: < mean - 0.5 std dev
    const lowThreshold = Math.max(0, mean - 0.5 * stdDev);

    const high = tokens.filter((t) => t.energy_mj > highThreshold);
    const low = tokens.filter((t) => t.energy_mj < lowThreshold);

    return {
      highEnergyThreshold: highThreshold,
      lowEnergyThreshold: lowThreshold,
      highEnergyTokens: high,
      lowEnergyTokens: low,
    };
  }, [tokens]);

  // Apply threshold filter
  const effectiveThreshold = useMemo(() => {
    if (showHighEnergy) return highEnergyThreshold;
    if (showLowEnergy) return undefined; // Will filter in component
    return energyThreshold;
  }, [showHighEnergy, showLowEnergy, energyThreshold, highEnergyThreshold]);

  // Get display tokens based on filter
  const displayTokens = useMemo(() => {
    if (showLowEnergy) {
      return lowEnergyTokens;
    }
    return tokens;
  }, [showLowEnergy, tokens, lowEnergyTokens]);

  if (!tokens.length) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded border border-gray-200">
        <p className="text-gray-500">No token data available for this run</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Metric Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Heatmap Metric
            </label>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as 'energy' | 'duration' | 'activation_mean' | 'sparsity')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="energy">Energy (mJ)</option>
              <option value="duration">Duration (ms)</option>
              <option value="activation_mean">Activation Mean</option>
              <option value="sparsity">Sparsity</option>
            </select>
          </div>

          {/* Energy Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Energy Threshold (mJ)
            </label>
            <input
              type="number"
              step="0.1"
              value={energyThreshold || ''}
              onChange={(e) => setEnergyThreshold(e.target.value ? parseFloat(e.target.value) : undefined)}
              placeholder="No filter"
              disabled={showHighEnergy || showLowEnergy}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
            />
          </div>

          {/* Quick Filters */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Filters
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setShowHighEnergy(!showHighEnergy);
                  setShowLowEnergy(false);
                  setEnergyThreshold(undefined);
                }}
                className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  showHighEnergy
                    ? 'bg-red-100 text-red-800 border-2 border-red-500'
                    : 'bg-gray-100 text-gray-700 border border-gray-300 hover:bg-gray-200'
                }`}
              >
                High Energy
                <div className="text-xs text-gray-600 mt-0.5">
                  {highEnergyTokens.length} tokens
                </div>
              </button>
              <button
                onClick={() => {
                  setShowLowEnergy(!showLowEnergy);
                  setShowHighEnergy(false);
                  setEnergyThreshold(undefined);
                }}
                className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  showLowEnergy
                    ? 'bg-blue-100 text-blue-800 border-2 border-blue-500'
                    : 'bg-gray-100 text-gray-700 border border-gray-300 hover:bg-gray-200'
                }`}
              >
                Low Energy
                <div className="text-xs text-gray-600 mt-0.5">
                  {lowEnergyTokens.length} tokens
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Active Filter Info */}
        {(showHighEnergy || showLowEnergy || energyThreshold) && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-sm text-blue-800">
              {showHighEnergy && (
                <>Showing tokens with energy &gt; {highEnergyThreshold.toFixed(2)} mJ (mean + 1σ)</>
              )}
              {showLowEnergy && (
                <>Showing tokens with energy &lt; {lowEnergyThreshold.toFixed(2)} mJ (mean - 0.5σ)</>
              )}
              {energyThreshold && !showHighEnergy && !showLowEnergy && (
                <>Showing tokens with energy ≥ {energyThreshold.toFixed(2)} mJ</>
              )}
            </p>
          </div>
        )}
      </div>

      {/* Token Energy Distribution Histogram */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Token Energy Distribution
        </h3>
        <TokenEnergyDistribution tokens={tokens} />
      </div>

      {/* Token-Layer Energy Heatmap */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Token-Layer {selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)} Heatmap
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Visualizes {selectedMetric.replace('_', ' ')} consumption across tokens and transformer layers.
          Hover over cells for details.
        </p>
        <div className="overflow-x-auto">
          <TokenEnergyHeatmap
            tokens={displayTokens}
            metric={selectedMetric}
            energyThreshold={effectiveThreshold}
          />
        </div>
      </div>

      {/* High Energy Tokens List */}
      {highEnergyTokens.length > 0 && (
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            High Energy Tokens
            <span className="ml-2 text-sm font-normal text-gray-600">
              ({highEnergyTokens.length} tokens, &gt; {highEnergyThreshold.toFixed(2)} mJ)
            </span>
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Position
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Token
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Energy (mJ)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Duration (ms)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Phase
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {highEnergyTokens
                  .sort((a, b) => b.energy_mj - a.energy_mj)
                  .slice(0, 20)
                  .map((token) => (
                    <tr key={token.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                        {token.token_position}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-mono text-gray-900">
                        &quot;{token.token_text}&quot;
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-red-600 font-semibold">
                        {token.energy_mj.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                        {token.duration_ms.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                        {token.phase}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {highEnergyTokens.length > 20 && (
              <p className="mt-2 text-sm text-gray-500 text-center">
                Showing top 20 of {highEnergyTokens.length} high-energy tokens
              </p>
            )}
          </div>
        </div>
      )}

      {/* Low Energy Tokens List */}
      {lowEnergyTokens.length > 0 && (
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Low Energy Tokens
            <span className="ml-2 text-sm font-normal text-gray-600">
              ({lowEnergyTokens.length} tokens, &lt; {lowEnergyThreshold.toFixed(2)} mJ)
            </span>
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Position
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Token
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Energy (mJ)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Duration (ms)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Phase
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {lowEnergyTokens
                  .sort((a, b) => a.energy_mj - b.energy_mj)
                  .slice(0, 20)
                  .map((token) => (
                    <tr key={token.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                        {token.token_position}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-mono text-gray-900">
                        &quot;{token.token_text}&quot;
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-blue-600 font-semibold">
                        {token.energy_mj.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                        {token.duration_ms.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                        {token.phase}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {lowEnergyTokens.length > 20 && (
              <p className="mt-2 text-sm text-gray-500 text-center">
                Showing 20 of {lowEnergyTokens.length} low-energy tokens
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
