'use client';

import React, { useState, useMemo } from 'react';
import { ProfilingRun } from '@/types';
import ModelComparisonChart from './charts/ModelComparisonChart';

interface CompareViewProps {
  runs: ProfilingRun[];
  onRemoveRun: (runId: string) => void;
}

type ComparisonMetric = 'energy' | 'duration' | 'efficiency' | 'power' | 'tokens';
type ChartType = 'scatter' | 'bar';

const CompareView: React.FC<CompareViewProps> = ({ runs, onRemoveRun }) => {
  const [selectedMetric, setSelectedMetric] = useState<ComparisonMetric>('energy');
  const [chartType, setChartType] = useState<ChartType>('scatter');

  // Calculate statistics for highlighting differences
  const statistics = useMemo(() => {
    if (runs.length < 2) return null;

    const energies = runs.map(r => r.total_energy_mj);
    const durations = runs.map(r => r.total_duration_ms);
    const efficiencies = runs.map(r => r.tokens_per_joule || 0);
    const avgPowers = runs.map(r => (r.total_energy_mj / r.total_duration_ms) * 1000); // mW
    const tokensPerSec = runs.map(r => ((r.input_tokens + r.output_tokens) / r.total_duration_ms) * 1000);

    return {
      energy: {
        min: Math.min(...energies),
        max: Math.max(...energies),
        avg: energies.reduce((a, b) => a + b, 0) / energies.length,
        stdDev: Math.sqrt(energies.reduce((sum, val, _, arr) => sum + Math.pow(val - arr.reduce((a, b) => a + b, 0) / arr.length, 2), 0) / energies.length)
      },
      duration: {
        min: Math.min(...durations),
        max: Math.max(...durations),
        avg: durations.reduce((a, b) => a + b, 0) / durations.length,
        stdDev: Math.sqrt(durations.reduce((sum, val, _, arr) => sum + Math.pow(val - arr.reduce((a, b) => a + b, 0) / arr.length, 2), 0) / durations.length)
      },
      efficiency: {
        min: Math.min(...efficiencies),
        max: Math.max(...efficiencies),
        avg: efficiencies.reduce((a, b) => a + b, 0) / efficiencies.length,
        stdDev: Math.sqrt(efficiencies.reduce((sum, val, _, arr) => sum + Math.pow(val - arr.reduce((a, b) => a + b, 0) / arr.length, 2), 0) / efficiencies.length)
      },
      power: {
        min: Math.min(...avgPowers),
        max: Math.max(...avgPowers),
        avg: avgPowers.reduce((a, b) => a + b, 0) / avgPowers.length,
        stdDev: Math.sqrt(avgPowers.reduce((sum, val, _, arr) => sum + Math.pow(val - arr.reduce((a, b) => a + b, 0) / arr.length, 2), 0) / avgPowers.length)
      },
      tokensPerSec: {
        min: Math.min(...tokensPerSec),
        max: Math.max(...tokensPerSec),
        avg: tokensPerSec.reduce((a, b) => a + b, 0) / tokensPerSec.length,
        stdDev: Math.sqrt(tokensPerSec.reduce((sum, val, _, arr) => sum + Math.pow(val - arr.reduce((a, b) => a + b, 0) / arr.length, 2), 0) / tokensPerSec.length)
      }
    };
  }, [runs]);

  // Helper function to get color based on value (green for best, red for worst)
  const getValueColor = (value: number, min: number, max: number, higherIsBetter: boolean = false) => {
    if (min === max) return 'text-gray-400';
    const isGood = higherIsBetter ? value >= max * 0.95 : value <= min * 1.05;
    const isBad = higherIsBetter ? value <= min * 1.05 : value >= max * 0.95;
    if (isGood) return 'text-green-400';
    if (isBad) return 'text-red-400';
    return 'text-yellow-400';
  };

  // Helper function to calculate percentage difference from average
  const getPercentDiff = (value: number, avg: number) => {
    if (avg === 0) return 0;
    return ((value - avg) / avg) * 100;
  };

  // Prepare chart data
  const chartData = useMemo(() => {
    return runs.map(run => ({
      run_id: run.id,
      model_name: run.model_name,
      total_params: run.total_params || 0,
      total_energy_mj: run.total_energy_mj,
      energy_per_token_mj: run.total_energy_mj / (run.input_tokens + run.output_tokens),
      tokens_per_joule: run.tokens_per_joule || 0
    }));
  }, [runs]);

  return (
    <div className="w-full h-full flex flex-col p-6 overflow-auto">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-white mb-2">
          Comparing {runs.length} Runs
        </h3>
        <p className="text-sm text-gray-400">
          Side-by-side comparison of profiling metrics
        </p>
      </div>

      {/* Metric Selector */}
      <div className="mb-6">
        <label className="text-xs font-medium text-gray-400 mb-2 block">
          Focus Metric
        </label>
        <div className="flex gap-2">
          {(['energy', 'duration', 'efficiency', 'power', 'tokens'] as ComparisonMetric[]).map(metric => (
            <button
              key={metric}
              onClick={() => setSelectedMetric(metric)}
              className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                selectedMetric === metric
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {metric.charAt(0).toUpperCase() + metric.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Runs Header with Remove Buttons */}
      <div className="grid gap-4 mb-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
        {runs.map((run, idx) => (
          <div key={run.id} className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex justify-between items-start mb-2">
              <span className="text-sm font-semibold text-white">Run {idx + 1}</span>
              <button
                onClick={() => onRemoveRun(run.id)}
                className="text-xs text-red-400 hover:text-red-300 px-2 py-1 rounded hover:bg-red-900/20 transition-colors"
              >
                Remove
              </button>
            </div>
            <div className="text-xs text-gray-400 mb-1">
              {new Date(run.timestamp).toLocaleString()}
            </div>
            <div className="text-xs text-blue-400 font-mono">
              {run.model_name}
            </div>
            {run.tags && run.tags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {run.tags.map(tag => (
                  <span key={tag} className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Metrics Comparison Table */}
      {statistics && (
        <div className="space-y-6">
          {/* Total Energy */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Total Energy</h4>
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
              {runs.map(run => {
                const color = getValueColor(run.total_energy_mj, statistics.energy.min, statistics.energy.max);
                const diff = getPercentDiff(run.total_energy_mj, statistics.energy.avg);
                return (
                  <div key={run.id} className="text-center">
                    <div className={`text-2xl font-bold ${color}`}>
                      {run.total_energy_mj.toFixed(0)}
                    </div>
                    <div className="text-xs text-gray-400">mJ</div>
                    <div className={`text-xs mt-1 ${diff > 0 ? 'text-red-400' : 'text-green-400'}`}>
                      {diff > 0 ? '+' : ''}{diff.toFixed(1)}% vs avg
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Duration */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Total Duration</h4>
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
              {runs.map(run => {
                const color = getValueColor(run.total_duration_ms, statistics.duration.min, statistics.duration.max);
                const diff = getPercentDiff(run.total_duration_ms, statistics.duration.avg);
                return (
                  <div key={run.id} className="text-center">
                    <div className={`text-2xl font-bold ${color}`}>
                      {run.total_duration_ms.toFixed(0)}
                    </div>
                    <div className="text-xs text-gray-400">ms</div>
                    <div className={`text-xs mt-1 ${diff > 0 ? 'text-red-400' : 'text-green-400'}`}>
                      {diff > 0 ? '+' : ''}{diff.toFixed(1)}% vs avg
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Efficiency (Tokens per Joule) */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Energy Efficiency</h4>
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
              {runs.map(run => {
                const efficiency = run.tokens_per_joule || 0;
                const color = getValueColor(efficiency, statistics.efficiency.min, statistics.efficiency.max, true);
                const diff = getPercentDiff(efficiency, statistics.efficiency.avg);
                return (
                  <div key={run.id} className="text-center">
                    <div className={`text-2xl font-bold ${color}`}>
                      {efficiency.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-400">tokens/J</div>
                    <div className={`text-xs mt-1 ${diff > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {diff > 0 ? '+' : ''}{diff.toFixed(1)}% vs avg
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Average Power */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Average Power</h4>
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
              {runs.map(run => {
                const avgPower = (run.total_energy_mj / run.total_duration_ms) * 1000;
                const color = getValueColor(avgPower, statistics.power.min, statistics.power.max);
                const diff = getPercentDiff(avgPower, statistics.power.avg);
                return (
                  <div key={run.id} className="text-center">
                    <div className={`text-2xl font-bold ${color}`}>
                      {avgPower.toFixed(0)}
                    </div>
                    <div className="text-xs text-gray-400">mW</div>
                    <div className={`text-xs mt-1 ${diff > 0 ? 'text-red-400' : 'text-green-400'}`}>
                      {diff > 0 ? '+' : ''}{diff.toFixed(1)}% vs avg
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Throughput */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Throughput</h4>
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
              {runs.map(run => {
                const tokensPerSec = ((run.input_tokens + run.output_tokens) / run.total_duration_ms) * 1000;
                const color = getValueColor(tokensPerSec, statistics.tokensPerSec.min, statistics.tokensPerSec.max, true);
                const diff = getPercentDiff(tokensPerSec, statistics.tokensPerSec.avg);
                return (
                  <div key={run.id} className="text-center">
                    <div className={`text-2xl font-bold ${color}`}>
                      {tokensPerSec.toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-400">tokens/s</div>
                    <div className={`text-xs mt-1 ${diff > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {diff > 0 ? '+' : ''}{diff.toFixed(1)}% vs avg
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Token Breakdown */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Token Breakdown</h4>
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
              {runs.map(run => (
                <div key={run.id} className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Input:</span>
                    <span className="text-white font-mono">{run.input_tokens}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Output:</span>
                    <span className="text-white font-mono">{run.output_tokens}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total:</span>
                    <span className="text-white font-mono">{run.input_tokens + run.output_tokens}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Phase Energy Breakdown (if available) */}
          {runs.every(r => r.prefill_energy_mj && r.decode_energy_mj) && (
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-white mb-3">Phase Energy Breakdown</h4>
              <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
                {runs.map(run => {
                  const prefillPct = ((run.prefill_energy_mj || 0) / run.total_energy_mj) * 100;
                  const decodePct = ((run.decode_energy_mj || 0) / run.total_energy_mj) * 100;
                  return (
                    <div key={run.id} className="space-y-2">
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Prefill</span>
                          <span className="text-blue-400">{prefillPct.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${prefillPct}%` }}
                          ></div>
                        </div>
                        <div className="text-xs text-gray-500 text-right">
                          {(run.prefill_energy_mj || 0).toFixed(0)} mJ
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Decode</span>
                          <span className="text-orange-400">{decodePct.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-orange-500 h-2 rounded-full"
                            style={{ width: `${decodePct}%` }}
                          ></div>
                        </div>
                        <div className="text-xs text-gray-500 text-right">
                          {(run.decode_energy_mj || 0).toFixed(0)} mJ
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Statistical Summary */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Statistical Summary</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left text-gray-400 font-medium pb-2">Metric</th>
                    <th className="text-right text-gray-400 font-medium pb-2">Min</th>
                    <th className="text-right text-gray-400 font-medium pb-2">Max</th>
                    <th className="text-right text-gray-400 font-medium pb-2">Avg</th>
                    <th className="text-right text-gray-400 font-medium pb-2">Std Dev</th>
                    <th className="text-right text-gray-400 font-medium pb-2">Range</th>
                  </tr>
                </thead>
                <tbody className="text-gray-300">
                  <tr className="border-b border-gray-700/50">
                    <td className="py-2">Energy (mJ)</td>
                    <td className="text-right font-mono">{statistics.energy.min.toFixed(0)}</td>
                    <td className="text-right font-mono">{statistics.energy.max.toFixed(0)}</td>
                    <td className="text-right font-mono">{statistics.energy.avg.toFixed(0)}</td>
                    <td className="text-right font-mono">{statistics.energy.stdDev.toFixed(0)}</td>
                    <td className="text-right font-mono">
                      {(((statistics.energy.max - statistics.energy.min) / statistics.energy.avg) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-2">Duration (ms)</td>
                    <td className="text-right font-mono">{statistics.duration.min.toFixed(0)}</td>
                    <td className="text-right font-mono">{statistics.duration.max.toFixed(0)}</td>
                    <td className="text-right font-mono">{statistics.duration.avg.toFixed(0)}</td>
                    <td className="text-right font-mono">{statistics.duration.stdDev.toFixed(0)}</td>
                    <td className="text-right font-mono">
                      {(((statistics.duration.max - statistics.duration.min) / statistics.duration.avg) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-2">Efficiency (t/J)</td>
                    <td className="text-right font-mono">{statistics.efficiency.min.toFixed(2)}</td>
                    <td className="text-right font-mono">{statistics.efficiency.max.toFixed(2)}</td>
                    <td className="text-right font-mono">{statistics.efficiency.avg.toFixed(2)}</td>
                    <td className="text-right font-mono">{statistics.efficiency.stdDev.toFixed(2)}</td>
                    <td className="text-right font-mono">
                      {statistics.efficiency.avg > 0 ? (((statistics.efficiency.max - statistics.efficiency.min) / statistics.efficiency.avg) * 100).toFixed(1) : '0'}%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Comparison Charts */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <h4 className="text-sm font-semibold text-white">Model Comparison</h4>
              <div className="flex gap-2">
                <button
                  onClick={() => setChartType('scatter')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    chartType === 'scatter'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Scatter Plot
                </button>
                <button
                  onClick={() => setChartType('bar')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    chartType === 'bar'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Bar Chart
                </button>
              </div>
            </div>
            <div className="h-96">
              <ModelComparisonChart runs={chartData} chartType={chartType} />
            </div>
            <div className="mt-4 text-xs text-gray-400">
              {chartType === 'scatter' ? (
                <p>Scatter plot shows model size vs energy consumption. Color indicates efficiency (green = more efficient).</p>
              ) : (
                <p>Bar chart shows energy per token sorted from most to least efficient.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CompareView;
