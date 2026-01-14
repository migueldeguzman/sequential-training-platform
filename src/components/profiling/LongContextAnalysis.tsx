/**
 * LongContextAnalysis - Analyze how context length affects energy consumption and KV cache pressure
 *
 * Features:
 * - Plot energy vs context length curve
 * - Show KV cache utilization trends
 * - Identify KV cache saturation point
 * - Warn when approaching memory limits
 */

import React, { useEffect, useState } from 'react';

interface ContextDataPoint {
  run_id: string;
  context_length: number;
  energy_mj: number;
  energy_per_token_mj: number;
  duration_ms: number;
  kv_cache_size_mb: number;
  kv_cache_utilization_pct: number;
}

interface KVCacheStats {
  avg_utilization_pct: number;
  max_utilization_pct: number;
  min_utilization_pct: number;
}

interface SaturationPoint {
  context_length: number;
  energy_increase_pct: number;
  message: string;
}

interface Warning {
  run_id: string;
  context_length: number;
  utilization_pct: number;
  message: string;
}

interface LongContextAnalysisData {
  context_length_vs_energy: ContextDataPoint[];
  kv_cache_stats: KVCacheStats | null;
  saturation_point: SaturationPoint | null;
  warnings: Warning[];
}

interface LongContextAnalysisProps {
  runId?: string;
  modelName?: string;
}

export const LongContextAnalysis: React.FC<LongContextAnalysisProps> = ({ runId, modelName }) => {
  const [data, setData] = useState<LongContextAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchLongContextAnalysis();
  }, [runId, modelName]);

  const fetchLongContextAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams();
      if (runId) params.append('run_id', runId);
      if (modelName) params.append('model_name', modelName);

      const response = await fetch(`/api/profiling/long-context-analysis?${params.toString()}`);
      if (!response.ok) {
        throw new Error('Failed to fetch long context analysis');
      }

      const analysisData = await response.json();
      setData(analysisData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="p-4 text-center">
        <div className="animate-pulse">Loading long context analysis...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded">
        <p className="text-red-700">Error: {error}</p>
      </div>
    );
  }

  if (!data || data.context_length_vs_energy.length === 0) {
    return (
      <div className="p-4 bg-gray-50 border border-gray-200 rounded">
        <p className="text-gray-600">No long context data available. Run profiling with different context lengths to see analysis.</p>
      </div>
    );
  }

  const maxEnergy = Math.max(...data.context_length_vs_energy.map(d => d.energy_per_token_mj));
  const maxContextLength = Math.max(...data.context_length_vs_energy.map(d => d.context_length));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">Long Context Energy Analysis</h2>
        <p className="text-gray-600">
          Analyze how context length affects energy consumption and KV cache pressure
        </p>
      </div>

      {/* Warnings */}
      {data.warnings.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-300 rounded p-4">
          <h3 className="font-semibold text-yellow-800 mb-2">Warnings</h3>
          <ul className="space-y-1">
            {data.warnings.map((warning, idx) => (
              <li key={idx} className="text-yellow-700 text-sm">
                {warning.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Saturation Point */}
      {data.saturation_point && (
        <div className="bg-red-50 border border-red-300 rounded p-4">
          <h3 className="font-semibold text-red-800 mb-2">KV Cache Saturation Detected</h3>
          <p className="text-red-700 text-sm">{data.saturation_point.message}</p>
        </div>
      )}

      {/* KV Cache Stats */}
      {data.kv_cache_stats && (
        <div className="bg-blue-50 border border-blue-300 rounded p-4">
          <h3 className="font-semibold mb-2">KV Cache Statistics</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-600">Avg Utilization</div>
              <div className="text-lg font-semibold">{data.kv_cache_stats.avg_utilization_pct.toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-gray-600">Max Utilization</div>
              <div className="text-lg font-semibold">{data.kv_cache_stats.max_utilization_pct.toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-gray-600">Min Utilization</div>
              <div className="text-lg font-semibold">{data.kv_cache_stats.min_utilization_pct.toFixed(1)}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Energy vs Context Length Chart */}
      <div className="border rounded p-4">
        <h3 className="font-semibold mb-4">Energy per Token vs Context Length</h3>
        <div className="relative h-64 bg-gray-50 rounded">
          <svg width="100%" height="100%" viewBox="0 0 800 256" preserveAspectRatio="xMidYMid meet">
            {/* Y-axis */}
            <line x1="50" y1="20" x2="50" y2="220" stroke="#999" strokeWidth="2" />
            {/* X-axis */}
            <line x1="50" y1="220" x2="780" y2="220" stroke="#999" strokeWidth="2" />

            {/* Y-axis label */}
            <text x="20" y="120" fontSize="12" fill="#666" transform="rotate(-90 20 120)" textAnchor="middle">
              Energy per Token (mJ/t)
            </text>

            {/* X-axis label */}
            <text x="415" y="245" fontSize="12" fill="#666" textAnchor="middle">
              Context Length (tokens)
            </text>

            {/* Data points and line */}
            {data.context_length_vs_energy.map((point, idx) => {
              const x = 50 + ((point.context_length / maxContextLength) * 730);
              const y = 220 - ((point.energy_per_token_mj / maxEnergy) * 200);

              const nextPoint = data.context_length_vs_energy[idx + 1];

              return (
                <g key={idx}>
                  {/* Line to next point */}
                  {nextPoint && (
                    <line
                      x1={x}
                      y1={y}
                      x2={50 + ((nextPoint.context_length / maxContextLength) * 730)}
                      y2={220 - ((nextPoint.energy_per_token_mj / maxEnergy) * 200)}
                      stroke="#3b82f6"
                      strokeWidth="2"
                    />
                  )}

                  {/* Data point */}
                  <circle
                    cx={x}
                    cy={y}
                    r="4"
                    fill="#3b82f6"
                    stroke="#fff"
                    strokeWidth="2"
                  >
                    <title>
                      Context: {point.context_length} tokens
                      Energy: {point.energy_per_token_mj.toFixed(3)} mJ/t
                      KV Cache: {point.kv_cache_utilization_pct?.toFixed(1)}%
                    </title>
                  </circle>
                </g>
              );
            })}

            {/* Saturation point marker */}
            {data.saturation_point && (
              <g>
                {data.context_length_vs_energy.map((point, idx) => {
                  if (point.context_length === data.saturation_point?.context_length) {
                    const x = 50 + ((point.context_length / maxContextLength) * 730);
                    const y = 220 - ((point.energy_per_token_mj / maxEnergy) * 200);

                    return (
                      <g key={`saturation-${idx}`}>
                        <line x1={x} y1="20" x2={x} y2="220" stroke="#ef4444" strokeWidth="2" strokeDasharray="4" />
                        <text x={x + 5} y="35" fontSize="10" fill="#ef4444" fontWeight="bold">
                          Saturation Point
                        </text>
                      </g>
                    );
                  }
                  return null;
                })}
              </g>
            )}
          </svg>
        </div>
      </div>

      {/* KV Cache Utilization Chart */}
      <div className="border rounded p-4">
        <h3 className="font-semibold mb-4">KV Cache Utilization vs Context Length</h3>
        <div className="relative h-64 bg-gray-50 rounded">
          <svg width="100%" height="100%" viewBox="0 0 800 256" preserveAspectRatio="xMidYMid meet">
            {/* Y-axis */}
            <line x1="50" y1="20" x2="50" y2="220" stroke="#999" strokeWidth="2" />
            {/* X-axis */}
            <line x1="50" y1="220" x2="780" y2="220" stroke="#999" strokeWidth="2" />

            {/* 80% warning line */}
            <line x1="50" y1={220 - (0.8 * 200)} x2="780" y2={220 - (0.8 * 200)} stroke="#f59e0b" strokeWidth="1" strokeDasharray="4" />
            <text x="55" y={220 - (0.8 * 200) - 5} fontSize="10" fill="#f59e0b">80% (Warning)</text>

            {/* Y-axis label */}
            <text x="20" y="120" fontSize="12" fill="#666" transform="rotate(-90 20 120)" textAnchor="middle">
              KV Cache Utilization (%)
            </text>

            {/* X-axis label */}
            <text x="415" y="245" fontSize="12" fill="#666" textAnchor="middle">
              Context Length (tokens)
            </text>

            {/* Data points and bars */}
            {data.context_length_vs_energy.map((point, idx) => {
              if (point.kv_cache_utilization_pct == null) return null;

              const x = 50 + ((point.context_length / maxContextLength) * 730);
              const barHeight = (point.kv_cache_utilization_pct / 100) * 200;
              const barY = 220 - barHeight;

              // Color based on utilization level
              let color = '#10b981'; // green
              if (point.kv_cache_utilization_pct > 80) {
                color = '#ef4444'; // red
              } else if (point.kv_cache_utilization_pct > 60) {
                color = '#f59e0b'; // orange
              }

              return (
                <rect
                  key={idx}
                  x={x - 10}
                  y={barY}
                  width="20"
                  height={barHeight}
                  fill={color}
                  opacity="0.7"
                >
                  <title>
                    Context: {point.context_length} tokens
                    Utilization: {point.kv_cache_utilization_pct.toFixed(1)}%
                    Cache Size: {point.kv_cache_size_mb?.toFixed(1)} MB
                  </title>
                </rect>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Data Table */}
      <div className="border rounded overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 text-left">Context Length</th>
              <th className="px-4 py-2 text-right">Energy (mJ)</th>
              <th className="px-4 py-2 text-right">Energy/Token (mJ/t)</th>
              <th className="px-4 py-2 text-right">Duration (ms)</th>
              <th className="px-4 py-2 text-right">KV Cache (MB)</th>
              <th className="px-4 py-2 text-right">Utilization (%)</th>
            </tr>
          </thead>
          <tbody>
            {data.context_length_vs_energy.map((point, idx) => (
              <tr key={idx} className="border-t hover:bg-gray-50">
                <td className="px-4 py-2">{point.context_length}</td>
                <td className="px-4 py-2 text-right">{point.energy_mj.toFixed(2)}</td>
                <td className="px-4 py-2 text-right">{point.energy_per_token_mj.toFixed(3)}</td>
                <td className="px-4 py-2 text-right">{point.duration_ms.toFixed(1)}</td>
                <td className="px-4 py-2 text-right">{point.kv_cache_size_mb?.toFixed(2) ?? 'N/A'}</td>
                <td className="px-4 py-2 text-right">
                  <span className={`
                    ${point.kv_cache_utilization_pct > 80 ? 'text-red-600 font-semibold' : ''}
                    ${point.kv_cache_utilization_pct > 60 && point.kv_cache_utilization_pct <= 80 ? 'text-orange-600' : ''}
                  `}>
                    {point.kv_cache_utilization_pct?.toFixed(1) ?? 'N/A'}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default LongContextAnalysis;
