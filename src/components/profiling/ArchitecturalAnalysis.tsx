'use client';

import React, { useState, useEffect, useRef } from 'react';
import { api } from '@/lib/api';

interface ArchitecturalAnalysisProps {
  modelFilter?: string;
  minParams?: number;
  maxParams?: number;
}

interface DataPoint {
  run_id: string;
  model_name: string;
  num_layers: number;
  hidden_size: number;
  intermediate_size: number;
  num_attention_heads: number;
  attention_mechanism: string;
  total_params: number;
  total_energy_mj: number;
  energy_per_token_mj: number;
  tokens_per_joule: number;
}

interface Correlation {
  coefficient: number | null;
  p_value: number | null;
  interpretation: string;
}

const ArchitecturalAnalysis: React.FC<ArchitecturalAnalysisProps> = ({
  modelFilter,
  minParams,
  maxParams,
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<{
    data_points: DataPoint[];
    correlations: Record<string, Correlation>;
    attention_mechanism_comparison: Record<string, {
      count: number;
      avg_energy_per_token: number;
      avg_tokens_per_joule: number;
    }>;
    regression_models: {
      linear_layers: {
        slope: number;
        intercept: number;
        r_squared: number;
        description: string;
      };
      quadratic_hidden_size: {
        coefficient: number;
        intercept: number;
        r_squared: number;
        description: string;
      };
    };
    message?: string;
  } | null>(null);
  const [selectedPlot, setSelectedPlot] = useState<'layers' | 'hidden_size' | 'intermediate_size' | 'params'>('layers');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    loadAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelFilter, minParams, maxParams]);

  useEffect(() => {
    if (data && data.data_points.length > 0) {
      drawScatterPlot();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, selectedPlot]);

  const loadAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getArchitecturalAnalysis({
        model_filter: modelFilter,
        min_params: minParams,
        max_params: maxParams,
      });
      if (response.success && response.data) {
        setData(response.data);
      } else {
        setError(response.error || 'Failed to load architectural analysis');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load architectural analysis');
    } finally {
      setLoading(false);
    }
  };

  const drawScatterPlot = () => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.data_points.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 50, right: 40, bottom: 80, left: 80 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Get X-axis data based on selected plot
    let xValues: number[];
    let xLabel: string;
    let xFormatter: (val: number) => string;

    switch (selectedPlot) {
      case 'layers':
        xValues = data.data_points.map((d: DataPoint) => d.num_layers);
        xLabel = 'Number of Layers';
        xFormatter = (val) => val.toFixed(0);
        break;
      case 'hidden_size':
        xValues = data.data_points.map((d: DataPoint) => d.hidden_size);
        xLabel = 'Hidden Size (dimension)';
        xFormatter = (val) => val.toFixed(0);
        break;
      case 'intermediate_size':
        xValues = data.data_points.map((d: DataPoint) => d.intermediate_size);
        xLabel = 'Intermediate Size (FFN dimension)';
        xFormatter = (val) => val.toFixed(0);
        break;
      case 'params':
        xValues = data.data_points.map((d: DataPoint) => d.total_params);
        xLabel = 'Total Parameters';
        xFormatter = (val) => val >= 1e9 ? `${(val / 1e9).toFixed(1)}B` : `${(val / 1e6).toFixed(0)}M`;
        break;
    }

    const yValues = data.data_points.map((d: DataPoint) => d.energy_per_token_mj);

    // Find data ranges
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // Add 10% padding to ranges
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const xPadded = { min: xMin - xRange * 0.1, max: xMax + xRange * 0.1 };
    const yPadded = { min: yMin - yRange * 0.1, max: yMax + yRange * 0.1 };

    // Draw axes
    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Vertical grid lines
    for (let i = 0; i <= 5; i++) {
      const x = padding.left + (chartWidth / 5) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    ctx.setLineDash([]);

    // Draw axis labels
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';

    // X-axis labels
    for (let i = 0; i <= 5; i++) {
      const x = padding.left + (chartWidth / 5) * i;
      const value = xPadded.min + (xPadded.max - xPadded.min) * (i / 5);
      ctx.fillText(xFormatter(value), x, height - padding.bottom + 20);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (chartHeight / 5) * i;
      const value = yPadded.min + (yPadded.max - yPadded.min) * (i / 5);
      ctx.fillText(value.toFixed(2), padding.left - 10, y + 4);
    }

    // Axis titles
    ctx.fillStyle = '#D1D5DB';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, width / 2, height - 10);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy per Token (mJ)', 0, 0);
    ctx.restore();

    // Draw title
    ctx.font = 'bold 16px Inter, sans-serif';
    const title = selectedPlot === 'layers' ? 'Layers Scale Linearly (Expected)' :
                  selectedPlot === 'hidden_size' ? 'Hidden Size Scales Quadratically (Expected)' :
                  selectedPlot === 'intermediate_size' ? 'Intermediate Size Impact' :
                  'Total Parameters Impact';
    ctx.fillText(title, width / 2, 25);

    // Draw regression line if available
    if (selectedPlot === 'layers' && data.regression_models?.linear_layers) {
      const model = data.regression_models.linear_layers;
      ctx.strokeStyle = '#60A5FA';
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i <= 100; i++) {
        const xVal = xPadded.min + (xPadded.max - xPadded.min) * (i / 100);
        const yVal = model.slope * xVal + model.intercept;
        const x = padding.left + ((xVal - xPadded.min) / (xPadded.max - xPadded.min)) * chartWidth;
        const y = height - padding.bottom - ((yVal - yPadded.min) / (yPadded.max - yPadded.min)) * chartHeight;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw R² label
      ctx.fillStyle = '#60A5FA';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`R² = ${model.r_squared.toFixed(3)}`, padding.left + 10, padding.top + 20);
    }

    if (selectedPlot === 'hidden_size' && data.regression_models?.quadratic_hidden_size) {
      const model = data.regression_models.quadratic_hidden_size;
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i <= 100; i++) {
        const xVal = xPadded.min + (xPadded.max - xPadded.min) * (i / 100);
        const yVal = model.coefficient * (xVal ** 2) + model.intercept;
        const x = padding.left + ((xVal - xPadded.min) / (xPadded.max - xPadded.min)) * chartWidth;
        const y = height - padding.bottom - ((yVal - yPadded.min) / (yPadded.max - yPadded.min)) * chartHeight;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw R² label
      ctx.fillStyle = '#F59E0B';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`R² = ${model.r_squared.toFixed(3)}`, padding.left + 10, padding.top + 20);
    }

    // Draw data points
    data.data_points.forEach((point: DataPoint, idx: number) => {
      const xVal = xValues[idx];
      const yVal = yValues[idx];

      const x = padding.left + ((xVal - xPadded.min) / (xPadded.max - xPadded.min)) * chartWidth;
      const y = height - padding.bottom - ((yVal - yPadded.min) / (yPadded.max - yPadded.min)) * chartHeight;

      // Color based on attention mechanism
      let color: string;
      switch (point.attention_mechanism) {
        case 'MHA':
          color = '#EF4444';
          break;
        case 'GQA':
          color = '#10B981';
          break;
        case 'MQA':
          color = '#3B82F6';
          break;
        default:
          color = '#9CA3AF';
      }

      // Draw point
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();

      // Draw outline
      ctx.strokeStyle = '#FFF';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Draw attention mechanism legend
    const legendX = width - padding.right - 120;
    const legendY = padding.top + 10;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';

    const mechanisms = [
      { name: 'MHA', color: '#EF4444' },
      { name: 'GQA', color: '#10B981' },
      { name: 'MQA', color: '#3B82F6' },
    ];

    mechanisms.forEach((mech, idx) => {
      const y = legendY + idx * 20;

      // Draw circle
      ctx.fillStyle = mech.color;
      ctx.beginPath();
      ctx.arc(legendX, y, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#FFF';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Draw label
      ctx.fillStyle = '#9CA3AF';
      ctx.fillText(mech.name, legendX + 10, y + 4);
    });
  };

  const getCorrelation = (type: string): Correlation | null => {
    if (!data || !data.correlations) return null;
    switch (type) {
      case 'layers':
        return data.correlations.energy_vs_layers || null;
      case 'hidden_size':
        return data.correlations.energy_vs_hidden_size || null;
      case 'intermediate_size':
        return data.correlations.energy_vs_intermediate_size || null;
      case 'params':
        return data.correlations.energy_vs_total_params || null;
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-400">Loading architectural analysis...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-red-400">Error: {error}</div>
      </div>
    );
  }

  if (!data || data.data_points.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-400">
          {data?.message || 'No data available for architectural analysis'}
        </div>
      </div>
    );
  }

  const correlation = getCorrelation(selectedPlot);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Architectural Impact Analysis</h2>
          <p className="text-sm text-gray-400 mt-1">
            How model architecture affects energy consumption (inspired by Caravaca et al. 2025)
          </p>
        </div>
        <button
          onClick={loadAnalysis}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"
        >
          Refresh
        </button>
      </div>

      {/* Plot selector */}
      <div className="flex space-x-2">
        <button
          onClick={() => setSelectedPlot('layers')}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${
            selectedPlot === 'layers'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Layers (Linear)
        </button>
        <button
          onClick={() => setSelectedPlot('hidden_size')}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${
            selectedPlot === 'hidden_size'
              ? 'bg-amber-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Hidden Size (Quadratic)
        </button>
        <button
          onClick={() => setSelectedPlot('intermediate_size')}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${
            selectedPlot === 'intermediate_size'
              ? 'bg-green-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Intermediate Size
        </button>
        <button
          onClick={() => setSelectedPlot('params')}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${
            selectedPlot === 'params'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Total Parameters
        </button>
      </div>

      {/* Chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <canvas
          ref={canvasRef}
          className="w-full"
          style={{ width: '100%', height: '500px' }}
        />
      </div>

      {/* Correlation info */}
      {correlation && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Correlation Analysis</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-400">Coefficient</div>
              <div className="text-xl font-bold text-white">
                {correlation.coefficient !== null ? correlation.coefficient.toFixed(3) : 'N/A'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">P-Value</div>
              <div className="text-xl font-bold text-white">
                {correlation.p_value !== null ? correlation.p_value.toFixed(4) : 'N/A'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Interpretation</div>
              <div className="text-xl font-bold text-white capitalize">
                {correlation.interpretation}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Attention mechanism comparison */}
      {data.attention_mechanism_comparison && Object.keys(data.attention_mechanism_comparison).length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Attention Mechanism Comparison</h3>
          <div className="grid grid-cols-3 gap-4">
            {Object.entries(data.attention_mechanism_comparison).map(([mechanism, stats]) => (
              <div key={mechanism} className="bg-gray-800 rounded-lg p-4">
                <div className="text-sm font-semibold text-white mb-2">{mechanism}</div>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-gray-400">Runs: </span>
                    <span className="text-white">{stats.count}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Avg Energy/Token: </span>
                    <span className="text-white">{stats.avg_energy_per_token.toFixed(2)} mJ</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Avg Efficiency: </span>
                    <span className="text-green-400">{stats.avg_tokens_per_joule.toFixed(2)} t/J</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key insights */}
      <div className="bg-blue-900 bg-opacity-20 border border-blue-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-300 mb-2">Key Insights</h3>
        <ul className="space-y-2 text-sm text-gray-300">
          <li>• <strong>Layers:</strong> Expected to scale linearly with energy consumption</li>
          <li>• <strong>Hidden Dimension:</strong> Expected to scale quadratically (h²) with energy due to attention complexity</li>
          <li>• <strong>Attention Mechanism:</strong> GQA and MQA are typically more energy-efficient than MHA</li>
          <li>• <strong>Parameter Count:</strong> Same parameter count can have order-of-magnitude energy differences based on architecture</li>
        </ul>
      </div>

      {/* Data summary */}
      <div className="text-sm text-gray-400">
        Analyzing {data.data_points.length} profiling runs
      </div>
    </div>
  );
};

export default ArchitecturalAnalysis;
