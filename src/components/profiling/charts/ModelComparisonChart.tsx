'use client';

import React, { useRef, useEffect, useState } from 'react';

interface ComparisonRun {
  run_id: string;
  model_name: string;
  total_params: number;
  total_energy_mj: number;
  energy_per_token_mj: number;
  tokens_per_joule: number;
}

interface ModelComparisonChartProps {
  runs: ComparisonRun[];
  chartType: 'scatter' | 'bar';
}

const ModelComparisonChart: React.FC<ModelComparisonChartProps> = ({ runs, chartType }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredRun, setHoveredRun] = useState<ComparisonRun | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || runs.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 40, right: 40, bottom: 60, left: 70 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (chartType === 'scatter') {
      drawScatterPlot(ctx, runs, width, height, padding, chartWidth, chartHeight);
    } else {
      drawBarChart(ctx, runs, width, height, padding, chartWidth, chartHeight);
    }
  }, [runs, chartType]);

  const drawScatterPlot = (
    ctx: CanvasRenderingContext2D,
    runs: ComparisonRun[],
    width: number,
    height: number,
    padding: { top: number; right: number; bottom: number; left: number },
    chartWidth: number,
    chartHeight: number
  ) => {
    // Find data ranges
    const params = runs.map(r => r.total_params);
    const energies = runs.map(r => r.total_energy_mj);

    const minParams = Math.min(...params);
    const maxParams = Math.max(...params);
    const minEnergy = Math.min(...energies);
    const maxEnergy = Math.max(...energies);

    // Add 10% padding to ranges
    const paramRange = maxParams - minParams;
    const energyRange = maxEnergy - minEnergy;
    const xMin = minParams - paramRange * 0.1;
    const xMax = maxParams + paramRange * 0.1;
    const yMin = minEnergy - energyRange * 0.1;
    const yMax = maxEnergy + energyRange * 0.1;

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
      const value = xMin + (xMax - xMin) * (i / 5);
      const label = value >= 1e9 ? `${(value / 1e9).toFixed(1)}B` : `${(value / 1e6).toFixed(0)}M`;
      ctx.fillText(label, x, height - padding.bottom + 20);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (chartHeight / 5) * i;
      const value = yMin + (yMax - yMin) * (i / 5);
      ctx.fillText(value.toFixed(0), padding.left - 10, y + 4);
    }

    // Axis titles
    ctx.fillStyle = '#D1D5DB';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Model Size (parameters)', width / 2, height - 5);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy per Token (mJ)', 0, 0);
    ctx.restore();

    // Draw title
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.fillText('Model Size vs Energy Efficiency', width / 2, 20);

    // Draw data points
    runs.forEach((run) => {
      const x = padding.left + ((run.total_params - xMin) / (xMax - xMin)) * chartWidth;
      const y = height - padding.bottom - ((run.total_energy_mj - yMin) / (yMax - yMin)) * chartHeight;

      // Color based on efficiency
      const efficiency = run.tokens_per_joule;
      const maxEff = Math.max(...runs.map(r => r.tokens_per_joule));
      const minEff = Math.min(...runs.map(r => r.tokens_per_joule));
      const normalizedEff = (efficiency - minEff) / (maxEff - minEff);

      // Color gradient from red (low efficiency) to green (high efficiency)
      const r = Math.floor(255 * (1 - normalizedEff));
      const g = Math.floor(255 * normalizedEff);
      const b = 50;

      // Draw point
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Draw outline
      ctx.strokeStyle = '#FFF';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw label for model
      ctx.fillStyle = '#E5E7EB';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(run.model_name.split('/').pop()?.slice(0, 15) || '', x, y - 12);
    });

    // Draw legend
    const legendX = width - padding.right - 150;
    const legendY = padding.top + 20;
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = '#9CA3AF';
    ctx.textAlign = 'left';
    ctx.fillText('Efficiency:', legendX, legendY);

    // Draw gradient bar
    const gradientWidth = 100;
    const gradientHeight = 15;
    const gradient = ctx.createLinearGradient(legendX, legendY + 5, legendX + gradientWidth, legendY + 5);
    gradient.addColorStop(0, 'rgb(255, 50, 50)');
    gradient.addColorStop(0.5, 'rgb(200, 200, 50)');
    gradient.addColorStop(1, 'rgb(50, 255, 50)');

    ctx.fillStyle = gradient;
    ctx.fillRect(legendX, legendY + 5, gradientWidth, gradientHeight);
    ctx.strokeStyle = '#6B7280';
    ctx.strokeRect(legendX, legendY + 5, gradientWidth, gradientHeight);

    ctx.fillStyle = '#9CA3AF';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText('Low', legendX, legendY + gradientHeight + 18);
    ctx.textAlign = 'right';
    ctx.fillText('High', legendX + gradientWidth, legendY + gradientHeight + 18);
  };

  const drawBarChart = (
    ctx: CanvasRenderingContext2D,
    runs: ComparisonRun[],
    width: number,
    height: number,
    padding: { top: number; right: number; bottom: number; left: number },
    chartWidth: number,
    chartHeight: number
  ) => {
    // Sort runs by energy per token
    const sortedRuns = [...runs].sort((a, b) => a.energy_per_token_mj - b.energy_per_token_mj);

    const maxEnergy = Math.max(...sortedRuns.map(r => r.energy_per_token_mj));
    const barWidth = chartWidth / sortedRuns.length;
    const barPadding = barWidth * 0.2;

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

    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    ctx.setLineDash([]);

    // Draw Y-axis labels
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (chartHeight / 5) * i;
      const value = (maxEnergy * 1.1) * (i / 5);
      ctx.fillText(value.toFixed(2), padding.left - 10, y + 4);
    }

    // Axis title
    ctx.fillStyle = '#D1D5DB';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy per Token (mJ)', 0, 0);
    ctx.restore();

    // Draw title
    ctx.textAlign = 'center';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.fillText('Energy per Token by Model', width / 2, 20);

    // Draw bars
    sortedRuns.forEach((run, idx) => {
      const x = padding.left + idx * barWidth + barPadding;
      const barHeight = (run.energy_per_token_mj / (maxEnergy * 1.1)) * chartHeight;
      const y = height - padding.bottom - barHeight;

      // Color gradient based on position (best to worst)
      const normalizedPos = idx / (sortedRuns.length - 1);
      const r = Math.floor(50 + 205 * normalizedPos);
      const g = Math.floor(200 - 150 * normalizedPos);
      const b = 50;

      // Draw bar
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x, y, barWidth - barPadding * 2, barHeight);

      // Draw value on top
      ctx.fillStyle = '#E5E7EB';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        run.energy_per_token_mj.toFixed(2),
        x + (barWidth - barPadding * 2) / 2,
        y - 5
      );

      // Draw model name below
      ctx.fillStyle = '#9CA3AF';
      ctx.font = '10px Inter, sans-serif';
      ctx.save();
      ctx.translate(x + (barWidth - barPadding * 2) / 2, height - padding.bottom + 10);
      ctx.rotate(-Math.PI / 4);
      const modelName = run.model_name.split('/').pop() || run.model_name;
      ctx.fillText(modelName.slice(0, 20), 0, 0);
      ctx.restore();
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });

    // Find if hovering over a point (scatter plot only)
    if (chartType === 'scatter') {
      const padding = { top: 40, right: 40, bottom: 60, left: 70 };
      const chartWidth = rect.width - padding.left - padding.right;
      const chartHeight = rect.height - padding.top - padding.bottom;

      const params = runs.map(r => r.total_params);
      const energies = runs.map(r => r.total_energy_mj);

      const minParams = Math.min(...params);
      const maxParams = Math.max(...params);
      const minEnergy = Math.min(...energies);
      const maxEnergy = Math.max(...energies);

      const paramRange = maxParams - minParams;
      const energyRange = maxEnergy - minEnergy;
      const xMin = minParams - paramRange * 0.1;
      const xMax = maxParams + paramRange * 0.1;
      const yMin = minEnergy - energyRange * 0.1;
      const yMax = maxEnergy + energyRange * 0.1;

      let foundRun: ComparisonRun | null = null;
      for (const run of runs) {
        const x = padding.left + ((run.total_params - xMin) / (xMax - xMin)) * chartWidth;
        const y = rect.height - padding.bottom - ((run.total_energy_mj - yMin) / (yMax - yMin)) * chartHeight;

        const distance = Math.sqrt(Math.pow(e.clientX - rect.left - x, 2) + Math.pow(e.clientY - rect.top - y, 2));
        if (distance < 10) {
          foundRun = run;
          break;
        }
      }
      setHoveredRun(foundRun);
    }
  };

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width: '100%', height: '100%' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredRun(null)}
      />

      {/* Tooltip */}
      {hoveredRun && (
        <div
          className="absolute bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg pointer-events-none z-10"
          style={{
            left: mousePos.x + 10,
            top: mousePos.y + 10,
          }}
        >
          <div className="text-xs space-y-1">
            <div className="font-semibold text-white">{hoveredRun.model_name}</div>
            <div className="text-gray-400">Params: {(hoveredRun.total_params / 1e9).toFixed(2)}B</div>
            <div className="text-gray-400">Energy: {hoveredRun.total_energy_mj.toFixed(0)} mJ</div>
            <div className="text-gray-400">E/token: {hoveredRun.energy_per_token_mj.toFixed(2)} mJ</div>
            <div className="text-green-400">Efficiency: {hoveredRun.tokens_per_joule.toFixed(2)} t/J</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelComparisonChart;
