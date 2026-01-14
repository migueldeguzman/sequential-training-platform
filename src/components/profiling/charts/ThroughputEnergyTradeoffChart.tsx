'use client';

import React, { useRef, useEffect, useState } from 'react';
import {
  ThroughputEnergyDataPoint,
  ParetoFrontierPoint,
  KneePoint,
  ThroughputEnergyStatistics,
} from '@/types';

interface ThroughputEnergyTradeoffChartProps {
  dataPoints: ThroughputEnergyDataPoint[];
  paretoFrontier: ParetoFrontierPoint[];
  kneePoint: KneePoint | null;
  statistics: ThroughputEnergyStatistics;
}

const ThroughputEnergyTradeoffChart: React.FC<ThroughputEnergyTradeoffChartProps> = ({
  dataPoints,
  paretoFrontier,
  kneePoint,
  statistics,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<ThroughputEnergyDataPoint | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dataPoints.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 60, right: 40, bottom: 70, left: 80 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Find data ranges
    const throughputs = dataPoints.map((d) => d.throughput_tokens_per_second);
    const energies = dataPoints.map((d) => d.energy_per_token_mj);

    const minThroughput = Math.min(...throughputs);
    const maxThroughput = Math.max(...throughputs);
    const minEnergy = Math.min(...energies);
    const maxEnergy = Math.max(...energies);

    // Add 10% padding to ranges
    const throughputRange = maxThroughput - minThroughput;
    const energyRange = maxEnergy - minEnergy;
    const xMin = Math.max(0, minThroughput - throughputRange * 0.1);
    const xMax = maxThroughput + throughputRange * 0.1;
    const yMin = Math.max(0, minEnergy - energyRange * 0.1);
    const yMax = maxEnergy + energyRange * 0.1;

    // Scale functions
    const scaleX = (value: number) =>
      padding.left + ((value - xMin) / (xMax - xMin)) * chartWidth;
    const scaleY = (value: number) =>
      height - padding.bottom - ((value - yMin) / (yMax - yMin)) * chartHeight;

    // Draw axes
    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = '#E5E7EB';
    ctx.lineWidth = 1;

    const numGridLines = 5;
    for (let i = 0; i <= numGridLines; i++) {
      const y = padding.top + (i / numGridLines) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      const x = padding.left + (i / numGridLines) * chartWidth;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }

    // Draw axis labels
    ctx.fillStyle = '#1F2937';
    ctx.font = '14px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Throughput (tokens/second)', width / 2, height - 20);

    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy per Token (mJ)', 0, 0);
    ctx.restore();

    // Draw title
    ctx.font = 'bold 16px system-ui, -apple-system, sans-serif';
    ctx.fillText('Throughput vs Energy Efficiency Tradeoff', width / 2, 30);

    // Draw tick marks and labels
    ctx.fillStyle = '#4B5563';
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';

    for (let i = 0; i <= numGridLines; i++) {
      const throughputValue = xMin + (i / numGridLines) * (xMax - xMin);
      const x = padding.left + (i / numGridLines) * chartWidth;
      ctx.fillText(throughputValue.toFixed(1), x, height - padding.bottom + 20);

      const energyValue = yMin + (i / numGridLines) * (yMax - yMin);
      const y = height - padding.bottom - (i / numGridLines) * chartHeight;
      ctx.textAlign = 'right';
      ctx.fillText(energyValue.toFixed(2), padding.left - 10, y + 4);
      ctx.textAlign = 'center';
    }

    // Draw Pareto frontier line
    if (paretoFrontier.length >= 2) {
      ctx.strokeStyle = '#3B82F6';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();

      const sortedFrontier = [...paretoFrontier].sort(
        (a, b) => a.throughput_tokens_per_second - b.throughput_tokens_per_second
      );

      sortedFrontier.forEach((point, i) => {
        const x = scaleX(point.throughput_tokens_per_second);
        const y = scaleY(point.energy_per_token_mj);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Calculate efficiency colors (green = efficient, red = inefficient)
    const efficiencies = dataPoints.map((d) => d.tokens_per_joule);
    const minEfficiency = Math.min(...efficiencies);
    const maxEfficiency = Math.max(...efficiencies);

    const getEfficiencyColor = (efficiency: number): string => {
      const normalized = (efficiency - minEfficiency) / (maxEfficiency - minEfficiency);
      const r = Math.round(255 * (1 - normalized));
      const g = Math.round(200 * normalized);
      return `rgb(${r}, ${g}, 0)`;
    };

    // Draw data points
    dataPoints.forEach((point) => {
      const x = scaleX(point.throughput_tokens_per_second);
      const y = scaleY(point.energy_per_token_mj);

      ctx.beginPath();
      ctx.arc(x, y, point.is_pareto_optimal ? 8 : 6, 0, 2 * Math.PI);

      ctx.fillStyle = getEfficiencyColor(point.tokens_per_joule);
      ctx.fill();

      if (point.is_pareto_optimal) {
        ctx.strokeStyle = '#3B82F6';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    // Draw knee point marker
    if (kneePoint) {
      const x = scaleX(kneePoint.throughput_tokens_per_second);
      const y = scaleY(kneePoint.energy_per_token_mj);

      // Draw star marker
      ctx.fillStyle = '#F59E0B';
      ctx.strokeStyle = '#D97706';
      ctx.lineWidth = 2;

      const starRadius = 12;
      const starPoints = 5;
      ctx.beginPath();
      for (let i = 0; i < starPoints * 2; i++) {
        const radius = i % 2 === 0 ? starRadius : starRadius / 2;
        const angle = (Math.PI / starPoints) * i;
        const px = x + radius * Math.sin(angle);
        const py = y - radius * Math.cos(angle);
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Label knee point
      ctx.fillStyle = '#D97706';
      ctx.font = 'bold 12px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Knee', x, y - 20);
    }

    // Draw legend
    const legendX = width - padding.right - 180;
    const legendY = padding.top + 20;
    const legendSpacing = 25;

    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'left';

    // Pareto optimal points
    ctx.fillStyle = '#3B82F6';
    ctx.beginPath();
    ctx.arc(legendX, legendY, 6, 0, 2 * Math.PI);
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#1F2937';
    ctx.fillText('Pareto Optimal', legendX + 15, legendY + 4);

    // Knee point
    if (kneePoint) {
      const ky = legendY + legendSpacing;
      ctx.fillStyle = '#F59E0B';
      ctx.beginPath();
      ctx.arc(ky, ky, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#1F2937';
      ctx.fillText('Knee Point', ky + 15, ky + 4);
    }

    // Efficiency color scale
    const scaleY2 = legendY + legendSpacing * 2;
    ctx.fillStyle = '#1F2937';
    ctx.fillText('Efficiency:', legendX, scaleY2 + 4);

    const gradientWidth = 100;
    const gradientHeight = 10;
    const gradient = ctx.createLinearGradient(legendX, scaleY2 + 10, legendX + gradientWidth, scaleY2 + 10);
    gradient.addColorStop(0, 'rgb(255, 0, 0)');
    gradient.addColorStop(0.5, 'rgb(128, 100, 0)');
    gradient.addColorStop(1, 'rgb(0, 200, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(legendX, scaleY2 + 10, gradientWidth, gradientHeight);

    ctx.fillStyle = '#6B7280';
    ctx.font = '10px system-ui, -apple-system, sans-serif';
    ctx.fillText('Low', legendX, scaleY2 + 32);
    ctx.textAlign = 'right';
    ctx.fillText('High', legendX + gradientWidth, scaleY2 + 32);

  }, [dataPoints, paretoFrontier, kneePoint]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMousePos({ x: e.clientX, y: e.clientY });

    // Find closest point
    const padding = { top: 60, right: 40, bottom: 70, left: 80 };
    const chartWidth = rect.width - padding.left - padding.right;
    const chartHeight = rect.height - padding.top - padding.bottom;

    const throughputs = dataPoints.map((d) => d.throughput_tokens_per_second);
    const energies = dataPoints.map((d) => d.energy_per_token_mj);

    const minThroughput = Math.min(...throughputs);
    const maxThroughput = Math.max(...throughputs);
    const minEnergy = Math.min(...energies);
    const maxEnergy = Math.max(...energies);

    const throughputRange = maxThroughput - minThroughput;
    const energyRange = maxEnergy - minEnergy;
    const xMin = Math.max(0, minThroughput - throughputRange * 0.1);
    const xMax = maxThroughput + throughputRange * 0.1;
    const yMin = Math.max(0, minEnergy - energyRange * 0.1);
    const yMax = maxEnergy + energyRange * 0.1;

    const scaleX = (value: number) =>
      padding.left + ((value - xMin) / (xMax - xMin)) * chartWidth;
    const scaleY = (value: number) =>
      rect.height - padding.bottom - ((value - yMin) / (yMax - yMin)) * chartHeight;

    let closestPoint: ThroughputEnergyDataPoint | null = null;
    let minDistance = Infinity;

    dataPoints.forEach((point) => {
      const px = scaleX(point.throughput_tokens_per_second);
      const py = scaleY(point.energy_per_token_mj);
      const distance = Math.sqrt((px - x) ** 2 + (py - y) ** 2);

      if (distance < 15 && distance < minDistance) {
        minDistance = distance;
        closestPoint = point;
      }
    });

    setHoveredPoint(closestPoint);
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className="w-full h-full flex flex-col">
      <div className="relative flex-1">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ width: '100%', height: '100%' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />

        {hoveredPoint && (
          <div
            className="absolute bg-gray-900 text-white px-3 py-2 rounded shadow-lg text-sm pointer-events-none z-10"
            style={{
              left: mousePos.x + 10,
              top: mousePos.y + 10,
            }}
          >
            <div className="font-semibold">{hoveredPoint.model_name}</div>
            <div className="text-gray-300">Run: {hoveredPoint.run_id.substring(0, 8)}</div>
            <div className="mt-1 space-y-1">
              <div>Throughput: {hoveredPoint.throughput_tokens_per_second.toFixed(2)} tokens/s</div>
              <div>Energy/token: {hoveredPoint.energy_per_token_mj.toFixed(4)} mJ</div>
              <div>Efficiency: {hoveredPoint.tokens_per_joule.toFixed(2)} tokens/J</div>
              {hoveredPoint.batch_size && <div>Batch size: {hoveredPoint.batch_size}</div>}
              {hoveredPoint.is_pareto_optimal && (
                <div className="text-blue-400 font-semibold">✓ Pareto Optimal</div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Interpretation Panel */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg space-y-4">
        <div>
          <h3 className="font-semibold text-gray-900 mb-2">Analysis Summary</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Total Runs:</span>{' '}
              <span className="font-medium">{statistics.total_runs}</span>
            </div>
            <div>
              <span className="text-gray-600">Unique Models:</span>{' '}
              <span className="font-medium">{statistics.unique_models}</span>
            </div>
            <div>
              <span className="text-gray-600">Throughput Range:</span>{' '}
              <span className="font-medium">
                {statistics.throughput_range[0].toFixed(1)} - {statistics.throughput_range[1].toFixed(1)} tokens/s
              </span>
            </div>
            <div>
              <span className="text-gray-600">Energy Range:</span>{' '}
              <span className="font-medium">
                {statistics.energy_per_token_range[0].toFixed(4)} - {statistics.energy_per_token_range[1].toFixed(4)} mJ/token
              </span>
            </div>
          </div>
        </div>

        {statistics.best_throughput && (
          <div>
            <h4 className="font-semibold text-gray-900 text-sm mb-1">Best Throughput</h4>
            <div className="text-sm text-gray-700">
              <span className="font-medium">{statistics.best_throughput.model_name}</span>:{' '}
              {statistics.best_throughput.throughput_tokens_per_second.toFixed(2)} tokens/s
              {' '}({statistics.best_throughput.energy_per_token_mj.toFixed(4)} mJ/token)
            </div>
          </div>
        )}

        {statistics.best_efficiency && (
          <div>
            <h4 className="font-semibold text-gray-900 text-sm mb-1">Best Efficiency</h4>
            <div className="text-sm text-gray-700">
              <span className="font-medium">{statistics.best_efficiency.model_name}</span>:{' '}
              {statistics.best_efficiency.tokens_per_joule.toFixed(2)} tokens/J
              {' '}({statistics.best_efficiency.throughput_tokens_per_second.toFixed(2)} tokens/s)
            </div>
          </div>
        )}

        {kneePoint && (
          <div className="bg-amber-50 border border-amber-200 rounded p-3">
            <h4 className="font-semibold text-amber-900 text-sm mb-1">
              ⭐ Recommended Configuration (Knee Point)
            </h4>
            <div className="text-sm text-amber-800">{kneePoint.interpretation}</div>
          </div>
        )}

        <div className="text-xs text-gray-500 border-t pt-3">
          <p>
            <strong>Pareto Frontier:</strong> Points where no other configuration offers both higher throughput and better efficiency.
            The <strong>knee point</strong> represents the best balance between throughput and energy efficiency.
          </p>
          <p className="mt-1">
            Based on TokenPowerBench research: Batching improves both throughput and efficiency up to a point.
            Points are color-coded by efficiency (green = high tokens/joule, red = low tokens/joule).
          </p>
        </div>
      </div>
    </div>
  );
};

export default ThroughputEnergyTradeoffChart;
