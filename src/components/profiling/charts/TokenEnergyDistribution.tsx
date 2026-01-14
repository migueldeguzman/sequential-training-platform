'use client';

import React, { useEffect, useRef, useState } from 'react';
import { TokenMetrics } from '@/types';

interface TokenEnergyDistributionProps {
  tokens: TokenMetrics[];
  width?: number;
  height?: number;
  numBins?: number;
}

export default function TokenEnergyDistribution({
  tokens,
  width = 800,
  height = 400,
  numBins = 30,
}: TokenEnergyDistributionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredBin, setHoveredBin] = useState<{
    binIndex: number;
    count: number;
    minEnergy: number;
    maxEnergy: number;
    screenX: number;
    screenY: number;
  } | null>(null);

  const MARGIN = React.useMemo(() => ({ top: 60, right: 40, bottom: 80, left: 80 }), []);

  // Calculate histogram bins
  const { bins, binWidth, minEnergy, maxEnergy } = React.useMemo(() => {
    if (!tokens.length) {
      return { bins: [], binWidth: 0, minEnergy: 0, maxEnergy: 0 };
    }

    const energies = tokens.map((t) => t.energy_mj);
    const min = Math.min(...energies);
    const max = Math.max(...energies);
    const range = max - min;
    const width = range / numBins;

    // Initialize bins
    const histogramBins: number[] = new Array(numBins).fill(0);

    // Fill bins
    energies.forEach((energy) => {
      const binIndex = Math.min(Math.floor((energy - min) / width), numBins - 1);
      histogramBins[binIndex]++;
    });

    return {
      bins: histogramBins,
      binWidth: width,
      minEnergy: min,
      maxEnergy: max,
    };
  }, [tokens, numBins]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bins.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution for high DPI displays
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate dimensions
    const chartWidth = width - MARGIN.left - MARGIN.right;
    const chartHeight = height - MARGIN.top - MARGIN.bottom;
    const barWidth = chartWidth / bins.length;
    const maxCount = Math.max(...bins);

    // Draw bars
    bins.forEach((count, i) => {
      const barHeight = (count / maxCount) * chartHeight;
      const x = MARGIN.left + i * barWidth;
      const y = MARGIN.top + chartHeight - barHeight;

      // Bar fill
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(x, y, barWidth - 2, barHeight);

      // Bar border
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, barWidth - 2, barHeight);
    });

    // Draw X-axis
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top + chartHeight);
    ctx.lineTo(MARGIN.left + chartWidth, MARGIN.top + chartHeight);
    ctx.stroke();

    // Draw Y-axis
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top);
    ctx.lineTo(MARGIN.left, MARGIN.top + chartHeight);
    ctx.stroke();

    // Draw X-axis labels (energy bins)
    ctx.fillStyle = '#374151';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Show labels for first, middle, and last bins
    const labelIndices = [0, Math.floor(bins.length / 2), bins.length - 1];
    labelIndices.forEach((i) => {
      const energy = minEnergy + i * binWidth;
      const x = MARGIN.left + (i + 0.5) * barWidth;
      const y = MARGIN.top + chartHeight + 10;
      ctx.fillText(`${energy.toFixed(2)}`, x, y);
    });

    // X-axis label
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText('Energy per Token (mJ)', MARGIN.left + chartWidth / 2, height - 20);

    // Draw Y-axis labels (counts)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '11px sans-serif';

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const count = Math.round((maxCount / yTicks) * i);
      const y = MARGIN.top + chartHeight - (chartHeight / yTicks) * i;
      ctx.fillText(count.toString(), MARGIN.left - 10, y);

      // Draw gridline
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, y);
      ctx.lineTo(MARGIN.left + chartWidth, y);
      ctx.stroke();
    }

    // Y-axis label
    ctx.save();
    ctx.translate(15, MARGIN.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Token Count', 0, 0);
    ctx.restore();

    // Draw title
    ctx.fillStyle = '#111827';
    ctx.font = 'bold 15px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Token Energy Distribution', width / 2, 10);

    // Draw statistics
    const mean = tokens.reduce((sum, t) => sum + t.energy_mj, 0) / tokens.length;
    const sorted = [...tokens].sort((a, b) => a.energy_mj - b.energy_mj);
    const median = sorted[Math.floor(sorted.length / 2)].energy_mj;

    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#6b7280';
    ctx.fillText(`Mean: ${mean.toFixed(3)} mJ`, MARGIN.left, 30);
    ctx.fillText(`Median: ${median.toFixed(3)} mJ`, MARGIN.left + 120, 30);
    ctx.fillText(`Range: ${minEnergy.toFixed(3)} - ${maxEnergy.toFixed(3)} mJ`, MARGIN.left + 240, 30);
  }, [bins, binWidth, minEnergy, maxEnergy, tokens, width, height, MARGIN]);

  // Handle mouse move for hover effect
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const chartWidth = width - MARGIN.left - MARGIN.right;
    const chartHeight = height - MARGIN.top - MARGIN.bottom;
    const barWidth = chartWidth / bins.length;

    // Check if mouse is within chart area
    if (
      x >= MARGIN.left &&
      x <= MARGIN.left + chartWidth &&
      y >= MARGIN.top &&
      y <= MARGIN.top + chartHeight
    ) {
      const binIndex = Math.floor((x - MARGIN.left) / barWidth);

      if (binIndex >= 0 && binIndex < bins.length) {
        const minBinEnergy = minEnergy + binIndex * binWidth;
        const maxBinEnergy = minBinEnergy + binWidth;

        setHoveredBin({
          binIndex,
          count: bins[binIndex],
          minEnergy: minBinEnergy,
          maxEnergy: maxBinEnergy,
          screenX: e.clientX,
          screenY: e.clientY,
        });
        canvas.style.cursor = 'pointer';
        return;
      }
    }

    setHoveredBin(null);
    canvas.style.cursor = 'default';
  };

  // Handle mouse leave
  const handleMouseLeave = () => {
    setHoveredBin(null);
  };

  if (!tokens.length) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded border border-gray-200">
        <p className="text-gray-500">No token data available</p>
      </div>
    );
  }

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="border border-gray-200 rounded"
      />

      {/* Tooltip */}
      {hoveredBin && (
        <div
          className="fixed z-50 bg-gray-900 text-white text-xs px-3 py-2 rounded shadow-lg pointer-events-none"
          style={{
            left: hoveredBin.screenX + 10,
            top: hoveredBin.screenY + 10,
          }}
        >
          <div className="font-semibold">{hoveredBin.count} tokens</div>
          <div className="text-gray-300">
            Energy: {hoveredBin.minEnergy.toFixed(3)} - {hoveredBin.maxEnergy.toFixed(3)} mJ
          </div>
        </div>
      )}
    </div>
  );
}
