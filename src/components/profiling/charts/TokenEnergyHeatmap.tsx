'use client';

import React, { useEffect, useRef, useState } from 'react';
import { TokenMetrics } from '@/types';

interface TokenEnergyHeatmapProps {
  tokens: TokenMetrics[];
  metric?: 'energy' | 'duration' | 'activation_mean' | 'sparsity';
  width?: number;
  height?: number;
  energyThreshold?: number; // Filter tokens above this energy (mJ)
}

export default function TokenEnergyHeatmap({
  tokens,
  metric = 'energy',
  width = 1000,
  height = 600,
  energyThreshold,
}: TokenEnergyHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{
    tokenIndex: number;
    layerIndex: number;
    value: number;
    tokenText: string;
    screenX: number;
    screenY: number;
  } | null>(null);

  const MARGIN = React.useMemo(() => ({ top: 50, right: 150, bottom: 100, left: 80 }), []);
  const LEGEND_WIDTH = 20;
  const LEGEND_HEIGHT = 200;

  // Filter tokens by energy threshold if provided
  const filteredTokens = React.useMemo(() => {
    if (energyThreshold === undefined || energyThreshold === null) {
      return tokens;
    }
    return tokens.filter((token) => token.energy_mj >= energyThreshold);
  }, [tokens, energyThreshold]);

  // Extract heatmap data
  const { data, minValue, maxValue, numLayers } = React.useMemo(() => {
    if (!filteredTokens.length) {
      return { data: [], minValue: 0, maxValue: 0, numLayers: 0 };
    }

    // Determine number of layers
    const maxLayers = Math.max(
      ...filteredTokens.map((token) => token.layers?.length || 0)
    );

    // Build 2D array: rows = layers, columns = tokens
    const heatmapData: number[][] = [];
    let min = Infinity;
    let max = -Infinity;

    for (let layer = 0; layer < maxLayers; layer++) {
      const row: number[] = [];
      for (const token of filteredTokens) {
        const layerMetrics = token.layers?.[layer];
        let value = 0;

        if (layerMetrics) {
          switch (metric) {
            case 'energy':
              value = layerMetrics.energy_mj;
              break;
            case 'duration':
              value = layerMetrics.total_duration_ms;
              break;
            case 'activation_mean':
              // Average across components
              if (layerMetrics.components?.length) {
                value =
                  layerMetrics.components.reduce(
                    (sum, comp) => sum + (comp.activation_mean || 0),
                    0
                  ) / layerMetrics.components.length;
              }
              break;
            case 'sparsity':
              // Average across components
              if (layerMetrics.components?.length) {
                value =
                  layerMetrics.components.reduce(
                    (sum, comp) => sum + (comp.sparsity || 0),
                    0
                  ) / layerMetrics.components.length;
              }
              break;
          }
        }

        row.push(value);
        if (value < min) min = value;
        if (value > max) max = value;
      }
      heatmapData.push(row);
    }

    return {
      data: heatmapData,
      minValue: min,
      maxValue: max,
      numLayers: maxLayers,
    };
  }, [filteredTokens, metric]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length || !filteredTokens.length) return;

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
    const cellWidth = chartWidth / filteredTokens.length;
    const cellHeight = chartHeight / numLayers;

    // Helper function to interpolate between colors (blue to red gradient)
    const interpolateColor = (value: number): string => {
      const normalized =
        maxValue === minValue ? 0.5 : (value - minValue) / (maxValue - minValue);

      // Blue (low) to Yellow (mid) to Red (high)
      let r, g, b;
      if (normalized < 0.5) {
        // Blue to Yellow
        const t = normalized * 2;
        r = Math.round(0 + 255 * t);
        g = Math.round(100 + 155 * t);
        b = Math.round(255 * (1 - t));
      } else {
        // Yellow to Red
        const t = (normalized - 0.5) * 2;
        r = 255;
        g = Math.round(255 * (1 - t));
        b = 0;
      }

      return `rgb(${r}, ${g}, ${b})`;
    };

    // Draw heatmap cells
    data.forEach((row, layerIndex) => {
      row.forEach((value, tokenIndex) => {
        const x = MARGIN.left + tokenIndex * cellWidth;
        const y = MARGIN.top + layerIndex * cellHeight;

        ctx.fillStyle = value === 0 ? '#f3f4f6' : interpolateColor(value);
        ctx.fillRect(x, y, cellWidth, cellHeight);

        // Draw cell border
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 0.3;
        ctx.strokeRect(x, y, cellWidth, cellHeight);
      });
    });

    // Draw X-axis labels (token positions - sample every Nth token if too many)
    ctx.fillStyle = '#374151';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    const labelInterval = Math.max(1, Math.floor(filteredTokens.length / 20));
    filteredTokens.forEach((token, i) => {
      if (i % labelInterval === 0 || i === filteredTokens.length - 1) {
        const x = MARGIN.left + (i + 0.5) * cellWidth;
        const y = MARGIN.top + chartHeight + 15;

        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(`${token.token_position}`, 0, 0);
        ctx.restore();
      }
    });

    // Draw Y-axis labels (layer indices)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '11px sans-serif';
    for (let i = 0; i < numLayers; i++) {
      const x = MARGIN.left - 10;
      const y = MARGIN.top + (i + 0.5) * cellHeight;
      ctx.fillText(`Layer ${i}`, x, y);
    }

    // Draw color scale legend
    const legendX = width - MARGIN.right + 40;
    const legendY = MARGIN.top + (chartHeight - LEGEND_HEIGHT) / 2;

    // Draw gradient
    const gradient = ctx.createLinearGradient(0, legendY + LEGEND_HEIGHT, 0, legendY);
    gradient.addColorStop(0, 'rgb(0, 100, 255)'); // Blue (low)
    gradient.addColorStop(0.5, 'rgb(255, 255, 0)'); // Yellow (mid)
    gradient.addColorStop(1, 'rgb(255, 0, 0)'); // Red (high)
    ctx.fillStyle = gradient;
    ctx.fillRect(legendX, legendY, LEGEND_WIDTH, LEGEND_HEIGHT);

    // Draw legend border
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, LEGEND_WIDTH, LEGEND_HEIGHT);

    // Draw legend labels
    ctx.fillStyle = '#374151';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const metricUnit = metric === 'energy' ? 'mJ' : metric === 'duration' ? 'ms' : '';
    ctx.fillText(`${maxValue.toFixed(2)} ${metricUnit}`, legendX + LEGEND_WIDTH + 5, legendY);
    ctx.fillText(`${minValue.toFixed(2)} ${metricUnit}`, legendX + LEGEND_WIDTH + 5, legendY + LEGEND_HEIGHT);

    // Draw axis titles
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Token Position', MARGIN.left + chartWidth / 2, height - 25);

    ctx.save();
    ctx.translate(15, MARGIN.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Transformer Layers', 0, 0);
    ctx.restore();

    // Draw title
    ctx.fillStyle = '#111827';
    ctx.font = 'bold 15px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const metricLabel = metric === 'energy' ? 'Energy' :
                        metric === 'duration' ? 'Duration' :
                        metric === 'activation_mean' ? 'Activation Mean' : 'Sparsity';
    ctx.fillText(`Token-Layer ${metricLabel} Heatmap`, width / 2, 10);
  }, [data, filteredTokens, metric, minValue, maxValue, numLayers, width, height, MARGIN]);

  // Handle mouse move for hover effect
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const chartWidth = width - MARGIN.left - MARGIN.right;
    const chartHeight = height - MARGIN.top - MARGIN.bottom;
    const cellWidth = chartWidth / filteredTokens.length;
    const cellHeight = chartHeight / numLayers;

    // Check if mouse is within chart area
    if (
      x >= MARGIN.left &&
      x <= MARGIN.left + chartWidth &&
      y >= MARGIN.top &&
      y <= MARGIN.top + chartHeight
    ) {
      const tokenIndex = Math.floor((x - MARGIN.left) / cellWidth);
      const layerIndex = Math.floor((y - MARGIN.top) / cellHeight);

      if (
        tokenIndex >= 0 &&
        tokenIndex < filteredTokens.length &&
        layerIndex >= 0 &&
        layerIndex < numLayers &&
        data[layerIndex] &&
        data[layerIndex][tokenIndex] !== undefined
      ) {
        setHoveredCell({
          tokenIndex,
          layerIndex,
          value: data[layerIndex][tokenIndex],
          tokenText: filteredTokens[tokenIndex].token_text,
          screenX: e.clientX,
          screenY: e.clientY,
        });
        canvas.style.cursor = 'pointer';
        return;
      }
    }

    setHoveredCell(null);
    canvas.style.cursor = 'default';
  };

  // Handle mouse leave
  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  if (!filteredTokens.length) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded border border-gray-200">
        <p className="text-gray-500">
          {energyThreshold ?
            `No tokens found with energy >= ${energyThreshold} mJ` :
            'No token data available'}
        </p>
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
      {hoveredCell && (
        <div
          className="fixed z-50 bg-gray-900 text-white text-xs px-3 py-2 rounded shadow-lg pointer-events-none"
          style={{
            left: hoveredCell.screenX + 10,
            top: hoveredCell.screenY + 10,
          }}
        >
          <div className="font-semibold">Token: &quot;{hoveredCell.tokenText}&quot;</div>
          <div className="text-gray-300">Position: {filteredTokens[hoveredCell.tokenIndex].token_position}</div>
          <div className="text-gray-300">Layer: {hoveredCell.layerIndex}</div>
          <div className="mt-1 font-mono">
            {hoveredCell.value.toFixed(3)} {metric === 'energy' ? 'mJ' : metric === 'duration' ? 'ms' : ''}
          </div>
        </div>
      )}
    </div>
  );
}
