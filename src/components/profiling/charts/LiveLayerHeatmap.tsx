'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { TokenMetrics, ComponentMetrics } from '@/types';

interface LiveLayerHeatmapProps {
  latestToken?: TokenMetrics;
  width?: number;
  height?: number;
  className?: string;
  metric?: 'duration' | 'energy' | 'activation_mean' | 'activation_max' | 'sparsity';
}

interface ChartConfig {
  padding: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  colors: {
    grid: string;
    text: string;
    background: string;
  };
  cellPadding: number;
}

const DEFAULT_CONFIG: ChartConfig = {
  padding: {
    top: 50,
    right: 30,
    bottom: 60,
    left: 80,
  },
  colors: {
    grid: '#374151',
    text: '#9ca3af',
    background: '#1f2937',
  },
  cellPadding: 2,
};

// Component names in display order
const COMPONENT_NAMES = [
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'down_proj',
  'input_layernorm',
  'post_attention_layernorm',
];

// Color scale: blue (low) to red (high)
function getColorForValue(value: number, min: number, max: number): string {
  if (max === min) {
    return 'rgb(59, 130, 246)'; // Default blue
  }

  const normalized = (value - min) / (max - min);

  // Use a blue-green-yellow-red gradient
  if (normalized < 0.25) {
    // Blue to cyan
    const t = normalized * 4;
    const r = Math.round(59 + (64 - 59) * t);
    const g = Math.round(130 + (224 - 130) * t);
    const b = Math.round(246 + (208 - 246) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (normalized < 0.5) {
    // Cyan to green
    const t = (normalized - 0.25) * 4;
    const r = Math.round(64 + (16 - 64) * t);
    const g = Math.round(224 + (185 - 224) * t);
    const b = Math.round(208 + (129 - 208) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (normalized < 0.75) {
    // Green to yellow
    const t = (normalized - 0.5) * 4;
    const r = Math.round(16 + (245 - 16) * t);
    const g = Math.round(185 + (158 - 185) * t);
    const b = Math.round(129 + (11 - 129) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Yellow to red
    const t = (normalized - 0.75) * 4;
    const r = Math.round(245 + (239 - 245) * t);
    const g = Math.round(158 + (68 - 158) * t);
    const b = Math.round(11 + (68 - 11) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

export function LiveLayerHeatmap({
  latestToken,
  width = 800,
  height = 600,
  className = '',
  metric = 'duration',
}: LiveLayerHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{
    layer: number;
    component: string;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  // Extract metric value from component
  const getMetricValue = useCallback((component: ComponentMetrics): number => {
    switch (metric) {
      case 'duration':
        return component.duration_ms;
      case 'energy':
        // Energy not directly on component, approximate from duration
        return component.duration_ms;
      case 'activation_mean':
        return component.activation_mean;
      case 'activation_max':
        return component.activation_max;
      case 'sparsity':
        return component.sparsity;
      default:
        return component.duration_ms;
    }
  }, [metric]);

  // Get metric label
  const getMetricLabel = useCallback((): string => {
    switch (metric) {
      case 'duration':
        return 'Duration (ms)';
      case 'energy':
        return 'Energy (approx)';
      case 'activation_mean':
        return 'Activation Mean';
      case 'activation_max':
        return 'Activation Max';
      case 'sparsity':
        return 'Sparsity';
      default:
        return 'Duration (ms)';
    }
  }, [metric]);

  // Handle mouse move for hover effects
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !latestToken || !latestToken.layers) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const chartWidth = width - DEFAULT_CONFIG.padding.left - DEFAULT_CONFIG.padding.right;
    const chartHeight = height - DEFAULT_CONFIG.padding.top - DEFAULT_CONFIG.padding.bottom;

    const numLayers = latestToken.layers.length;
    const numComponents = COMPONENT_NAMES.length;

    const cellWidth = chartWidth / numComponents;
    const cellHeight = chartHeight / numLayers;

    // Check if mouse is within chart area
    if (
      x < DEFAULT_CONFIG.padding.left ||
      x > width - DEFAULT_CONFIG.padding.right ||
      y < DEFAULT_CONFIG.padding.top ||
      y > height - DEFAULT_CONFIG.padding.bottom
    ) {
      setHoveredCell(null);
      return;
    }

    // Calculate which cell is hovered
    const componentIndex = Math.floor((x - DEFAULT_CONFIG.padding.left) / cellWidth);
    const layerIndex = Math.floor((y - DEFAULT_CONFIG.padding.top) / cellHeight);

    if (componentIndex >= 0 && componentIndex < numComponents && layerIndex >= 0 && layerIndex < numLayers) {
      const componentName = COMPONENT_NAMES[componentIndex];
      const layer = latestToken.layers[layerIndex];

      // Find the component in this layer
      const component = layer.components.find(c => c.component_name === componentName);

      if (component) {
        const value = getMetricValue(component);
        setHoveredCell({
          layer: layerIndex,
          component: componentName,
          value,
          x,
          y,
        });
      } else {
        setHoveredCell(null);
      }
    } else {
      setHoveredCell(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

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
    ctx.fillStyle = DEFAULT_CONFIG.colors.background;
    ctx.fillRect(0, 0, width, height);

    if (!latestToken || !latestToken.layers || latestToken.layers.length === 0) {
      // Draw "No data" message
      ctx.fillStyle = DEFAULT_CONFIG.colors.text;
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No layer data available', width / 2, height / 2);
      ctx.font = '12px sans-serif';
      ctx.fillText('Waiting for token generation...', width / 2, height / 2 + 25);
      return;
    }

    // Calculate chart dimensions
    const chartWidth = width - DEFAULT_CONFIG.padding.left - DEFAULT_CONFIG.padding.right;
    const chartHeight = height - DEFAULT_CONFIG.padding.top - DEFAULT_CONFIG.padding.bottom;

    const numLayers = latestToken.layers.length;
    const numComponents = COMPONENT_NAMES.length;

    const cellWidth = chartWidth / numComponents;
    const cellHeight = chartHeight / numLayers;

    // Build data grid and find min/max for scaling
    const grid: (number | null)[][] = [];
    let minValue = Infinity;
    let maxValue = -Infinity;

    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const layer = latestToken.layers[layerIdx];
      const row: (number | null)[] = [];

      for (const componentName of COMPONENT_NAMES) {
        const component = layer.components.find(c => c.component_name === componentName);

        if (component) {
          const value = getMetricValue(component);
          row.push(value);
          minValue = Math.min(minValue, value);
          maxValue = Math.max(maxValue, value);
        } else {
          row.push(null);
        }
      }

      grid.push(row);
    }

    // Draw cells
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      for (let compIdx = 0; compIdx < numComponents; compIdx++) {
        const value = grid[layerIdx][compIdx];

        if (value !== null) {
          const x = DEFAULT_CONFIG.padding.left + compIdx * cellWidth;
          const y = DEFAULT_CONFIG.padding.top + layerIdx * cellHeight;

          const color = getColorForValue(value, minValue, maxValue);
          ctx.fillStyle = color;
          ctx.fillRect(
            x + DEFAULT_CONFIG.cellPadding,
            y + DEFAULT_CONFIG.cellPadding,
            cellWidth - 2 * DEFAULT_CONFIG.cellPadding,
            cellHeight - 2 * DEFAULT_CONFIG.cellPadding
          );
        }
      }
    }

    // Draw grid lines
    ctx.strokeStyle = DEFAULT_CONFIG.colors.grid;
    ctx.lineWidth = 1;

    // Vertical lines (between components)
    for (let i = 0; i <= numComponents; i++) {
      const x = DEFAULT_CONFIG.padding.left + i * cellWidth;
      ctx.beginPath();
      ctx.moveTo(x, DEFAULT_CONFIG.padding.top);
      ctx.lineTo(x, height - DEFAULT_CONFIG.padding.bottom);
      ctx.stroke();
    }

    // Horizontal lines (between layers)
    for (let i = 0; i <= numLayers; i++) {
      const y = DEFAULT_CONFIG.padding.top + i * cellHeight;
      ctx.beginPath();
      ctx.moveTo(DEFAULT_CONFIG.padding.left, y);
      ctx.lineTo(width - DEFAULT_CONFIG.padding.right, y);
      ctx.stroke();
    }

    // Draw X-axis labels (component names)
    ctx.fillStyle = DEFAULT_CONFIG.colors.text;
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let i = 0; i < numComponents; i++) {
      const x = DEFAULT_CONFIG.padding.left + (i + 0.5) * cellWidth;
      const y = height - DEFAULT_CONFIG.padding.bottom + 10;

      // Rotate text for better fit
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(COMPONENT_NAMES[i], 0, 0);
      ctx.restore();
    }

    // Draw Y-axis labels (layer indices)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < numLayers; i++) {
      const x = DEFAULT_CONFIG.padding.left - 10;
      const y = DEFAULT_CONFIG.padding.top + (i + 0.5) * cellHeight;
      ctx.fillText(`Layer ${i}`, x, y);
    }

    // Draw title
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`Layer Heatmap - ${getMetricLabel()}`, width / 2, 10);

    // Draw token info
    ctx.font = '12px sans-serif';
    ctx.fillText(`Token #${latestToken.token_position}: "${latestToken.token_text}"`, width / 2, 28);

    // Draw color scale legend
    const legendWidth = 200;
    const legendHeight = 15;
    const legendX = width - DEFAULT_CONFIG.padding.right - legendWidth;
    const legendY = 10;

    // Draw gradient
    for (let i = 0; i < legendWidth; i++) {
      const value = minValue + (i / legendWidth) * (maxValue - minValue);
      const color = getColorForValue(value, minValue, maxValue);
      ctx.fillStyle = color;
      ctx.fillRect(legendX + i, legendY, 1, legendHeight);
    }

    // Draw legend border
    ctx.strokeStyle = DEFAULT_CONFIG.colors.grid;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    // Draw legend labels
    ctx.fillStyle = DEFAULT_CONFIG.colors.text;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(minValue.toFixed(3), legendX, legendY + legendHeight + 2);
    ctx.textAlign = 'right';
    ctx.fillText(maxValue.toFixed(3), legendX + legendWidth, legendY + legendHeight + 2);

    // Draw hover tooltip
    if (hoveredCell) {
      const tooltipX = hoveredCell.x > width / 2 ? hoveredCell.x - 160 : hoveredCell.x + 10;
      const tooltipY = hoveredCell.y > height / 2 ? hoveredCell.y - 80 : hoveredCell.y + 10;
      const tooltipWidth = 150;
      const tooltipHeight = 70;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 1;
      ctx.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
      ctx.strokeRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);

      // Tooltip text
      ctx.fillStyle = '#ffffff';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';

      const tooltipLines = [
        `Layer: ${hoveredCell.layer}`,
        `Component: ${hoveredCell.component}`,
        `${getMetricLabel()}:`,
        `  ${hoveredCell.value.toFixed(4)}`,
      ];

      tooltipLines.forEach((line, index) => {
        ctx.fillText(line, tooltipX + 8, tooltipY + 8 + index * 14);
      });
    }

  }, [latestToken, width, height, metric, hoveredCell, getMetricValue, getMetricLabel]);

  return (
    <div className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: hoveredCell ? 'crosshair' : 'default' }}
      />
    </div>
  );
}
