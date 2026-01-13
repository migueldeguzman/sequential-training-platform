'use client';

import React, { useRef, useEffect, useState } from 'react';
import type { PowerSample } from '@/types';

interface PowerTimeSeriesChartProps {
  samples: PowerSample[];
  width?: number;
  height?: number;
  className?: string;
  autoScroll?: boolean;
  windowDurationMs?: number; // Duration to show in auto-scroll mode
}

interface ChartConfig {
  padding: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  colors: {
    cpu: string;
    gpu: string;
    ane: string;
    dram: string;
    total: string;
    grid: string;
    text: string;
    background: string;
  };
  lineWidth: number;
}

const DEFAULT_CONFIG: ChartConfig = {
  padding: {
    top: 30,
    right: 120,
    bottom: 40,
    left: 60,
  },
  colors: {
    cpu: '#3b82f6', // blue
    gpu: '#10b981', // green
    ane: '#f59e0b', // orange
    dram: '#a855f7', // purple
    total: '#ef4444', // red
    grid: '#374151',
    text: '#9ca3af',
    background: '#1f2937',
  },
  lineWidth: 2,
};

export function PowerTimeSeriesChart({
  samples,
  width = 800,
  height = 400,
  className = '',
  autoScroll = true,
  windowDurationMs = 30000, // 30 seconds default
}: PowerTimeSeriesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{
    x: number;
    y: number;
    sample: PowerSample;
  } | null>(null);

  // Mouse move handler for hover effects
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || samples.length === 0) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert canvas coordinates to data space
    const chartWidth = width - DEFAULT_CONFIG.padding.left - DEFAULT_CONFIG.padding.right;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const chartHeight = height - DEFAULT_CONFIG.padding.top - DEFAULT_CONFIG.padding.bottom;

    // Check if mouse is within chart area
    if (
      x < DEFAULT_CONFIG.padding.left ||
      x > width - DEFAULT_CONFIG.padding.right ||
      y < DEFAULT_CONFIG.padding.top ||
      y > height - DEFAULT_CONFIG.padding.bottom
    ) {
      setHoveredPoint(null);
      return;
    }

    // Find nearest sample
    const startTime = autoScroll && samples.length > 0
      ? Math.max(0, samples[samples.length - 1].timestamp - windowDurationMs)
      : 0;
    const endTime = autoScroll && samples.length > 0
      ? samples[samples.length - 1].timestamp
      : samples.length > 0 ? samples[samples.length - 1].timestamp : 1000;

    const mouseTime = startTime + ((x - DEFAULT_CONFIG.padding.left) / chartWidth) * (endTime - startTime);

    // Find closest sample
    let closestSample = samples[0];
    let minDistance = Math.abs(samples[0].timestamp - mouseTime);

    for (const sample of samples) {
      const distance = Math.abs(sample.timestamp - mouseTime);
      if (distance < minDistance) {
        minDistance = distance;
        closestSample = sample;
      }
    }

    // Only show hover if within reasonable distance (e.g., 500ms)
    if (minDistance < 500) {
      setHoveredPoint({ x, y, sample: closestSample });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
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

    if (samples.length === 0) {
      // Draw "No data" message
      ctx.fillStyle = DEFAULT_CONFIG.colors.text;
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No power data available', width / 2, height / 2);
      return;
    }

    // Calculate chart dimensions
    const chartWidth = width - DEFAULT_CONFIG.padding.left - DEFAULT_CONFIG.padding.right;
    const chartHeight = height - DEFAULT_CONFIG.padding.top - DEFAULT_CONFIG.padding.bottom;

    // Determine time range
    let startTime: number;
    let endTime: number;

    if (autoScroll) {
      endTime = samples[samples.length - 1].timestamp;
      startTime = Math.max(0, endTime - windowDurationMs);
    } else {
      startTime = samples[0].timestamp;
      endTime = samples[samples.length - 1].timestamp;
    }

    // Filter samples to visible range
    const visibleSamples = samples.filter(
      s => s.timestamp >= startTime && s.timestamp <= endTime
    );

    if (visibleSamples.length === 0) return;

    // Find max power for scaling
    let maxPower = 0;
    for (const sample of visibleSamples) {
      maxPower = Math.max(
        maxPower,
        sample.cpu_power_mw,
        sample.gpu_power_mw,
        sample.ane_power_mw || 0,
        sample.dram_power_mw || 0,
        sample.total_power_mw
      );
    }

    // Round up to nearest nice number
    const powerScale = Math.ceil(maxPower / 1000) * 1000;

    // Helper functions
    const timeToX = (time: number) => {
      const ratio = (time - startTime) / (endTime - startTime);
      return DEFAULT_CONFIG.padding.left + ratio * chartWidth;
    };

    const powerToY = (power: number) => {
      const ratio = power / powerScale;
      return DEFAULT_CONFIG.padding.top + chartHeight - ratio * chartHeight;
    };

    // Draw grid
    ctx.strokeStyle = DEFAULT_CONFIG.colors.grid;
    ctx.lineWidth = 1;

    // Horizontal grid lines (power)
    const powerSteps = 5;
    for (let i = 0; i <= powerSteps; i++) {
      const power = (i / powerSteps) * powerScale;
      const y = powerToY(power);

      ctx.beginPath();
      ctx.moveTo(DEFAULT_CONFIG.padding.left, y);
      ctx.lineTo(width - DEFAULT_CONFIG.padding.right, y);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = DEFAULT_CONFIG.colors.text;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${Math.round(power)}`, DEFAULT_CONFIG.padding.left - 10, y);
    }

    // Vertical grid lines (time)
    const timeSteps = 6;
    for (let i = 0; i <= timeSteps; i++) {
      const time = startTime + (i / timeSteps) * (endTime - startTime);
      const x = timeToX(time);

      ctx.beginPath();
      ctx.moveTo(x, DEFAULT_CONFIG.padding.top);
      ctx.lineTo(x, height - DEFAULT_CONFIG.padding.bottom);
      ctx.stroke();

      // X-axis labels
      ctx.fillStyle = DEFAULT_CONFIG.colors.text;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${(time / 1000).toFixed(1)}s`, x, height - DEFAULT_CONFIG.padding.bottom + 10);
    }

    // Draw axes labels
    ctx.fillStyle = DEFAULT_CONFIG.colors.text;
    ctx.font = '14px sans-serif';

    // Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Power (mW)', 0, 0);
    ctx.restore();

    // X-axis label
    ctx.textAlign = 'center';
    ctx.fillText('Time (seconds)', width / 2, height - 5);

    // Helper function to draw a line series
    const drawLine = (color: string, getValue: (sample: PowerSample) => number) => {
      if (visibleSamples.length === 0) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = DEFAULT_CONFIG.lineWidth;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      ctx.beginPath();

      for (let i = 0; i < visibleSamples.length; i++) {
        const sample = visibleSamples[i];
        const x = timeToX(sample.timestamp);
        const y = powerToY(getValue(sample));

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    };

    // Draw lines (order matters for layering)
    drawLine(DEFAULT_CONFIG.colors.dram, s => s.dram_power_mw || 0);
    drawLine(DEFAULT_CONFIG.colors.ane, s => s.ane_power_mw || 0);
    drawLine(DEFAULT_CONFIG.colors.cpu, s => s.cpu_power_mw);
    drawLine(DEFAULT_CONFIG.colors.gpu, s => s.gpu_power_mw);
    drawLine(DEFAULT_CONFIG.colors.total, s => s.total_power_mw);

    // Draw legend
    const legendX = width - DEFAULT_CONFIG.padding.right + 10;
    const legendY = DEFAULT_CONFIG.padding.top;
    const legendLineHeight = 25;

    const legendItems = [
      { label: 'Total', color: DEFAULT_CONFIG.colors.total },
      { label: 'CPU', color: DEFAULT_CONFIG.colors.cpu },
      { label: 'GPU', color: DEFAULT_CONFIG.colors.gpu },
      { label: 'ANE', color: DEFAULT_CONFIG.colors.ane },
      { label: 'DRAM', color: DEFAULT_CONFIG.colors.dram },
    ];

    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    legendItems.forEach((item, index) => {
      const y = legendY + index * legendLineHeight;

      // Draw line sample
      ctx.strokeStyle = item.color;
      ctx.lineWidth = DEFAULT_CONFIG.lineWidth;
      ctx.beginPath();
      ctx.moveTo(legendX, y);
      ctx.lineTo(legendX + 20, y);
      ctx.stroke();

      // Draw label
      ctx.fillStyle = DEFAULT_CONFIG.colors.text;
      ctx.fillText(item.label, legendX + 25, y);
    });

    // Draw hover indicator
    if (hoveredPoint) {
      const { sample } = hoveredPoint;
      const x = timeToX(sample.timestamp);

      // Draw vertical line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(x, DEFAULT_CONFIG.padding.top);
      ctx.lineTo(x, height - DEFAULT_CONFIG.padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw tooltip (positioned to avoid overflow)
      const tooltipX = hoveredPoint.x > width / 2 ? hoveredPoint.x - 150 : hoveredPoint.x + 10;
      const tooltipY = hoveredPoint.y;
      const tooltipWidth = 140;
      const tooltipHeight = 110;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 1;
      ctx.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
      ctx.strokeRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);

      // Tooltip text
      ctx.fillStyle = '#ffffff';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';

      const tooltipLines = [
        `Time: ${(sample.timestamp / 1000).toFixed(2)}s`,
        `Total: ${Math.round(sample.total_power_mw)} mW`,
        `CPU: ${Math.round(sample.cpu_power_mw)} mW`,
        `GPU: ${Math.round(sample.gpu_power_mw)} mW`,
        `ANE: ${Math.round(sample.ane_power_mw || 0)} mW`,
        `DRAM: ${Math.round(sample.dram_power_mw || 0)} mW`,
      ];

      tooltipLines.forEach((line, index) => {
        ctx.fillText(line, tooltipX + 8, tooltipY + 15 + index * 15);
      });
    }

  }, [samples, width, height, autoScroll, windowDurationMs, hoveredPoint]);

  return (
    <div className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: hoveredPoint ? 'crosshair' : 'default' }}
      />
    </div>
  );
}
