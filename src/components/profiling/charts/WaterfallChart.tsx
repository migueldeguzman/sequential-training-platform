import React, { useRef, useEffect, useState } from 'react';
import { TokenMetrics } from '@/types';

interface WaterfallChartProps {
  tokens: TokenMetrics[];
  width?: number;
  height?: number;
}

interface TooltipData {
  x: number;
  y: number;
  token: TokenMetrics;
  tokenIndex: number;
}

export const WaterfallChart: React.FC<WaterfallChartProps> = ({
  tokens,
  width = 800,
  height = 400,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [scrollLeft, setScrollLeft] = useState(0);

  useEffect(() => {
    // Component colors for stacked bars
    const componentColors = {
      attention: '#3b82f6', // blue
      mlp: '#10b981',       // green
      layernorm: '#f59e0b', // amber
      other: '#6b7280',     // gray
    };

    const lineColor = '#ef4444'; // red for cumulative energy

    if (!canvasRef.current || tokens.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size for high DPI displays
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate dimensions
    const padding = { top: 40, right: 60, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Token column width
    const tokenCount = tokens.length;
    const columnWidth = Math.max(20, Math.min(60, chartWidth / tokenCount));
    const actualChartWidth = columnWidth * tokenCount;

    // Find max energy for scaling
    const maxEnergy = Math.max(...tokens.map(t =>
      (t.energy_mj || 0) + (t.layers?.reduce((sum, l) => sum + (l.energy_mj || 0), 0) || 0)
    ));

    // Calculate cumulative energy
    const cumulativeEnergy = tokens.reduce((acc, token, i) => {
      const tokenEnergy = (token.energy_mj || 0) +
        (token.layers?.reduce((sum, l) => sum + (l.energy_mj || 0), 0) || 0);
      acc.push((acc[i - 1] || 0) + tokenEnergy);
      return acc;
    }, [] as number[]);
    const maxCumulativeEnergy = cumulativeEnergy[cumulativeEnergy.length - 1] || 1;

    // Helper function to scale y values
    const scaleY = (value: number) => {
      return padding.top + chartHeight - (value / maxEnergy) * chartHeight;
    };

    const scaleCumulativeY = (value: number) => {
      return padding.top + chartHeight - (value / maxCumulativeEnergy) * chartHeight;
    };

    // Draw axes
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(padding.left + actualChartWidth, height - padding.bottom);
    ctx.stroke();

    // Draw Y-axis label
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.save();
    ctx.translate(20, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy per Token (mJ)', 0, 0);
    ctx.restore();

    // Draw Y-axis ticks (left side - per token energy)
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const value = (maxEnergy / yTicks) * i;
      const y = scaleY(value);

      // Tick mark
      ctx.beginPath();
      ctx.moveTo(padding.left - 5, y);
      ctx.lineTo(padding.left, y);
      ctx.stroke();

      // Label
      ctx.fillText(value.toFixed(1), padding.left - 10, y + 4);
    }

    // Draw right Y-axis label
    ctx.textAlign = 'center';
    ctx.save();
    ctx.translate(width - 20, padding.top + chartHeight / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillText('Cumulative Energy (mJ)', 0, 0);
    ctx.restore();

    // Draw right Y-axis ticks (cumulative energy)
    ctx.textAlign = 'left';
    for (let i = 0; i <= yTicks; i++) {
      const value = (maxCumulativeEnergy / yTicks) * i;
      const y = scaleCumulativeY(value);

      // Tick mark
      ctx.beginPath();
      ctx.moveTo(width - padding.right, y);
      ctx.lineTo(width - padding.right + 5, y);
      ctx.stroke();

      // Label
      ctx.fillText(value.toFixed(1), width - padding.right + 10, y + 4);
    }

    // Draw stacked bars for each token
    tokens.forEach((token, i) => {
      const x = padding.left + i * columnWidth;

      // Calculate component durations to estimate energy proportions
      const attentionDuration = token.layers?.reduce((sum, l) => {
        const attnComponents = l.components?.filter(c =>
          c.component_name.includes('attention') ||
          c.component_name.includes('attn')
        ) || [];
        return sum + attnComponents.reduce((s, c) => s + (c.duration_ms || 0), 0);
      }, 0) || 0;

      const mlpDuration = token.layers?.reduce((sum, l) => {
        const mlpComponents = l.components?.filter(c =>
          c.component_name.includes('mlp') ||
          c.component_name.includes('ffn')
        ) || [];
        return sum + mlpComponents.reduce((s, c) => s + (c.duration_ms || 0), 0);
      }, 0) || 0;

      const layernormDuration = token.layers?.reduce((sum, l) => {
        const lnComponents = l.components?.filter(c =>
          c.component_name.includes('norm')
        ) || [];
        return sum + lnComponents.reduce((s, c) => s + (c.duration_ms || 0), 0);
      }, 0) || 0;

      const totalLayerEnergy = token.layers?.reduce((sum, l) =>
        sum + (l.energy_mj || 0), 0
      ) || 0;

      const totalComponentDuration = attentionDuration + mlpDuration + layernormDuration;

      // Distribute energy proportionally by duration
      const attentionEnergy = totalComponentDuration > 0
        ? (attentionDuration / totalComponentDuration) * totalLayerEnergy
        : 0;
      const mlpEnergy = totalComponentDuration > 0
        ? (mlpDuration / totalComponentDuration) * totalLayerEnergy
        : 0;
      const layernormEnergy = totalComponentDuration > 0
        ? (layernormDuration / totalComponentDuration) * totalLayerEnergy
        : 0;

      const otherEnergy = totalLayerEnergy - attentionEnergy - mlpEnergy - layernormEnergy;

      // Draw stacked bars from bottom to top
      let currentY = height - padding.bottom;

      // Attention
      if (attentionEnergy > 0) {
        const barHeight = (attentionEnergy / maxEnergy) * chartHeight;
        ctx.fillStyle = componentColors.attention;
        ctx.fillRect(x + 2, currentY - barHeight, columnWidth - 4, barHeight);
        currentY -= barHeight;
      }

      // MLP
      if (mlpEnergy > 0) {
        const barHeight = (mlpEnergy / maxEnergy) * chartHeight;
        ctx.fillStyle = componentColors.mlp;
        ctx.fillRect(x + 2, currentY - barHeight, columnWidth - 4, barHeight);
        currentY -= barHeight;
      }

      // LayerNorm
      if (layernormEnergy > 0) {
        const barHeight = (layernormEnergy / maxEnergy) * chartHeight;
        ctx.fillStyle = componentColors.layernorm;
        ctx.fillRect(x + 2, currentY - barHeight, columnWidth - 4, barHeight);
        currentY -= barHeight;
      }

      // Other
      if (otherEnergy > 0) {
        const barHeight = (otherEnergy / maxEnergy) * chartHeight;
        ctx.fillStyle = componentColors.other;
        ctx.fillRect(x + 2, currentY - barHeight, columnWidth - 4, barHeight);
      }
    });

    // Draw cumulative energy line
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    cumulativeEnergy.forEach((cumEnergy, i) => {
      const x = padding.left + i * columnWidth + columnWidth / 2;
      const y = scaleCumulativeY(cumEnergy);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw cumulative energy points
    ctx.fillStyle = lineColor;
    cumulativeEnergy.forEach((cumEnergy, i) => {
      const x = padding.left + i * columnWidth + columnWidth / 2;
      const y = scaleCumulativeY(cumEnergy);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw X-axis labels (token indices)
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    const labelInterval = Math.max(1, Math.floor(tokenCount / 10));
    tokens.forEach((token, i) => {
      if (i % labelInterval === 0 || i === tokenCount - 1) {
        const x = padding.left + i * columnWidth + columnWidth / 2;
        ctx.fillText(`${i}`, x, height - padding.bottom + 20);
      }
    });

    // Draw legend
    const legendX = padding.left;
    const legendY = 10;
    const legendItemWidth = 100;
    let legendOffset = 0;

    const legendItems = [
      { label: 'Attention', color: componentColors.attention },
      { label: 'MLP', color: componentColors.mlp },
      { label: 'LayerNorm', color: componentColors.layernorm },
      { label: 'Other', color: componentColors.other },
      { label: 'Cumulative', color: lineColor, isLine: true },
    ];

    legendItems.forEach((item) => {
      const x = legendX + legendOffset;

      if (item.isLine) {
        // Draw line for cumulative
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, legendY);
        ctx.lineTo(x + 15, legendY);
        ctx.stroke();
      } else {
        // Draw box for components
        ctx.fillStyle = item.color;
        ctx.fillRect(x, legendY - 5, 15, 10);
      }

      // Draw label
      ctx.fillStyle = '#374151';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(item.label, x + 20, legendY + 4);

      legendOffset += legendItemWidth;
    });

  }, [tokens, width, height, scrollLeft]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || tokens.length === 0) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = { top: 40, right: 60, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const tokenCount = tokens.length;
    const columnWidth = Math.max(20, Math.min(60, chartWidth / tokenCount));

    // Check if mouse is over a bar
    if (x >= padding.left && x <= padding.left + columnWidth * tokenCount &&
        y >= padding.top && y <= height - padding.bottom) {

      const tokenIndex = Math.floor((x - padding.left) / columnWidth);
      if (tokenIndex >= 0 && tokenIndex < tokens.length) {
        setTooltip({
          x: e.clientX,
          y: e.clientY,
          token: tokens[tokenIndex],
          tokenIndex,
        });
        return;
      }
    }

    setTooltip(null);
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollLeft(e.currentTarget.scrollLeft);
  };

  const tokenCount = tokens.length;
  const chartWidth = width - 120;
  const columnWidth = Math.max(20, Math.min(60, chartWidth / tokenCount));
  const actualWidth = columnWidth * tokenCount + 120;
  const needsScroll = actualWidth > width;

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="overflow-x-auto"
        style={{ width: `${width}px`, maxWidth: '100%' }}
        onScroll={handleScroll}
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            cursor: 'crosshair',
            minWidth: needsScroll ? `${actualWidth}px` : undefined,
          }}
        />
      </div>

      {tooltip && (
        <div
          className="absolute bg-gray-900 text-white text-xs rounded px-3 py-2 pointer-events-none z-10 shadow-lg"
          style={{
            left: `${tooltip.x + 10}px`,
            top: `${tooltip.y + 10}px`,
            maxWidth: '300px',
          }}
        >
          <div className="font-semibold mb-1">Token {tooltip.tokenIndex}</div>
          <div className="mb-1">Text: &quot;{tooltip.token.token_text}&quot;</div>
          <div className="mb-1">Position: {tooltip.token.token_position}</div>
          <div className="mb-1">Duration: {tooltip.token.duration_ms?.toFixed(2)}ms</div>
          <div className="mb-1">Energy: {tooltip.token.energy_mj?.toFixed(3)}mJ</div>
          {tooltip.token.layers && tooltip.token.layers.length > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-700">
              <div className="font-semibold mb-1">Component Breakdown:</div>
              <div>Layers: {tooltip.token.layers.length}</div>
              <div>
                Total Layer Energy: {
                  tooltip.token.layers.reduce((sum, l) => sum + (l.energy_mj || 0), 0).toFixed(3)
                }mJ
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
