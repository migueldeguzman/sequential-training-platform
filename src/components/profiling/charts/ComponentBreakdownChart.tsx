'use client';

import React, { useRef, useEffect, useState } from 'react';

interface ComponentBreakdown {
  cpu_energy_mj: number;
  gpu_energy_mj: number;
  ane_energy_mj: number;
  dram_energy_mj: number;
  total_energy_mj: number;
  cpu_energy_percentage: number;
  gpu_energy_percentage: number;
  ane_energy_percentage: number;
  dram_energy_percentage: number;
  avg_cpu_power_mw: number;
  avg_gpu_power_mw: number;
  avg_ane_power_mw: number;
  avg_dram_power_mw: number;
  peak_cpu_power_mw: number;
  peak_gpu_power_mw: number;
  peak_ane_power_mw: number;
  peak_dram_power_mw: number;
}

interface PhaseComponent {
  phase: string;
  cpu_energy_mj: number;
  gpu_energy_mj: number;
  ane_energy_mj: number;
  dram_energy_mj: number;
  total_energy_mj: number;
}

interface ComponentBreakdownChartProps {
  breakdown: ComponentBreakdown | null;
  phaseBreakdown?: PhaseComponent[];
  viewMode?: 'total' | 'by-phase';
}

const ComponentBreakdownChart: React.FC<ComponentBreakdownChartProps> = ({
  breakdown,
  phaseBreakdown = [],
  viewMode = 'total'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredSegment, setHoveredSegment] = useState<{
    component: string;
    value: number;
    percentage: number;
    phase?: string;
  } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || (!breakdown && phaseBreakdown.length === 0)) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (viewMode === 'total' && breakdown) {
      drawTotalBreakdown(ctx, breakdown, width, height);
    } else if (viewMode === 'by-phase' && phaseBreakdown.length > 0) {
      drawPhaseBreakdown(ctx, phaseBreakdown, width, height);
    }
  }, [breakdown, phaseBreakdown, viewMode]);

  const drawTotalBreakdown = (
    ctx: CanvasRenderingContext2D,
    breakdown: ComponentBreakdown,
    width: number,
    height: number
  ) => {
    const padding = { top: 40, right: 40, bottom: 60, left: 70 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Component data
    const components = [
      { name: 'CPU', energy: breakdown.cpu_energy_mj, percentage: breakdown.cpu_energy_percentage, color: '#3B82F6' },
      { name: 'GPU', energy: breakdown.gpu_energy_mj, percentage: breakdown.gpu_energy_percentage, color: '#10B981' },
      { name: 'ANE', energy: breakdown.ane_energy_mj, percentage: breakdown.ane_energy_percentage, color: '#F59E0B' },
      { name: 'DRAM', energy: breakdown.dram_energy_mj, percentage: breakdown.dram_energy_percentage, color: '#8B5CF6' }
    ];

    // Draw title
    ctx.fillStyle = '#F9FAFB';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Component Energy Breakdown', width / 2, 25);

    // Calculate bar dimensions
    const barWidth = chartWidth * 0.6;
    const barHeight = 60;
    const barX = padding.left + (chartWidth - barWidth) / 2;
    const barY = padding.top + (chartHeight - barHeight) / 2;

    // Draw stacked bar
    let currentX = barX;
    components.forEach((comp) => {
      const segmentWidth = (comp.energy / breakdown.total_energy_mj) * barWidth;

      // Draw segment
      ctx.fillStyle = comp.color;
      ctx.fillRect(currentX, barY, segmentWidth, barHeight);

      // Draw border
      ctx.strokeStyle = '#1F2937';
      ctx.lineWidth = 2;
      ctx.strokeRect(currentX, barY, segmentWidth, barHeight);

      currentX += segmentWidth;
    });

    // Draw legend and statistics below the bar
    const legendY = barY + barHeight + 40;
    const legendItemWidth = chartWidth / 4;

    components.forEach((comp, index) => {
      const legendX = padding.left + index * legendItemWidth;

      // Draw color box
      ctx.fillStyle = comp.color;
      ctx.fillRect(legendX, legendY, 20, 20);
      ctx.strokeStyle = '#4B5563';
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, legendY, 20, 20);

      // Draw component name
      ctx.fillStyle = '#F9FAFB';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(comp.name, legendX + 30, legendY + 15);

      // Draw energy value
      ctx.font = '12px sans-serif';
      ctx.fillStyle = '#D1D5DB';
      ctx.fillText(`${comp.energy.toFixed(1)} mJ`, legendX + 30, legendY + 35);

      // Draw percentage
      ctx.fillStyle = '#9CA3AF';
      ctx.fillText(`${comp.percentage.toFixed(1)}%`, legendX + 30, legendY + 50);
    });

    // Draw total energy at the top right
    ctx.fillStyle = '#F9FAFB';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`Total: ${breakdown.total_energy_mj.toFixed(1)} mJ`, width - padding.right, padding.top);
  };

  const drawPhaseBreakdown = (
    ctx: CanvasRenderingContext2D,
    phaseBreakdown: PhaseComponent[],
    width: number,
    height: number
  ) => {
    const padding = { top: 60, right: 40, bottom: 80, left: 100 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Draw title
    ctx.fillStyle = '#F9FAFB';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Component Energy by Phase', width / 2, 25);

    // Component colors
    const componentColors = {
      cpu: '#3B82F6',
      gpu: '#10B981',
      ane: '#F59E0B',
      dram: '#8B5CF6'
    };

    // Find max total energy for scaling
    const maxEnergy = Math.max(...phaseBreakdown.map(p => p.total_energy_mj));

    // Draw bars for each phase
    const barWidth = chartWidth / phaseBreakdown.length * 0.7;
    const barSpacing = chartWidth / phaseBreakdown.length;

    phaseBreakdown.forEach((phase, index) => {
      const barX = padding.left + index * barSpacing + (barSpacing - barWidth) / 2;
      const barMaxHeight = chartHeight;

      // Draw stacked bar
      let currentY = height - padding.bottom;

      // Stack components from bottom to top
      const components = [
        { name: 'cpu', energy: phase.cpu_energy_mj, color: componentColors.cpu },
        { name: 'gpu', energy: phase.gpu_energy_mj, color: componentColors.gpu },
        { name: 'ane', energy: phase.ane_energy_mj, color: componentColors.ane },
        { name: 'dram', energy: phase.dram_energy_mj, color: componentColors.dram }
      ];

      components.forEach((comp) => {
        const segmentHeight = (comp.energy / maxEnergy) * barMaxHeight;

        // Draw segment
        ctx.fillStyle = comp.color;
        ctx.fillRect(barX, currentY - segmentHeight, barWidth, segmentHeight);

        // Draw border
        ctx.strokeStyle = '#1F2937';
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, currentY - segmentHeight, barWidth, segmentHeight);

        currentY -= segmentHeight;
      });

      // Draw phase label
      ctx.fillStyle = '#F9FAFB';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.save();
      ctx.translate(barX + barWidth / 2, height - padding.bottom + 20);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(phase.phase, 0, 0);
      ctx.restore();

      // Draw total value on top
      ctx.fillStyle = '#D1D5DB';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${phase.total_energy_mj.toFixed(1)}`, barX + barWidth / 2, currentY - 5);
    });

    // Draw Y-axis
    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    // Draw Y-axis label
    ctx.fillStyle = '#F9FAFB';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy (mJ)', 0, 0);
    ctx.restore();

    // Draw Y-axis ticks
    const tickCount = 5;
    for (let i = 0; i <= tickCount; i++) {
      const y = height - padding.bottom - (chartHeight / tickCount) * i;
      const value = (maxEnergy / tickCount) * i;

      // Draw tick
      ctx.strokeStyle = '#4B5563';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left - 5, y);
      ctx.lineTo(padding.left, y);
      ctx.stroke();

      // Draw label
      ctx.fillStyle = '#D1D5DB';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(value.toFixed(0), padding.left - 10, y + 4);
    }

    // Draw legend
    const legendX = padding.left;
    const legendY = 45;
    const legendItemWidth = 100;

    Object.entries(componentColors).forEach(([name, color], index) => {
      const itemX = legendX + index * legendItemWidth;

      // Draw color box
      ctx.fillStyle = color;
      ctx.fillRect(itemX, legendY, 15, 15);
      ctx.strokeStyle = '#4B5563';
      ctx.lineWidth = 1;
      ctx.strokeRect(itemX, legendY, 15, 15);

      // Draw name
      ctx.fillStyle = '#F9FAFB';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(name.toUpperCase(), itemX + 20, legendY + 12);
    });
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    setMousePos({ x: event.clientX, y: event.clientY });
    // TODO: Implement hover detection for segments based on mouse position
  };

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ minHeight: '400px' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredSegment(null)}
      />
      {hoveredSegment && (
        <div
          className="absolute bg-gray-900 text-white p-2 rounded shadow-lg text-sm pointer-events-none z-10"
          style={{
            left: mousePos.x + 10,
            top: mousePos.y + 10,
          }}
        >
          <div className="font-bold">{hoveredSegment.component}</div>
          {hoveredSegment.phase && <div className="text-gray-400">{hoveredSegment.phase}</div>}
          <div>{hoveredSegment.value.toFixed(2)} mJ</div>
          <div className="text-gray-400">{hoveredSegment.percentage.toFixed(1)}%</div>
        </div>
      )}
    </div>
  );
};

export default ComponentBreakdownChart;
