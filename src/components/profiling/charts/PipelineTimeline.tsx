'use client';

import React, { useRef, useEffect, useState } from 'react';
import type { PipelineSection } from '@/types';

interface PipelineTimelineProps {
  sections: PipelineSection[];
  width?: number;
  height?: number;
  className?: string;
  onSectionClick?: (section: PipelineSection) => void;
}

interface TimelineConfig {
  padding: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  barHeight: number;
  colors: {
    pre_inference: string;
    prefill: string;
    decode: string;
    post_inference: string;
    text: string;
    background: string;
    border: string;
    hover: string;
  };
}

const DEFAULT_CONFIG: TimelineConfig = {
  padding: {
    top: 20,
    right: 20,
    bottom: 60,
    left: 20,
  },
  barHeight: 80,
  colors: {
    pre_inference: '#6366f1', // indigo
    prefill: '#8b5cf6', // violet
    decode: '#ec4899', // pink
    post_inference: '#14b8a6', // teal
    text: '#f9fafb',
    background: '#1f2937',
    border: '#374151',
    hover: '#4b5563',
  },
};

export function PipelineTimeline({
  sections,
  width = 1000,
  height = 200,
  className = '',
  onSectionClick,
}: PipelineTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredSection, setHoveredSection] = useState<PipelineSection | null>(null);
  const [hoveredPosition, setHoveredPosition] = useState<{ x: number; y: number } | null>(null);

  // Group sections by phase
  const phaseGroups = sections.reduce((acc, section) => {
    if (!acc[section.phase]) {
      acc[section.phase] = [];
    }
    acc[section.phase].push(section);
    return acc;
  }, {} as Record<string, PipelineSection[]>);

  // Calculate total duration
  const totalDuration = sections.length > 0
    ? Math.max(...sections.map(s => s.end_time))
    : 0;

  // Calculate phase aggregates
  const phaseData = ['pre_inference', 'prefill', 'decode', 'post_inference'].map(phase => {
    const phaseSections = phaseGroups[phase] || [];
    const startTime = phaseSections.length > 0
      ? Math.min(...phaseSections.map(s => s.start_time))
      : 0;
    const endTime = phaseSections.length > 0
      ? Math.max(...phaseSections.map(s => s.end_time))
      : 0;
    const duration = endTime - startTime;
    const energy = phaseSections.reduce((sum, s) => sum + s.energy_mj, 0);

    return {
      phase,
      startTime,
      endTime,
      duration,
      energy,
      sections: phaseSections,
    };
  }).filter(p => p.duration > 0);

  // Mouse move handler for hover effects
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || phaseData.length === 0) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const chartWidth = width - DEFAULT_CONFIG.padding.left - DEFAULT_CONFIG.padding.right;
    const barY = DEFAULT_CONFIG.padding.top;
    const barHeight = DEFAULT_CONFIG.barHeight;

    // Check if mouse is within a bar
    if (
      x >= DEFAULT_CONFIG.padding.left &&
      x <= width - DEFAULT_CONFIG.padding.right &&
      y >= barY &&
      y <= barY + barHeight
    ) {
      // Find which phase is hovered
      const relativeX = x - DEFAULT_CONFIG.padding.left;
      let accumulatedX = 0;

      for (const phase of phaseData) {
        const phaseWidth = (phase.duration / totalDuration) * chartWidth;

        if (relativeX >= accumulatedX && relativeX <= accumulatedX + phaseWidth) {
          // Find the specific section if there are multiple in this phase
          const sectionHovered = phase.sections[0]; // For now, use first section
          setHoveredSection(sectionHovered);
          setHoveredPosition({ x, y });
          return;
        }

        accumulatedX += phaseWidth;
      }
    }

    setHoveredSection(null);
    setHoveredPosition(null);
  };

  const handleMouseLeave = () => {
    setHoveredSection(null);
    setHoveredPosition(null);
  };

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !onSectionClick || phaseData.length === 0) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const chartWidth = width - DEFAULT_CONFIG.padding.left - DEFAULT_CONFIG.padding.right;
    const barY = DEFAULT_CONFIG.padding.top;
    const barHeight = DEFAULT_CONFIG.barHeight;

    // Check if click is within a bar
    if (
      x >= DEFAULT_CONFIG.padding.left &&
      x <= width - DEFAULT_CONFIG.padding.right &&
      y >= barY &&
      y <= barY + barHeight
    ) {
      // Find which phase was clicked
      const relativeX = x - DEFAULT_CONFIG.padding.left;
      let accumulatedX = 0;

      for (const phase of phaseData) {
        const phaseWidth = (phase.duration / totalDuration) * chartWidth;

        if (relativeX >= accumulatedX && relativeX <= accumulatedX + phaseWidth) {
          const sectionClicked = phase.sections[0];
          onSectionClick(sectionClicked);
          return;
        }

        accumulatedX += phaseWidth;
      }
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || phaseData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution for sharp rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = DEFAULT_CONFIG.colors.background;
    ctx.fillRect(0, 0, width, height);

    const chartWidth = width - DEFAULT_CONFIG.padding.left - DEFAULT_CONFIG.padding.right;
    const barY = DEFAULT_CONFIG.padding.top;
    const barHeight = DEFAULT_CONFIG.barHeight;

    let accumulatedX = DEFAULT_CONFIG.padding.left;

    // Draw phase bars
    phaseData.forEach(phase => {
      const phaseWidth = (phase.duration / totalDuration) * chartWidth;
      const phaseColor = DEFAULT_CONFIG.colors[phase.phase as keyof typeof DEFAULT_CONFIG.colors] || '#6b7280';

      // Calculate color intensity based on energy
      const maxEnergy = Math.max(...phaseData.map(p => p.energy));
      const energyRatio = phase.energy / maxEnergy;
      const alpha = 0.5 + (energyRatio * 0.5); // Range from 0.5 to 1.0

      // Draw bar with energy-based alpha
      ctx.fillStyle = phaseColor + Math.round(alpha * 255).toString(16).padStart(2, '0');
      ctx.fillRect(accumulatedX, barY, phaseWidth, barHeight);

      // Draw hover overlay
      if (hoveredSection && hoveredSection.phase === phase.phase) {
        ctx.fillStyle = DEFAULT_CONFIG.colors.hover + '40'; // 25% opacity
        ctx.fillRect(accumulatedX, barY, phaseWidth, barHeight);
      }

      // Draw border
      ctx.strokeStyle = DEFAULT_CONFIG.colors.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(accumulatedX, barY, phaseWidth, barHeight);

      // Draw phase label in center of bar
      const labelX = accumulatedX + phaseWidth / 2;
      const labelY = barY + barHeight / 2;

      ctx.fillStyle = DEFAULT_CONFIG.colors.text;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Format phase name
      const phaseName = phase.phase.split('_').map(word =>
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
      ctx.fillText(phaseName, labelX, labelY - 15);

      // Draw duration
      ctx.font = '12px sans-serif';
      ctx.fillText(`${phase.duration.toFixed(1)}ms`, labelX, labelY);

      // Draw energy
      ctx.fillText(`${phase.energy.toFixed(2)}mJ`, labelX, labelY + 15);

      // Draw percentage
      const percentage = ((phase.duration / totalDuration) * 100).toFixed(1);
      ctx.font = 'bold 11px sans-serif';
      ctx.fillStyle = DEFAULT_CONFIG.colors.text + 'aa'; // Slightly transparent
      ctx.fillText(`${percentage}%`, labelX, labelY + 30);

      accumulatedX += phaseWidth;
    });

    // Draw timeline axis below bars
    const axisY = barY + barHeight + 20;
    ctx.strokeStyle = DEFAULT_CONFIG.colors.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(DEFAULT_CONFIG.padding.left, axisY);
    ctx.lineTo(width - DEFAULT_CONFIG.padding.right, axisY);
    ctx.stroke();

    // Draw time markers
    const numMarkers = 5;
    for (let i = 0; i <= numMarkers; i++) {
      const markerX = DEFAULT_CONFIG.padding.left + (i / numMarkers) * chartWidth;
      const timeValue = (i / numMarkers) * totalDuration;

      // Draw tick
      ctx.beginPath();
      ctx.moveTo(markerX, axisY);
      ctx.lineTo(markerX, axisY + 5);
      ctx.stroke();

      // Draw label
      ctx.fillStyle = DEFAULT_CONFIG.colors.text;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${timeValue.toFixed(0)}ms`, markerX, axisY + 8);
    }

  }, [sections, width, height, hoveredSection, phaseData, totalDuration]);

  return (
    <div className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="cursor-pointer"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        style={{ width: `${width}px`, height: `${height}px` }}
      />

      {/* Tooltip */}
      {hoveredSection && hoveredPosition && (
        <div
          className="absolute pointer-events-none bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm shadow-lg z-10"
          style={{
            left: `${hoveredPosition.x + 10}px`,
            top: `${hoveredPosition.y - 80}px`,
          }}
        >
          <div className="font-bold text-white mb-1">
            {hoveredSection.phase.split('_').map(word =>
              word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ')}
          </div>
          <div className="text-gray-300">
            <div>Section: {hoveredSection.section_name}</div>
            <div>Duration: {hoveredSection.duration_ms.toFixed(2)}ms</div>
            <div>Energy: {hoveredSection.energy_mj.toFixed(2)}mJ</div>
            <div>Time: {hoveredSection.start_time.toFixed(0)}-{hoveredSection.end_time.toFixed(0)}ms</div>
          </div>
        </div>
      )}
    </div>
  );
}
