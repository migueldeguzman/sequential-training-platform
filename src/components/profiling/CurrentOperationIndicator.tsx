'use client';

import React from 'react';
import { useProfilingContext } from './ProfilingContext';

/**
 * CurrentOperationIndicator Component
 *
 * Shows the current operation being profiled during inference with smooth transitions.
 * Displays:
 * - Current phase (pre_inference/prefill/decode/post_inference)
 * - Current section within the phase
 * - Visual progress indicator with phase-specific colors
 * - Animated transitions between operations
 *
 * Part of the Energy Profiler real-time monitoring interface.
 */

interface PhaseConfig {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: string;
}

const PHASE_CONFIGS: Record<string, PhaseConfig> = {
  idle: {
    label: 'Idle',
    color: 'text-gray-400',
    bgColor: 'bg-gray-900',
    borderColor: 'border-gray-700',
    icon: '⏸️',
  },
  pre_inference: {
    label: 'Pre-Inference',
    color: 'text-blue-400',
    bgColor: 'bg-blue-950',
    borderColor: 'border-blue-700',
    icon: '🔧',
  },
  prefill: {
    label: 'Prefill',
    color: 'text-purple-400',
    bgColor: 'bg-purple-950',
    borderColor: 'border-purple-700',
    icon: '📝',
  },
  decode: {
    label: 'Decode',
    color: 'text-green-400',
    bgColor: 'bg-green-950',
    borderColor: 'border-green-700',
    icon: '⚡',
  },
  post_inference: {
    label: 'Post-Inference',
    color: 'text-orange-400',
    bgColor: 'bg-orange-950',
    borderColor: 'border-orange-700',
    icon: '✅',
  },
};

export default function CurrentOperationIndicator() {
  const { currentSection, isProfiling, tokens } = useProfilingContext();

  // Determine current phase
  const phase = currentSection?.phase || (isProfiling ? 'idle' : 'idle');
  const sectionName = currentSection?.section_name || 'Waiting...';
  const config = PHASE_CONFIGS[phase] || PHASE_CONFIGS.idle;

  // Format section name for display (convert snake_case to Title Case)
  const formatSectionName = (name: string): string => {
    return name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Get current token info for decode phase
  const currentToken = tokens.length > 0 ? tokens[tokens.length - 1] : null;
  const isDecoding = phase === 'decode';

  return (
    <div className="w-full">
      {/* Main indicator card */}
      <div
        className={`
          relative overflow-hidden rounded-lg border-2 transition-all duration-500
          ${config.borderColor} ${config.bgColor}
        `}
      >
        {/* Animated progress bar */}
        {isProfiling && (
          <div className="absolute top-0 left-0 h-1 w-full overflow-hidden">
            <div
              className={`h-full animate-pulse ${config.color.replace('text-', 'bg-')}`}
              style={{
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
              }}
            />
          </div>
        )}

        {/* Content */}
        <div className="p-4">
          <div className="flex items-center justify-between">
            {/* Left: Phase info */}
            <div className="flex items-center gap-3">
              {/* Phase icon with animation */}
              <div
                className={`
                  text-2xl transform transition-transform duration-300
                  ${isProfiling ? 'scale-110 animate-bounce' : 'scale-100'}
                `}
                style={{
                  animation: isProfiling
                    ? 'bounce 1s ease-in-out infinite'
                    : 'none',
                }}
              >
                {config.icon}
              </div>

              {/* Phase and section text */}
              <div>
                <div className={`text-sm font-medium ${config.color}`}>
                  {config.label}
                </div>
                <div
                  className={`
                    text-lg font-bold text-white transition-all duration-300
                    ${isProfiling ? 'opacity-100' : 'opacity-60'}
                  `}
                >
                  {formatSectionName(sectionName)}
                </div>
              </div>
            </div>

            {/* Right: Additional context */}
            {isDecoding && currentToken && (
              <div className="text-right">
                <div className="text-xs text-gray-400">Current Token</div>
                <div className="text-sm font-medium text-white">
                  #{currentToken.token_position}
                  <span className="ml-2 text-gray-400">
                    &ldquo;{currentToken.token_text}&rdquo;
                  </span>
                </div>
                <div className="text-xs text-gray-500">
                  {currentToken.duration_ms.toFixed(1)}ms •{' '}
                  {currentToken.energy_mj.toFixed(2)}mJ
                </div>
              </div>
            )}

            {/* Status badge when not profiling */}
            {!isProfiling && (
              <div className="px-3 py-1 rounded-full bg-gray-800 border border-gray-700">
                <span className="text-xs font-medium text-gray-400">
                  Ready
                </span>
              </div>
            )}
          </div>

          {/* Token count indicator for decode phase */}
          {isDecoding && (
            <div className="mt-3 flex items-center gap-2">
              <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full ${config.color.replace('text-', 'bg-')} transition-all duration-300`}
                  style={{
                    width: `${Math.min((tokens.length / 100) * 100, 100)}%`,
                  }}
                />
              </div>
              <span className="text-xs text-gray-400 min-w-[60px] text-right">
                {tokens.length} tokens
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Optional: Timeline dots showing phase progression */}
      {isProfiling && (
        <div className="flex items-center justify-center gap-2 mt-4">
          {Object.entries(PHASE_CONFIGS)
            .filter(([key]) => key !== 'idle')
            .map(([key, phaseConfig]) => {
              const isActive = key === phase;
              const isPast =
                ['pre_inference', 'prefill', 'decode', 'post_inference'].indexOf(
                  key
                ) <
                ['pre_inference', 'prefill', 'decode', 'post_inference'].indexOf(
                  phase
                );

              return (
                <div key={key} className="flex items-center">
                  {/* Phase dot */}
                  <div
                    className={`
                      h-3 w-3 rounded-full transition-all duration-300
                      ${
                        isActive
                          ? `${phaseConfig.color.replace('text-', 'bg-')} scale-125`
                          : isPast
                          ? 'bg-gray-600'
                          : 'bg-gray-800'
                      }
                    `}
                    title={phaseConfig.label}
                  />
                  {/* Connector line */}
                  {key !== 'post_inference' && (
                    <div
                      className={`
                        h-0.5 w-8 transition-all duration-300
                        ${isPast ? 'bg-gray-600' : 'bg-gray-800'}
                      `}
                    />
                  )}
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}
