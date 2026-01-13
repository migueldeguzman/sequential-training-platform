'use client';

import React from 'react';

export type MetricType =
  | 'time'
  | 'energy'
  | 'power'
  | 'activation_mean'
  | 'activation_max'
  | 'sparsity'
  | 'attention_entropy';

export interface MetricOption {
  value: MetricType;
  label: string;
  unit: string;
  description: string;
  requiresDeepProfiling?: boolean;
}

export const METRIC_OPTIONS: MetricOption[] = [
  {
    value: 'time',
    label: 'Time',
    unit: 'ms',
    description: 'Execution time in milliseconds',
  },
  {
    value: 'energy',
    label: 'Energy',
    unit: 'mJ',
    description: 'Energy consumption in millijoules',
  },
  {
    value: 'power',
    label: 'Power',
    unit: 'mW',
    description: 'Power draw in milliwatts',
  },
  {
    value: 'activation_mean',
    label: 'Activation Mean',
    unit: '',
    description: 'Mean absolute value of activations',
  },
  {
    value: 'activation_max',
    label: 'Activation Max',
    unit: '',
    description: 'Maximum absolute value of activations',
  },
  {
    value: 'sparsity',
    label: 'Sparsity',
    unit: '%',
    description: 'Percentage of near-zero activations',
  },
  {
    value: 'attention_entropy',
    label: 'Attention Entropy',
    unit: '',
    description: 'Entropy of attention weights (deep profiling only)',
    requiresDeepProfiling: true,
  },
];

export interface MetricSelectorProps {
  selectedMetric: MetricType;
  onMetricChange: (metric: MetricType) => void;
  isDeepProfiling?: boolean;
  disabled?: boolean;
  className?: string;
}

export function MetricSelector({
  selectedMetric,
  onMetricChange,
  isDeepProfiling = false,
  disabled = false,
  className = '',
}: MetricSelectorProps) {
  // Filter options based on deep profiling availability
  const availableOptions = METRIC_OPTIONS.filter(
    (option) => !option.requiresDeepProfiling || isDeepProfiling
  );

  const selectedOption = METRIC_OPTIONS.find((opt) => opt.value === selectedMetric);

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-4 ${className}`}>
      <div className="space-y-3">
        <div>
          <label htmlFor="metric-selector" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Select Metric
          </label>
          <select
            id="metric-selector"
            value={selectedMetric}
            onChange={(e) => onMetricChange(e.target.value as MetricType)}
            disabled={disabled}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          >
            {availableOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label} {option.unit ? `(${option.unit})` : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Metric Description */}
        {selectedOption && (
          <div className="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700/50 rounded p-2">
            <span className="font-medium">Description: </span>
            {selectedOption.description}
          </div>
        )}

        {/* Warning for deep profiling metrics */}
        {selectedOption?.requiresDeepProfiling && !isDeepProfiling && (
          <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded p-2">
            <span className="font-medium">Note: </span>
            This metric requires deep profiling to be enabled
          </div>
        )}
      </div>
    </div>
  );
}
