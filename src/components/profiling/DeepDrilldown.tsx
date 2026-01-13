'use client';

import React from 'react';
import { ComponentMetrics, DeepOperationMetrics } from '@/types';

interface DeepDrilldownProps {
  componentMetrics: ComponentMetrics | null;
  deepOperations: DeepOperationMetrics[];
  layerIndex: number;
  onClose: () => void;
}

export default function DeepDrilldown({
  componentMetrics,
  deepOperations,
  layerIndex,
  onClose,
}: DeepDrilldownProps) {
  if (!componentMetrics) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center bg-gradient-to-r from-blue-50 to-indigo-50">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Deep Operation Metrics</h2>
            <p className="text-sm text-gray-600 mt-1">
              Layer {layerIndex} - {componentMetrics.component_name}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Close"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Component Summary */}
          <div className="mb-6 bg-gray-50 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Component Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <div className="text-xs text-gray-500">Duration</div>
                <div className="text-lg font-mono text-gray-900">
                  {componentMetrics.duration_ms.toFixed(3)} ms
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Activation Mean</div>
                <div className="text-lg font-mono text-gray-900">
                  {componentMetrics.activation_mean.toFixed(4)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Activation Max</div>
                <div className="text-lg font-mono text-gray-900">
                  {componentMetrics.activation_max.toFixed(4)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Activation Std</div>
                <div className="text-lg font-mono text-gray-900">
                  {componentMetrics.activation_std.toFixed(4)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Sparsity</div>
                <div className="text-lg font-mono text-gray-900">
                  {(componentMetrics.sparsity * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>

          {/* Deep Operations */}
          {deepOperations.length > 0 ? (
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">
                Deep Operations Breakdown
              </h3>
              <div className="space-y-3">
                {deepOperations.map((operation) => (
                  <div
                    key={operation.id}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h4 className="font-semibold text-gray-900">
                        {formatOperationName(operation.operation_name)}
                      </h4>
                      <span className="text-sm font-mono text-gray-600">
                        {operation.duration_ms.toFixed(3)} ms
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      {/* Attention Metrics */}
                      {operation.attention_entropy !== undefined &&
                        operation.attention_entropy !== null && (
                          <MetricCard
                            label="Attention Entropy"
                            value={operation.attention_entropy}
                            format={(v) => v.toFixed(4)}
                            tooltip="Higher entropy means more distributed attention"
                            color="blue"
                          />
                        )}
                      {operation.max_attention_weight !== undefined &&
                        operation.max_attention_weight !== null && (
                          <MetricCard
                            label="Max Attention Weight"
                            value={operation.max_attention_weight}
                            format={(v) => v.toFixed(4)}
                            tooltip="Peak attention value across heads"
                            color="indigo"
                          />
                        )}
                      {operation.attention_sparsity !== undefined &&
                        operation.attention_sparsity !== null && (
                          <MetricCard
                            label="Attention Sparsity"
                            value={operation.attention_sparsity}
                            format={(v) => `${(v * 100).toFixed(2)}%`}
                            tooltip="Percentage of near-zero attention weights"
                            color="purple"
                          />
                        )}

                      {/* MLP Metrics */}
                      {operation.activation_kill_ratio !== undefined &&
                        operation.activation_kill_ratio !== null && (
                          <MetricCard
                            label="Activation Kill Ratio"
                            value={operation.activation_kill_ratio}
                            format={(v) => `${(v * 100).toFixed(2)}%`}
                            tooltip="Percentage of activations zeroed by GELU/SiLU"
                            color="orange"
                          />
                        )}

                      {/* LayerNorm Metrics */}
                      {operation.variance_ratio !== undefined &&
                        operation.variance_ratio !== null && (
                          <MetricCard
                            label="Variance Ratio"
                            value={operation.variance_ratio}
                            format={(v) => v.toFixed(4)}
                            tooltip="Output variance / Input variance"
                            color="green"
                          />
                        )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <svg
                className="w-12 h-12 mx-auto mb-3 text-gray-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <p className="text-sm">No deep operation metrics available</p>
              <p className="text-xs mt-1">
                Enable deep profiling mode to capture detailed operation metrics
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// Helper Components
interface MetricCardProps {
  label: string;
  value: number;
  format: (value: number) => string;
  tooltip: string;
  color: 'blue' | 'indigo' | 'purple' | 'orange' | 'green';
}

function MetricCard({ label, value, format, tooltip, color }: MetricCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    indigo: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    purple: 'bg-purple-50 text-purple-700 border-purple-200',
    orange: 'bg-orange-50 text-orange-700 border-orange-200',
    green: 'bg-green-50 text-green-700 border-green-200',
  };

  return (
    <div
      className={`border rounded p-3 ${colorClasses[color]}`}
      title={tooltip}
    >
      <div className="text-xs font-medium opacity-75 mb-1">{label}</div>
      <div className="text-base font-mono font-semibold">{format(value)}</div>
    </div>
  );
}

// Helper Functions
function formatOperationName(name: string): string {
  // Convert snake_case to Title Case
  return name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
