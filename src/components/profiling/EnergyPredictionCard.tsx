'use client';

import React, { useState, useCallback } from 'react';
import { profilingApi } from '@/lib/api';
import type { EnergyPrediction } from '@/types';

interface EnergyPredictionCardProps {
  /** Pre-fill with the currently selected model path */
  modelPath?: string;
}

export function EnergyPredictionCard({ modelPath }: EnergyPredictionCardProps) {
  const [modelName, setModelName] = useState(modelPath || '');
  const [inputTokens, setInputTokens] = useState(100);
  const [outputTokens, setOutputTokens] = useState(50);
  const [batchSize, setBatchSize] = useState(1);
  const [prediction, setPrediction] = useState<EnergyPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Sync if parent changes modelPath
  React.useEffect(() => {
    if (modelPath) setModelName(modelPath);
  }, [modelPath]);

  const handlePredict = useCallback(async () => {
    if (!modelName.trim()) {
      setError('Model name is required');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await profilingApi.predictEnergy({
        model_name: modelName.trim(),
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        batch_size: batchSize,
      });

      if (response.success && response.data) {
        setPrediction(response.data);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to predict energy');
    } finally {
      setLoading(false);
    }
  }, [modelName, inputTokens, outputTokens, batchSize]);

  const formatEnergy = (mj: number): string => {
    if (mj >= 1000) return `${(mj / 1000).toFixed(1)} J`;
    return `${mj.toFixed(1)} mJ`;
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-lg">⚡</span>
        <h3 className="text-sm font-semibold text-gray-900">Energy Prediction</h3>
        <span className="text-xs text-gray-500">Estimate before profiling</span>
      </div>

      {/* Input Form */}
      <div className="grid grid-cols-2 gap-3">
        <div className="col-span-2">
          <label className="block text-xs font-medium text-gray-600 mb-1">
            Model Name / Path
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., meta-llama/Llama-3.2-1B"
            className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            Input Tokens
          </label>
          <input
            type="number"
            value={inputTokens}
            onChange={(e) => setInputTokens(Math.max(1, parseInt(e.target.value) || 1))}
            min={1}
            className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            Output Tokens
          </label>
          <input
            type="number"
            value={outputTokens}
            onChange={(e) => setOutputTokens(Math.max(1, parseInt(e.target.value) || 1))}
            min={1}
            className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            Batch Size
          </label>
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
            min={1}
            className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="flex items-end">
          <button
            onClick={handlePredict}
            disabled={loading || !modelName.trim()}
            className="w-full px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-3">
          <p className="text-xs text-red-700">{error}</p>
        </div>
      )}

      {/* Results */}
      {prediction && (
        <div className="space-y-3">
          {/* Main prediction */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-md p-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-gray-500">Total Energy</p>
                <p className="text-lg font-bold text-blue-700">
                  {formatEnergy(prediction.predicted_total_energy_mj)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Per Token</p>
                <p className="text-lg font-bold text-indigo-700">
                  {formatEnergy(prediction.predicted_energy_per_token_mj)}
                </p>
              </div>
            </div>
          </div>

          {/* Phase breakdown */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-50 rounded-md p-2">
              <p className="text-xs text-gray-500">Prefill</p>
              <p className="text-sm font-semibold text-gray-800">
                {formatEnergy(prediction.predicted_prefill_energy_mj)}
              </p>
            </div>
            <div className="bg-gray-50 rounded-md p-2">
              <p className="text-xs text-gray-500">Decode</p>
              <p className="text-sm font-semibold text-gray-800">
                {formatEnergy(prediction.predicted_decode_energy_mj)}
              </p>
            </div>
          </div>

          {/* Confidence & accuracy */}
          <div className="flex justify-between items-center text-xs text-gray-500">
            <span>
              95% CI: {formatEnergy(prediction.confidence_interval_95_pct[0])} –{' '}
              {formatEnergy(prediction.confidence_interval_95_pct[1])}
            </span>
            <span>
              Model R²: {prediction.model_accuracy_r2.toFixed(3)}
            </span>
          </div>

          {/* Prefill/decode proportion bar */}
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden flex">
            <div
              className="bg-blue-400 h-full"
              style={{
                width: `${(prediction.predicted_prefill_energy_mj / prediction.predicted_total_energy_mj) * 100}%`,
              }}
              title={`Prefill: ${((prediction.predicted_prefill_energy_mj / prediction.predicted_total_energy_mj) * 100).toFixed(0)}%`}
            />
            <div
              className="bg-indigo-500 h-full"
              style={{
                width: `${(prediction.predicted_decode_energy_mj / prediction.predicted_total_energy_mj) * 100}%`,
              }}
              title={`Decode: ${((prediction.predicted_decode_energy_mj / prediction.predicted_total_energy_mj) * 100).toFixed(0)}%`}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-400">
            <span>Prefill</span>
            <span>Decode</span>
          </div>

          {/* Notes */}
          {prediction.prediction_notes && (
            <p className="text-xs text-amber-600 bg-amber-50 rounded-md p-2">
              ⚠ {prediction.prediction_notes}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default EnergyPredictionCard;
