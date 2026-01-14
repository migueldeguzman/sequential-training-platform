/**
 * EP-089: Cost and Carbon Estimation Component
 *
 * Displays electricity cost and CO2 emissions estimates for a profiling run.
 * Based on TokenPowerBench methodology for sustainability reporting.
 *
 * Key metrics:
 * - Cost in USD: Calculated from energy consumption and electricity price
 * - CO2 emissions: Calculated from energy and regional carbon intensity
 * - Per-token costs: Normalized by token count
 * - Contextual comparisons: Car miles equivalent for CO2
 */

import React from 'react';
import { ProfilingRunSummary } from '@/types';

interface CostCarbonCardProps {
  summary: ProfilingRunSummary | null;
  className?: string;
}

export const CostCarbonCard: React.FC<CostCarbonCardProps> = ({
  summary,
  className = '',
}) => {
  if (!summary || !summary.cost_carbon_metrics) {
    return (
      <div className={`p-6 bg-white rounded-lg shadow ${className}`}>
        <h3 className="text-lg font-semibold mb-4">Cost & Carbon Footprint</h3>
        <p className="text-gray-500">No cost/carbon data available</p>
      </div>
    );
  }

  const metrics = summary.cost_carbon_metrics;

  // Helper to format numbers with proper precision
  const formatNumber = (value: number | null, decimals: number = 2): string => {
    if (value === null || value === undefined) return 'N/A';
    return value.toFixed(decimals);
  };

  // Helper to format currency
  const formatCurrency = (value: number | null): string => {
    if (value === null || value === undefined) return 'N/A';
    if (value < 0.01) {
      // Show in micro-dollars if very small
      return `$${(value * 1000000).toFixed(2)}µ`;
    }
    if (value < 1) {
      return `$${(value * 1000).toFixed(3)}m`;
    }
    return `$${value.toFixed(4)}`;
  };

  // Helper to format CO2 with appropriate unit
  const formatCO2 = (grams: number | null): string => {
    if (grams === null || grams === undefined) return 'N/A';
    if (grams < 1) {
      return `${(grams * 1000).toFixed(2)} mg`;
    }
    if (grams < 1000) {
      return `${grams.toFixed(2)} g`;
    }
    return `${(grams / 1000).toFixed(3)} kg`;
  };

  // Get carbon intensity rating
  const getCarbonIntensityRating = (gPerKwh: number): { label: string; color: string } => {
    // Based on global carbon intensity standards
    if (gPerKwh < 100) return { label: 'Very Low', color: 'text-green-600' };
    if (gPerKwh < 300) return { label: 'Low', color: 'text-blue-600' };
    if (gPerKwh < 500) return { label: 'Medium', color: 'text-yellow-600' };
    if (gPerKwh < 700) return { label: 'High', color: 'text-orange-600' };
    return { label: 'Very High', color: 'text-red-600' };
  };

  const carbonRating = getCarbonIntensityRating(metrics.carbon_intensity_g_per_kwh);

  // Get regional presets info
  const getRegionalInfo = (carbonIntensity: number): string => {
    // Common regional carbon intensities (approximate)
    if (Math.abs(carbonIntensity - 50) < 10) return 'Iceland/Norway (hydro)';
    if (Math.abs(carbonIntensity - 100) < 20) return 'France (nuclear)';
    if (Math.abs(carbonIntensity - 250) < 30) return 'California';
    if (Math.abs(carbonIntensity - 400) < 50) return 'US Average';
    if (Math.abs(carbonIntensity - 500) < 50) return 'EU Average';
    if (Math.abs(carbonIntensity - 700) < 100) return 'China/India (coal)';
    return 'Custom';
  };

  const regionalInfo = getRegionalInfo(metrics.carbon_intensity_g_per_kwh);

  return (
    <div className={`p-6 bg-white rounded-lg shadow ${className}`}>
      <h3 className="text-lg font-semibold mb-4">Cost & Carbon Footprint</h3>

      {/* Primary metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Total Cost */}
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-gray-600 mb-1">Total Cost</p>
          <p className="text-2xl font-bold text-blue-600">
            {formatCurrency(metrics.cost_usd)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            @ ${formatNumber(metrics.electricity_price_per_kwh, 2)}/kWh
          </p>
        </div>

        {/* Total CO2 */}
        <div className="p-4 bg-green-50 rounded-lg border border-green-200">
          <p className="text-sm text-gray-600 mb-1">CO₂ Emissions</p>
          <p className="text-2xl font-bold text-green-600">
            {formatCO2(metrics.co2_grams)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {formatNumber(metrics.co2_kg, 6)} kg total
          </p>
        </div>
      </div>

      {/* Per-token metrics */}
      <div className="mb-6 p-4 bg-gray-50 rounded">
        <p className="text-sm font-medium text-gray-700 mb-3">Per-Token Costs</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-600 mb-1">Cost per Token</p>
            <p className="text-lg font-semibold text-blue-600">
              {formatCurrency(metrics.cost_per_token_usd)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600 mb-1">CO₂ per Token</p>
            <p className="text-lg font-semibold text-green-600">
              {formatCO2(metrics.co2_per_token_grams)}
            </p>
          </div>
        </div>
      </div>

      {/* Carbon Intensity Info */}
      <div className="mb-4 p-4 bg-gray-50 rounded">
        <div className="flex justify-between items-start mb-2">
          <div>
            <p className="text-sm font-medium text-gray-700">Grid Carbon Intensity</p>
            <p className="text-xs text-gray-500 mt-1">{regionalInfo}</p>
          </div>
          <span className={`px-2 py-1 rounded text-xs font-semibold ${carbonRating.color} bg-opacity-10 bg-current`}>
            {carbonRating.label}
          </span>
        </div>
        <p className="text-2xl font-bold text-gray-900">
          {formatNumber(metrics.carbon_intensity_g_per_kwh, 0)} g/kWh
        </p>
      </div>

      {/* Contextual comparisons */}
      {metrics.equivalent_car_miles > 0 && (
        <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
          <p className="text-sm font-medium text-gray-700 mb-2">Environmental Context</p>
          <div className="flex items-center">
            <span className="text-2xl mr-2">🚗</span>
            <div>
              <p className="text-sm text-gray-700">
                Equivalent to driving{' '}
                <span className="font-semibold text-yellow-700">
                  {formatNumber(metrics.equivalent_car_miles, 3)} miles
                </span>
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Based on average car emissions of 404g CO₂/mile (EPA 2023)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Settings info footer */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          Cost and carbon estimates are based on configured electricity price and regional carbon intensity.
          Actual values may vary based on time of day, energy mix, and location.
        </p>
      </div>

      {/* Edit settings hint */}
      <div className="mt-2">
        <p className="text-xs text-gray-400 italic">
          To adjust electricity price or carbon intensity, configure them when starting a profiling run.
        </p>
      </div>
    </div>
  );
};

export default CostCarbonCard;
