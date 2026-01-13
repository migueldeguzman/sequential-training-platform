'use client';

import React, { useState } from 'react';
import { ProfilingRun } from '@/types';
import RunList from './RunList';
import RunDetail from './RunDetail';
import CompareView from './CompareView';

const HistoryBrowser: React.FC = () => {
  const [selectedRun, setSelectedRun] = useState<ProfilingRun | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [selectedRunsForCompare, setSelectedRunsForCompare] = useState<ProfilingRun[]>([]);

  const handleSelectRun = (run: ProfilingRun) => {
    if (compareMode) {
      // In compare mode, add to comparison list (max 4 runs)
      if (selectedRunsForCompare.find(r => r.id === run.id)) {
        // Remove if already selected
        setSelectedRunsForCompare(selectedRunsForCompare.filter(r => r.id !== run.id));
      } else if (selectedRunsForCompare.length < 4) {
        // Add if under limit
        setSelectedRunsForCompare([...selectedRunsForCompare, run]);
      }
    } else {
      // In normal mode, show details
      setSelectedRun(run);
    }
  };

  const handleToggleCompareMode = () => {
    if (compareMode) {
      // Exiting compare mode, clear selection
      setSelectedRunsForCompare([]);
    }
    setCompareMode(!compareMode);
  };

  const handleCloseDetail = () => {
    setSelectedRun(null);
  };

  const handleRemoveFromCompare = (runId: string) => {
    setSelectedRunsForCompare(selectedRunsForCompare.filter(r => r.id !== runId));
  };

  return (
    <div className="flex h-full gap-4">
      {/* Left Panel: Run List */}
      <div className="w-1/3 flex flex-col bg-gray-900 border border-gray-700 rounded-lg p-4">
        {/* Compare Mode Toggle */}
        <div className="mb-4">
          <button
            onClick={handleToggleCompareMode}
            className={`w-full px-4 py-2 rounded text-sm font-medium transition-colors ${
              compareMode
                ? 'bg-orange-600 hover:bg-orange-700 text-white'
                : 'bg-gray-700 hover:bg-gray-600 text-white'
            }`}
          >
            {compareMode ? 'Exit Compare Mode' : 'Compare Runs'}
          </button>
          {compareMode && (
            <p className="text-xs text-gray-400 mt-2 text-center">
              Select 2-4 runs to compare ({selectedRunsForCompare.length} selected)
            </p>
          )}
        </div>

        <RunList
          onSelectRun={handleSelectRun}
          selectedRunId={selectedRun?.id}
        />
      </div>

      {/* Right Panel: Run Detail or Compare View */}
      <div className="flex-1 flex flex-col bg-gray-900 border border-gray-700 rounded-lg">
        {compareMode ? (
          // Compare View
          selectedRunsForCompare.length === 0 ? (
            <div className="flex flex-col h-full items-center justify-center p-8 text-gray-400">
              <div className="text-center">
                <svg
                  className="w-16 h-16 mx-auto mb-4 text-gray-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                <h3 className="text-lg font-medium text-gray-300 mb-2">Compare Runs</h3>
                <p className="text-sm">Select 2-4 runs from the list to compare their metrics</p>
              </div>
            </div>
          ) : selectedRunsForCompare.length === 1 ? (
            <div className="flex flex-col h-full items-center justify-center p-8 text-gray-400">
              <div className="text-center">
                <p className="text-sm">
                  {selectedRunsForCompare.length} run selected. Select at least one more to compare.
                </p>
              </div>
            </div>
          ) : (
            <CompareView runs={selectedRunsForCompare} onRemoveRun={handleRemoveFromCompare} />
          )
        ) : selectedRun ? (
          // Run Detail View
          <RunDetail runId={selectedRun.id} onClose={handleCloseDetail} />
        ) : (
          // Empty state
          <div className="flex flex-col h-full items-center justify-center p-8 text-gray-400">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-gray-600"
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
            <h3 className="text-lg font-medium text-gray-300 mb-2">No Run Selected</h3>
            <p className="text-sm text-center">
              Select a profiling run from the list to view its details and analysis
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default HistoryBrowser;
