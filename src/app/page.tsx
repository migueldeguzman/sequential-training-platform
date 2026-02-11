"use client";

import { useState, useEffect, useCallback, lazy, Suspense } from "react";
import type { Dataset, TextDataset, TrainingConfig } from "@/types";
import { trainingApi, datasetApi, textDatasetsApi } from "@/lib/api";

import DatasetPanel from "@/components/DatasetPanel";
import TrainingConfigPanel from "@/components/TrainingConfigPanel";
import TrainingMonitor from "@/components/TrainingMonitor";
import ModelManagement from "@/components/ModelManagement";
import SettingsPanel from "@/components/SettingsPanel";
import TrainingHistory from "@/components/TrainingHistory";
import TestingPanel from "@/components/TestingPanel";
import ErrorBoundary from "@/components/ErrorBoundary";

// Lazy load EnergyProfilerPanel for better performance
const EnergyProfilerPanel = lazy(() => import("@/components/profiling/EnergyProfilerPanel"));

type TabType = "datasets" | "training" | "history" | "models" | "testing" | "profiling" | "settings";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("training");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [textDatasets, setTextDatasets] = useState<TextDataset[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  // Fetch datasets at page level so they're always available
  const fetchDatasets = useCallback(async () => {
    const response = await datasetApi.list();
    if (response.success && response.data) {
      setDatasets(response.data);
    }
  }, []);

  const fetchTextDatasets = useCallback(async () => {
    const response = await textDatasetsApi.list();
    if (response.success && response.data) {
      setTextDatasets(response.data);
    }
  }, []);

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets();
    fetchTextDatasets();
  }, [fetchDatasets, fetchTextDatasets]);

  useEffect(() => {
    const checkTrainingStatus = async () => {
      const response = await trainingApi.status();
      if (response.success && response.data) {
        setIsTraining(response.data.isRunning);
      }
    };
    checkTrainingStatus();
    const interval = setInterval(checkTrainingStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = useCallback(async (config: TrainingConfig) => {
    const response = await trainingApi.start(config);
    if (response.success) {
      setIsTraining(true);
      setActiveTab("training");
    } else {
      alert(response.error || "Failed to start training");
    }
  }, []);

  const handleTrainingStop = useCallback(() => {
    setIsTraining(false);
  }, []);

  const tabs: { id: TabType; label: string; icon?: string }[] = [
    { id: "datasets", label: "Datasets" },
    { id: "training", label: "Training" },
    { id: "history", label: "History" },
    { id: "models", label: "Models" },
    { id: "testing", label: "Testing" },
    { id: "profiling", label: "Energy Profiler", icon: "⚡" },
    { id: "settings", label: "Settings" },
  ];

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-xl font-bold">Sequential Training and Testing Platform</h1>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 font-medium capitalize transition-colors ${
                  activeTab === tab.id
                    ? "text-blue-400 border-b-2 border-blue-400"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {tab.icon && <span className="mr-2">{tab.icon}</span>}
                {tab.label}
                {tab.id === "training" && isTraining && (
                  <span className="ml-2 w-2 h-2 bg-green-400 rounded-full inline-block animate-pulse"></span>
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <ErrorBoundary>
          {activeTab === "datasets" && (
            <DatasetPanel onDatasetsChange={setDatasets} onRefresh={fetchDatasets} />
          )}

          {/* Training section - always mounted to preserve logs, hidden when not active */}
          <div className={activeTab === "training" ? "" : "hidden"}>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <TrainingConfigPanel
                datasets={datasets}
                textDatasets={textDatasets}
                onStartTraining={handleStartTraining}
                disabled={isTraining}
              />
              <div className="lg:col-span-2">
                <TrainingMonitor
                  isTraining={isTraining}
                  onTrainingStop={handleTrainingStop}
                />
              </div>
            </div>
          </div>

          {activeTab === "history" && <TrainingHistory />}

          {activeTab === "models" && <ModelManagement />}

          {activeTab === "testing" && <TestingPanel />}

          {activeTab === "profiling" && (
            <Suspense
              fallback={
                <div className="flex items-center justify-center py-12">
                  <div className="text-lg text-gray-400">Loading Energy Profiler...</div>
                </div>
              }
            >
              <EnergyProfilerPanel />
            </Suspense>
          )}

          {activeTab === "settings" && <SettingsPanel />}
        </ErrorBoundary>
      </main>
    </div>
  );
}
