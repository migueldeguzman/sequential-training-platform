'use client';

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import type {
  ProfilingRun,
  PowerSample,
  TokenMetrics,
  PipelineSection,
  ProfiledGenerateRequest,
  InferenceCompleteMessage,
} from '@/types';
import { useProfilingWebSocket, type ConnectionState } from '@/lib/profilingWebsocket';
import { api } from '@/lib/api';

// Current section info
export interface CurrentSection {
  phase: string;
  section_name: string;
  start_time: number;
}

// Profiling state
export interface ProfilingState {
  // Current profiling run
  currentRun: ProfilingRun | null;
  isRunning: boolean;
  isProfiling: boolean;

  // Real-time data streams
  powerSamples: PowerSample[];
  tokens: TokenMetrics[];
  sections: PipelineSection[];

  // Current section being profiled
  currentSection: CurrentSection | null;

  // Model loading state
  isLoadingModel: boolean;
  modelLoadingMessage: string | null;

  // WebSocket connection state
  connectionState: ConnectionState;
  wsError: Error | null;

  // Run ID for database reference after completion
  completedRunId: string | null;
}

// Profiling actions
export interface ProfilingActions {
  startProfiling: (request: ProfiledGenerateRequest) => Promise<void>;
  stopProfiling: () => void;
  clearState: () => void;
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;
}

// Combined context value
export interface ProfilingContextValue extends ProfilingState, ProfilingActions {}

// Create context
const ProfilingContext = createContext<ProfilingContextValue | undefined>(undefined);

// Initial state
const initialState: ProfilingState = {
  currentRun: null,
  isRunning: false,
  isProfiling: false,
  powerSamples: [],
  tokens: [],
  sections: [],
  currentSection: null,
  isLoadingModel: false,
  modelLoadingMessage: null,
  connectionState: 'disconnected',
  wsError: null,
  completedRunId: null,
};

// Provider component
export function ProfilingProvider({ children }: { children: React.ReactNode }) {
  // State
  const [state, setState] = useState<ProfilingState>(initialState);
  const profilingRequestRef = useRef<ProfiledGenerateRequest | null>(null);

  // WebSocket integration
  const {
    connectionState,
    error: wsError,
    connect: wsConnect,
    disconnect: wsDisconnect,
    subscribe,
  } = useProfilingWebSocket({
    autoConnect: false,
  });

  // Update connection state in our state
  useEffect(() => {
    setState((prev) => ({
      ...prev,
      connectionState,
      wsError,
    }));
  }, [connectionState, wsError]);

  // Subscribe to WebSocket events
  useEffect(() => {
    // Power sample handler
    subscribe('power_sample', (message) => {
      setState((prev) => ({
        ...prev,
        powerSamples: [...prev.powerSamples, message.data],
      }));
    });

    // Section start handler
    subscribe('section_start', (message) => {
      setState((prev) => ({
        ...prev,
        currentSection: {
          phase: message.data.phase,
          section_name: message.data.section_name,
          start_time: message.timestamp,
        },
      }));
    });

    // Section end handler
    subscribe('section_end', (message) => {
      setState((prev) => {
        const section: PipelineSection = {
          id: prev.sections.length,
          run_id: prev.currentRun?.id || '',
          phase: message.data.phase,
          section_name: message.data.section_name,
          start_time: message.timestamp - message.data.duration_ms,
          end_time: message.timestamp,
          duration_ms: message.data.duration_ms,
          energy_mj: message.data.energy_mj,
        };

        return {
          ...prev,
          sections: [...prev.sections, section],
          currentSection: null,
        };
      });
    });

    // Token complete handler
    subscribe('token_complete', (message) => {
      setState((prev) => {
        const token: TokenMetrics = {
          id: prev.tokens.length,
          run_id: prev.currentRun?.id || '',
          token_position: message.token_position,
          token_text: message.token_text,
          phase: message.token_position === 0 ? 'prefill' : 'decode',
          start_time: message.timestamp - message.duration_ms,
          end_time: message.timestamp,
          duration_ms: message.duration_ms,
          energy_mj: message.energy_mj,
          power_snapshot_mw: message.power_snapshot_mw,
          layers: message.layer_metrics || [],
        };

        return {
          ...prev,
          tokens: [...prev.tokens, token],
        };
      });
    });

    // Model loading handler
    subscribe('model_loading', (message) => {
      setState((prev) => ({
        ...prev,
        isLoadingModel: message.data.status === 'loading',
        modelLoadingMessage: message.data.message,
      }));
    });

    // Inference complete handler
    subscribe('inference_complete', (message: InferenceCompleteMessage) => {
      setState((prev) => ({
        ...prev,
        isRunning: false,
        isProfiling: false,
        completedRunId: message.run_id,
        currentSection: null,
      }));
    });
  }, [subscribe]);

  // Start profiling action
  const startProfiling = useCallback(async (request: ProfiledGenerateRequest) => {
    try {
      // Clear previous state
      setState(initialState);

      // Store request for reference
      profilingRequestRef.current = request;

      // Connect WebSocket first
      wsConnect();

      // Wait a moment for WebSocket to connect
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Update state to running
      setState((prev) => ({
        ...prev,
        isRunning: true,
        isProfiling: true,
        currentRun: {
          id: '', // Will be set by inference_complete message
          timestamp: new Date().toISOString(),
          model_name: request.model_path || 'default',
          model_size_mb: 0,
          prompt: request.prompt,
          response: '',
          total_duration_ms: 0,
          total_energy_mj: 0,
          input_tokens: 0,
          output_tokens: 0,
          profiling_depth: request.profiling_depth || 'module',
          tags: request.tags,
          experiment_name: request.experiment_name,
        } as ProfilingRun,
      }));

      // Start profiled inference via API
      // Note: The API will stream results via WebSocket
      await api.profiledGenerate(request);

    } catch (error) {
      console.error('Failed to start profiling:', error);
      setState((prev) => ({
        ...prev,
        isRunning: false,
        isProfiling: false,
        wsError: error instanceof Error ? error : new Error('Failed to start profiling'),
      }));
    }
  }, [wsConnect]);

  // Stop profiling action
  const stopProfiling = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isRunning: false,
      isProfiling: false,
    }));

    // Note: We don't disconnect WebSocket immediately
    // in case there are pending messages
    setTimeout(() => {
      wsDisconnect();
    }, 1000);
  }, [wsDisconnect]);

  // Clear state action
  const clearState = useCallback(() => {
    setState(initialState);
    profilingRequestRef.current = null;
  }, []);

  // Connect WebSocket action
  const connectWebSocket = useCallback(() => {
    wsConnect();
  }, [wsConnect]);

  // Disconnect WebSocket action
  const disconnectWebSocket = useCallback(() => {
    wsDisconnect();
  }, [wsDisconnect]);

  // Context value
  const value: ProfilingContextValue = {
    ...state,
    startProfiling,
    stopProfiling,
    clearState,
    connectWebSocket,
    disconnectWebSocket,
  };

  return (
    <ProfilingContext.Provider value={value}>
      {children}
    </ProfilingContext.Provider>
  );
}

// Hook to use profiling context
export function useProfilingContext() {
  const context = useContext(ProfilingContext);
  if (context === undefined) {
    throw new Error('useProfilingContext must be used within a ProfilingProvider');
  }
  return context;
}
