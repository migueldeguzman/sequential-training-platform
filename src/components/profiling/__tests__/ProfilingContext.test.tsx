import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { ProfilingProvider, useProfilingContext } from '../ProfilingContext';
import type { ProfiledGenerateRequest } from '@/types';

// Mock the API
const mockProfiledGenerate = jest.fn();
jest.mock('@/lib/api', () => ({
  api: {
    profiledGenerate: (request: ProfiledGenerateRequest) => mockProfiledGenerate(request),
  },
}));

// Mock the WebSocket
const mockConnect = jest.fn();
const mockDisconnect = jest.fn();
jest.mock('@/lib/profilingWebsocket', () => ({
  useProfilingWebSocket: () => ({
    connect: mockConnect,
    disconnect: mockDisconnect,
    connectionState: 'disconnected' as const,
    error: null,
  }),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ProfilingProvider>{children}</ProfilingProvider>
);

describe('ProfilingContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockProfiledGenerate.mockResolvedValue({ run_id: 'test-run-123' });
  });

  it('provides initial state', () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    expect(result.current.currentRun).toBeNull();
    expect(result.current.isRunning).toBe(false);
    expect(result.current.isProfiling).toBe(false);
    expect(result.current.powerSamples).toEqual([]);
    expect(result.current.tokens).toEqual([]);
    expect(result.current.sections).toEqual([]);
    expect(result.current.currentSection).toBeNull();
    expect(result.current.completedRunId).toBeNull();
  });

  it('provides action functions', () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    expect(typeof result.current.startProfiling).toBe('function');
    expect(typeof result.current.stopProfiling).toBe('function');
    expect(typeof result.current.clearState).toBe('function');
    expect(typeof result.current.connectWebSocket).toBe('function');
    expect(typeof result.current.disconnectWebSocket).toBe('function');
  });

  it('connects WebSocket when connectWebSocket is called', () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    act(() => {
      result.current.connectWebSocket();
    });

    expect(mockConnect).toHaveBeenCalled();
  });

  it('disconnects WebSocket when disconnectWebSocket is called', () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    act(() => {
      result.current.disconnectWebSocket();
    });

    expect(mockDisconnect).toHaveBeenCalled();
  });

  it('starts profiling and updates state', async () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    const request: ProfiledGenerateRequest = {
      prompt: 'Test prompt',
      profiling_depth: 'module',
      temperature: 0.7,
      max_length: 100,
    };

    await act(async () => {
      await result.current.startProfiling(request);
    });

    await waitFor(() => {
      expect(result.current.isProfiling).toBe(true);
    });

    expect(mockConnect).toHaveBeenCalled();
    expect(mockProfiledGenerate).toHaveBeenCalledWith(request);
  });

  it('stops profiling and updates state', async () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    const request: ProfiledGenerateRequest = {
      prompt: 'Test prompt',
      profiling_depth: 'module',
      temperature: 0.7,
      max_length: 100,
    };

    await act(async () => {
      await result.current.startProfiling(request);
    });

    act(() => {
      result.current.stopProfiling();
    });

    expect(result.current.isProfiling).toBe(false);
    expect(mockDisconnect).toHaveBeenCalled();
  });

  it('clears state when clearState is called', async () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    const request: ProfiledGenerateRequest = {
      prompt: 'Test prompt',
      profiling_depth: 'module',
      temperature: 0.7,
      max_length: 100,
    };

    await act(async () => {
      await result.current.startProfiling(request);
    });

    act(() => {
      result.current.clearState();
    });

    expect(result.current.currentRun).toBeNull();
    expect(result.current.powerSamples).toEqual([]);
    expect(result.current.tokens).toEqual([]);
    expect(result.current.sections).toEqual([]);
    expect(result.current.completedRunId).toBeNull();
  });

  it('throws error when used outside provider', () => {
    // Suppress console error for this test
    const consoleError = jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useProfilingContext());
    }).toThrow('useProfilingContext must be used within a ProfilingProvider');

    consoleError.mockRestore();
  });

  it('handles profiling errors gracefully', async () => {
    mockProfiledGenerate.mockRejectedValueOnce(new Error('API Error'));

    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    const request: ProfiledGenerateRequest = {
      prompt: 'Test prompt',
      profiling_depth: 'module',
      temperature: 0.7,
      max_length: 100,
    };

    await act(async () => {
      try {
        await result.current.startProfiling(request);
      } catch {
        // Expected to throw
      }
    });

    // Should remain in non-profiling state after error
    expect(result.current.isProfiling).toBe(false);
  });

  it('stores connectionState from WebSocket', () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    expect(result.current.connectionState).toBe('disconnected');
  });

  it('tracks wsError state', () => {
    const { result } = renderHook(() => useProfilingContext(), { wrapper });

    expect(result.current.wsError).toBeNull();
  });
});
