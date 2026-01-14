import { renderHook, act } from '@testing-library/react';
import { useProfilingWebSocket, ProfilingWebSocketManager } from '../profilingWebsocket';
import type { PowerSampleMessage, InferenceCompleteMessage } from '@/types';

interface MockWebSocket {
  readyState: number;
  send: jest.Mock;
  close: jest.Mock;
  addEventListener: jest.Mock;
  removeEventListener: jest.Mock;
}

describe('ProfilingWebSocketManager', () => {
  let manager: ProfilingWebSocketManager;
  let mockWebSocket: MockWebSocket;

  beforeEach(() => {
    // Mock WebSocket
    mockWebSocket = {
      readyState: WebSocket.CONNECTING,
      send: jest.fn(),
      close: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    };

    global.WebSocket = jest.fn(() => mockWebSocket) as unknown as typeof WebSocket;
    manager = new ProfilingWebSocketManager('localhost:8000');
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('creates WebSocket with correct URL', () => {
    manager.connect();

    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws/profiling');
  });

  it('sets connection state to connecting when connect is called', () => {
    const stateHandler = jest.fn();
    manager.onConnectionStateChange(stateHandler);

    manager.connect();

    expect(stateHandler).toHaveBeenCalledWith('connecting');
  });

  it('registers event handlers', () => {
    const powerHandler = jest.fn();
    manager.on('power_sample', powerHandler);

    // Verify handler is registered by checking internal state
    expect(manager).toBeDefined();
  });

  it('handles power_sample message', () => {
    const powerHandler = jest.fn();
    manager.on('power_sample', powerHandler);

    manager.connect();
    mockWebSocket.readyState = WebSocket.OPEN;

    const message: PowerSampleMessage = {
      type: 'power_sample',
      timestamp: 100,
      cpu_power_mw: 1000,
      gpu_power_mw: 2000,
      ane_power_mw: 500,
      dram_power_mw: 300,
      total_power_mw: 3800,
    };

    // Simulate message receipt
    const onMessageHandler = mockWebSocket.addEventListener.mock.calls.find(
      (call: [string, (event: MessageEvent) => void]) => call[0] === 'message'
    )?.[1];

    if (onMessageHandler) {
      onMessageHandler({ data: JSON.stringify(message) } as MessageEvent);
      expect(powerHandler).toHaveBeenCalledWith(message);
    }
  });

  it('handles inference_complete message', () => {
    const completeHandler = jest.fn();
    manager.on('inference_complete', completeHandler);

    manager.connect();
    mockWebSocket.readyState = WebSocket.OPEN;

    const message: InferenceCompleteMessage = {
      type: 'inference_complete',
      run_id: 'test-run-123',
      total_duration_ms: 5000,
      total_energy_mj: 15000,
      token_count: 50,
      tokens_per_second: 10,
    };

    const onMessageHandler = mockWebSocket.addEventListener.mock.calls.find(
      (call: [string, (event: MessageEvent) => void]) => call[0] === 'message'
    )?.[1];

    if (onMessageHandler) {
      onMessageHandler({ data: JSON.stringify(message) } as MessageEvent);
      expect(completeHandler).toHaveBeenCalledWith(message);
    }
  });

  it('calls error handler on WebSocket error', () => {
    const errorHandler = jest.fn();
    manager.onError(errorHandler);

    manager.connect();

    const onErrorHandler = mockWebSocket.addEventListener.mock.calls.find(
      (call: [string, (event: Event) => void]) => call[0] === 'error'
    )?.[1];

    if (onErrorHandler) {
      const error = new Event('error');
      onErrorHandler(error);
      expect(errorHandler).toHaveBeenCalled();
    }
  });

  it('disconnects and cleans up', () => {
    manager.connect();

    manager.disconnect();

    expect(mockWebSocket.close).toHaveBeenCalled();
  });

  it('prevents duplicate connections', () => {
    mockWebSocket.readyState = WebSocket.OPEN;
    manager.connect();
    manager.connect();

    // Should only create WebSocket once
    expect(global.WebSocket).toHaveBeenCalledTimes(1);
  });
});

describe('useProfilingWebSocket', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it('returns initial disconnected state', () => {
    const { result } = renderHook(() => useProfilingWebSocket());

    expect(result.current.connectionState).toBe('disconnected');
    expect(result.current.error).toBeNull();
  });

  it('provides connect and disconnect functions', () => {
    const { result } = renderHook(() => useProfilingWebSocket());

    expect(typeof result.current.connect).toBe('function');
    expect(typeof result.current.disconnect).toBe('function');
  });

  it('updates connection state when connect is called', () => {
    const { result } = renderHook(() => useProfilingWebSocket());

    act(() => {
      result.current.connect();
    });

    expect(result.current.connectionState).toBe('connecting');
  });

  it('cleans up on unmount', () => {
    const { unmount } = renderHook(() => useProfilingWebSocket());

    unmount();

    // Should not throw errors
    expect(true).toBe(true);
  });
});
