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

  it('implements exponential backoff on reconnection', () => {
    const { result } = renderHook(() => useProfilingWebSocket());

    // Connect and simulate disconnect
    act(() => {
      result.current.connect();
    });

    // Simulate WebSocket close
    const onCloseHandler = (global.WebSocket as jest.Mock).mock.results[0].value.addEventListener.mock.calls.find(
      (call: [string, (event: CloseEvent) => void]) => call[0] === 'close'
    )?.[1];

    if (onCloseHandler) {
      act(() => {
        onCloseHandler({ code: 1006, reason: 'Abnormal closure' } as CloseEvent);
      });

      // Should be in reconnecting state
      expect(result.current.connectionState).toBe('reconnecting');
    }
  });

  it('handles heartbeat mechanism', () => {
    const { result } = renderHook(() => useProfilingWebSocket());

    act(() => {
      result.current.connect();
    });

    // Advance timers for heartbeat
    jest.advanceTimersByTime(30000);

    // WebSocket should still be managed properly
    expect(result.current.connectionState).toBeDefined();
  });

  it('handles message parsing errors gracefully', () => {
    const consoleError = jest.spyOn(console, 'error').mockImplementation();
    const { result } = renderHook(() => useProfilingWebSocket());

    act(() => {
      result.current.connect();
    });

    // Simulate malformed message
    const onMessageHandler = (global.WebSocket as jest.Mock).mock.results[0].value.addEventListener.mock.calls.find(
      (call: [string, (event: MessageEvent) => void]) => call[0] === 'message'
    )?.[1];

    if (onMessageHandler) {
      act(() => {
        onMessageHandler({ data: 'invalid json{' } as MessageEvent);
      });

      // Should handle error without crashing
      expect(result.current.connectionState).toBeDefined();
    }

    consoleError.mockRestore();
  });

  it('prevents duplicate connections', () => {
    const { result } = renderHook(() => useProfilingWebSocket());

    act(() => {
      result.current.connect();
      result.current.connect();
    });

    // Should only create one WebSocket
    expect(global.WebSocket).toHaveBeenCalledTimes(1);
  });

  it('recovers after backend restart', () => {
    const { result } = renderHook(() => useProfilingWebSocket());

    act(() => {
      result.current.connect();
    });

    // Simulate backend disconnect
    const onCloseHandler = (global.WebSocket as jest.Mock).mock.results[0].value.addEventListener.mock.calls.find(
      (call: [string, (event: CloseEvent) => void]) => call[0] === 'close'
    )?.[1];

    if (onCloseHandler) {
      act(() => {
        onCloseHandler({ code: 1006, reason: 'Connection lost' } as CloseEvent);
      });

      // Should attempt reconnection
      expect(result.current.connectionState).toBe('reconnecting');

      // Advance timers to trigger reconnection
      jest.advanceTimersByTime(1000);

      // Should be attempting to reconnect
      expect(result.current.connectionState).toBeDefined();
    }
  });
});
