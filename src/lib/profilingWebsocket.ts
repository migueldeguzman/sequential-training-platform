import { useEffect, useRef, useState, useCallback } from 'react';
import { getWebSocketUrl } from './config';
import type {
  ProfilingMessage,
  ProfilingMessageType,
  PowerSampleMessage,
  SectionStartMessage,
  SectionEndMessage,
  TokenCompleteMessage,
  LayerMetricsMessage,
  ComponentMetricsMessage,
  InferenceCompleteMessage,
  ModelLoadingMessage,
} from '@/types';

// WebSocket connection states
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'disconnecting';

// Event handler types for each message type
export type ProfilingEventHandlers = {
  power_sample?: (message: PowerSampleMessage) => void;
  section_start?: (message: SectionStartMessage) => void;
  section_end?: (message: SectionEndMessage) => void;
  token_complete?: (message: TokenCompleteMessage) => void;
  layer_metrics?: (message: LayerMetricsMessage) => void;
  component_metrics?: (message: ComponentMetricsMessage) => void;
  inference_complete?: (message: InferenceCompleteMessage) => void;
  model_loading?: (message: ModelLoadingMessage) => void;
  error?: (error: Error) => void;
  connectionStateChange?: (state: ConnectionState) => void;
};

// WebSocket manager class
export class ProfilingWebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay = 30000; // Max 30 seconds
  private reconnectTimer: NodeJS.Timeout | null = null;
  private handlers: ProfilingEventHandlers = {};
  private connectionState: ConnectionState = 'disconnected';
  private shouldReconnect = true;

  constructor(baseUrl?: string) {
    // Use config if no baseUrl provided, which will check environment variables
    if (!baseUrl) {
      this.url = getWebSocketUrl('/ws/profiling');
    } else {
      // Legacy support: if baseUrl is provided, use it directly
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      this.url = `${protocol}//${baseUrl}/ws/profiling`;
    }
  }

  // Set event handlers
  on<T extends ProfilingMessageType>(
    type: T,
    handler: (message: Extract<ProfilingMessage, { type: T }>) => void
  ): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.handlers[type] = handler as any;
  }

  // Set error handler
  onError(handler: (error: Error) => void): void {
    this.handlers.error = handler;
  }

  // Set connection state change handler
  onConnectionStateChange(handler: (state: ConnectionState) => void): void {
    this.handlers.connectionStateChange = handler;
  }

  // Update connection state and notify
  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.handlers.connectionStateChange?.(state);
  }

  // Connect to WebSocket
  connect(): void {
    // Don't start a new connection if already connecting or connected
    if (this.connectionState === 'connecting' || this.connectionState === 'connected') {
      return;
    }

    // Don't start a new connection if disconnecting
    if (this.connectionState === 'disconnecting') {
      return;
    }

    // Cancel any pending reconnection attempt
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.setConnectionState('connecting');
    this.shouldReconnect = true;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.setConnectionState('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const message: ProfilingMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          const err = error instanceof Error ? error : new Error('Failed to parse WebSocket message');
          this.handlers.error?.(err);
        }
      };

      this.ws.onerror = () => {
        const error = new Error('WebSocket error occurred');
        this.handlers.error?.(error);
      };

      this.ws.onclose = () => {
        this.setConnectionState('disconnected');
        if (this.shouldReconnect) {
          this.scheduleReconnect();
        }
      };
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Failed to create WebSocket connection');
      this.handlers.error?.(err);
      this.setConnectionState('disconnected');
    }
  }

  // Handle incoming messages
  private handleMessage(message: ProfilingMessage): void {
    const handler = this.handlers[message.type];
    if (handler) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (handler as any)(message);
    }
  }

  // Schedule reconnection with exponential backoff
  private scheduleReconnect(): void {
    // Don't reconnect if already connecting or if a reconnection is scheduled
    if (this.connectionState === 'connecting' || this.connectionState === 'connected') {
      return;
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.setConnectionState('disconnected');
      this.handlers.error?.(new Error('Max reconnection attempts reached'));
      return;
    }

    // Don't schedule multiple reconnection attempts
    if (this.reconnectTimer) {
      return;
    }

    this.setConnectionState('reconnecting');
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, this.reconnectDelay);

    // Exponential backoff
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }

  // Disconnect from WebSocket
  disconnect(): void {
    this.shouldReconnect = false;

    // Cancel any pending reconnection timer
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    // If already disconnected, nothing to do
    if (this.connectionState === 'disconnected') {
      return;
    }

    // Set disconnecting state to prevent new connections
    this.setConnectionState('disconnecting');

    if (this.ws) {
      // Close the WebSocket connection
      this.ws.close();
      this.ws = null;
    }

    this.setConnectionState('disconnected');
  }

  // Get current connection state
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  // Check if connected
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// React hook for using profiling WebSocket
export function useProfilingWebSocket(options?: {
  autoConnect?: boolean;
  baseUrl?: string;
  handlers?: ProfilingEventHandlers;
}) {
  const { autoConnect = false, baseUrl, handlers } = options || {};
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [error, setError] = useState<Error | null>(null);
  const wsManagerRef = useRef<ProfilingWebSocketManager | null>(null);

  // Initialize WebSocket manager
  useEffect(() => {
    const manager = new ProfilingWebSocketManager(baseUrl);
    wsManagerRef.current = manager;

    // Set up connection state handler
    manager.onConnectionStateChange((state) => {
      setConnectionState(state);
    });

    // Set up error handler
    manager.onError((err) => {
      setError(err);
    });

    // Set up message handlers if provided
    if (handlers) {
      Object.entries(handlers).forEach(([type, handler]) => {
        if (type === 'error' || type === 'connectionStateChange') {
          return;
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        manager.on(type as ProfilingMessageType, handler as any);
      });
    }

    // Auto-connect if enabled
    if (autoConnect) {
      manager.connect();
    }

    // Cleanup on unmount
    return () => {
      manager.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, autoConnect]);

  // Update handlers when they change
  useEffect(() => {
    if (handlers && wsManagerRef.current) {
      Object.entries(handlers).forEach(([type, handler]) => {
        if (type === 'error') {
          wsManagerRef.current?.onError(handler as (error: Error) => void);
        } else if (type === 'connectionStateChange') {
          wsManagerRef.current?.onConnectionStateChange(handler as (state: ConnectionState) => void);
        } else {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          wsManagerRef.current?.on(type as ProfilingMessageType, handler as any);
        }
      });
    }
  }, [handlers]);

  // Connect method
  const connect = useCallback(() => {
    wsManagerRef.current?.connect();
    setError(null);
  }, []);

  // Disconnect method
  const disconnect = useCallback(() => {
    wsManagerRef.current?.disconnect();
  }, []);

  // Subscribe to specific message type
  const subscribe = useCallback(
    <T extends ProfilingMessageType>(
      type: T,
      handler: (message: Extract<ProfilingMessage, { type: T }>) => void
    ) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      wsManagerRef.current?.on(type, handler as any);
    },
    []
  );

  return {
    connectionState,
    error,
    connect,
    disconnect,
    subscribe,
    isConnected: connectionState === 'connected',
    manager: wsManagerRef.current,
  };
}
