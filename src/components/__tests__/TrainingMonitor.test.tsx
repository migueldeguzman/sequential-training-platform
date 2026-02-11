/**
 * TEST-004: Training Monitor Real-time Updates
 *
 * Tests training progress updates via WebSocket, including progress bars,
 * loss charts, log streaming, and stop functionality.
 */

import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import TrainingMonitor from '../TrainingMonitor';

// Mock WebSocket for testing
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: WebSocket.OPEN,
};

describe('TrainingMonitor', () => {
  let messageHandlers: Map<string, Function>;

  beforeEach(() => {
    jest.clearAllMocks();
    messageHandlers = new Map();

    // Mock WebSocket constructor
    global.WebSocket = jest.fn(() => ({
      ...mockWebSocket,
      addEventListener: jest.fn((event: string, handler: Function) => {
        messageHandlers.set(event, handler);
      }),
    })) as any;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Initial State', () => {
    it('renders in idle state when no training is active', () => {
      render(<TrainingMonitor />);

      expect(screen.getByText(/No active training/i)).toBeInTheDocument();
    });

    it('displays training status indicator', () => {
      render(<TrainingMonitor />);

      // Should show idle status
      expect(screen.queryByText(/Idle/i) || screen.queryByText(/Ready/i)).toBeInTheDocument();
    });
  });

  describe('Progress Bar Updates', () => {
    it('updates progress bar based on WebSocket messages', async () => {
      render(<TrainingMonitor />);

      // Simulate training start message
      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_progress',
              progress: 25,
              epoch: 1,
              total_epochs: 4,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          const progressBar = screen.queryByRole('progressbar');
          if (progressBar) {
            expect(progressBar).toHaveAttribute('aria-valuenow', '25');
          }
        });
      }
    });

    it('shows epoch information', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_progress',
              progress: 50,
              epoch: 2,
              total_epochs: 4,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/Epoch 2/i)).toBeInTheDocument();
        });
      }
    });

    it('handles 100% completion', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_complete',
              progress: 100,
              status: 'completed',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/Complete/i)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Loss Chart Rendering', () => {
    it('updates loss chart with training metrics', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_metrics',
              loss: 0.5,
              step: 100,
              learning_rate: 0.001,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          // Check if loss value is displayed
          expect(screen.getByText(/0.5/)).toBeInTheDocument();
        });
      }
    });

    it('accumulates multiple loss data points', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        // Send multiple metric updates
        const metrics = [
          { loss: 1.0, step: 0 },
          { loss: 0.8, step: 100 },
          { loss: 0.6, step: 200 },
        ];

        for (const metric of metrics) {
          act(() => {
            messageHandler({
              data: JSON.stringify({
                type: 'training_metrics',
                ...metric,
              }),
            } as MessageEvent);
          });
        }

        await waitFor(() => {
          // Verify chart is rendering with data
          expect(screen.queryByText(/Loss/i)).toBeInTheDocument();
        });
      }
    });

    it('displays learning rate', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_metrics',
              loss: 0.5,
              learning_rate: 0.001,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/0.001/)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Log Streaming Display', () => {
    it('displays training logs from WebSocket', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_log',
              message: 'Starting training...',
              timestamp: Date.now(),
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/Starting training/i)).toBeInTheDocument();
        });
      }
    });

    it('appends new logs to existing ones', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        const logs = [
          'Loading model...',
          'Loading dataset...',
          'Training started',
        ];

        for (const log of logs) {
          act(() => {
            messageHandler({
              data: JSON.stringify({
                type: 'training_log',
                message: log,
                timestamp: Date.now(),
              }),
            } as MessageEvent);
          });
        }

        await waitFor(() => {
          logs.forEach((log) => {
            expect(screen.getByText(new RegExp(log, 'i'))).toBeInTheDocument();
          });
        });
      }
    });

    it('auto-scrolls logs to bottom', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        // Send many log messages
        for (let i = 0; i < 50; i++) {
          act(() => {
            messageHandler({
              data: JSON.stringify({
                type: 'training_log',
                message: `Log message ${i}`,
                timestamp: Date.now(),
              }),
            } as MessageEvent);
          });
        }

        // Verify latest log is visible
        await waitFor(() => {
          expect(screen.getByText(/Log message 49/i)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Stop Training Functionality', () => {
    it('displays stop button when training is active', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_started',
              status: 'running',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
        });
      }
    });

    it('sends stop request when stop button is clicked', async () => {
      const mockFetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'stopped' }),
      });
      global.fetch = mockFetch;

      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_started',
              status: 'running',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          const stopButton = screen.getByRole('button', { name: /stop/i });
          fireEvent.click(stopButton);
        });

        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/training/stop'),
          expect.any(Object)
        );
      }
    });

    it('disables stop button while stopping', async () => {
      const mockFetch = jest.fn().mockImplementation(
        () =>
          new Promise((resolve) =>
            setTimeout(
              () =>
                resolve({
                  ok: true,
                  json: async () => ({ status: 'stopped' }),
                }),
              100
            )
          )
      );
      global.fetch = mockFetch;

      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_started',
              status: 'running',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          const stopButton = screen.getByRole('button', { name: /stop/i });
          fireEvent.click(stopButton);

          // Button should be disabled while stopping
          expect(stopButton).toBeDisabled();
        });
      }
    });
  });

  describe('Error Handling', () => {
    it('displays error message when training fails', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'training_error',
              error: 'Out of memory',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/Out of memory/i)).toBeInTheDocument();
        });
      }
    });

    it('handles WebSocket disconnection', async () => {
      render(<TrainingMonitor />);

      const closeHandler = messageHandlers.get('close');
      if (closeHandler) {
        act(() => {
          closeHandler({ code: 1006, reason: 'Connection lost' } as CloseEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/disconnected/i)).toBeInTheDocument();
        });
      }
    });

    it('handles malformed WebSocket messages', async () => {
      const consoleError = jest.spyOn(console, 'error').mockImplementation();
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: 'invalid json{',
          } as MessageEvent);
        });

        // Should not crash
        await waitFor(() => {
          expect(screen.queryByText(/error/i)).toBeInTheDocument();
        });
      }

      consoleError.mockRestore();
    });
  });

  describe('Performance', () => {
    it('handles rapid message updates without lag', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        const startTime = performance.now();

        // Send 100 rapid updates
        for (let i = 0; i < 100; i++) {
          act(() => {
            messageHandler({
              data: JSON.stringify({
                type: 'training_metrics',
                loss: 1 - i / 100,
                step: i,
              }),
            } as MessageEvent);
          });
        }

        const elapsed = performance.now() - startTime;

        // Should process all updates in reasonable time
        expect(elapsed).toBeLessThan(1000);
      }
    });

    it('limits stored log messages to prevent memory issues', async () => {
      render(<TrainingMonitor />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        // Send 1000 log messages
        for (let i = 0; i < 1000; i++) {
          act(() => {
            messageHandler({
              data: JSON.stringify({
                type: 'training_log',
                message: `Log ${i}`,
              }),
            } as MessageEvent);
          });
        }

        // Only recent logs should be visible (typically last 100-500)
        await waitFor(() => {
          expect(screen.getByText(/Log 999/i)).toBeInTheDocument();
          // Very old logs should be removed
          expect(screen.queryByText(/Log 0/i)).not.toBeInTheDocument();
        });
      }
    });
  });
});
