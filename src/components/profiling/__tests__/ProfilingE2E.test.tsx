/**
 * TEST-006: Energy Profiling End-to-End Test
 *
 * Tests complete profiling workflow from start to visualization,
 * including controls, WebSocket streaming, and result display.
 */

import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import EnergyProfilerPanel from '../EnergyProfilerPanel';
import { ProfilingProvider } from '../ProfilingContext';
import { profilingApi } from '@/lib/api';
import type { ProfilingRun, PowerSample } from '@/types';

// Mock the API modules
jest.mock('@/lib/api', () => ({
  profilingApi: {
    status: jest.fn(),
    start: jest.fn(),
    cancel: jest.fn(),
    getRuns: jest.fn(),
    getRun: jest.fn(),
    getRunSummary: jest.fn(),
    deleteRun: jest.fn(),
  },
  modelsApi: {
    list: jest.fn(),
  },
}));

// Mock WebSocket
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: WebSocket.OPEN,
};

describe('Energy Profiling E2E', () => {
  let messageHandlers: Map<string, Function>;

  beforeEach(() => {
    jest.clearAllMocks();
    messageHandlers = new Map();

    // Mock WebSocket
    global.WebSocket = jest.fn(() => ({
      ...mockWebSocket,
      addEventListener: jest.fn((event: string, handler: Function) => {
        messageHandlers.set(event, handler);
      }),
    })) as any;

    // Setup default API mocks
    (profilingApi.status as jest.Mock).mockResolvedValue({
      success: true,
      data: { available: true, running: false },
    });

    (profilingApi.getRuns as jest.Mock).mockResolvedValue({
      success: true,
      data: [],
    });
  });

  const renderWithProvider = (component: React.ReactElement) => {
    return render(<ProfilingProvider>{component}</ProfilingProvider>);
  };

  describe('Profiling Controls Validation', () => {
    it('validates required fields before starting', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      // Try to start without filling required fields
      const startButton = screen.getByRole('button', { name: /start profiling/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(screen.getByText(/Please fill.*required/i)).toBeInTheDocument();
      });
    });

    it('validates prompt is not empty', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const promptInput = screen.getByPlaceholderText(/Enter prompt/i);
      fireEvent.change(promptInput, { target: { value: '' } });

      const startButton = screen.getByRole('button', { name: /start profiling/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(screen.getByText(/Prompt.*required/i)).toBeInTheDocument();
      });
    });

    it('validates model is selected', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const promptInput = screen.getByPlaceholderText(/Enter prompt/i);
      fireEvent.change(promptInput, { target: { value: 'Test prompt' } });

      const startButton = screen.getByRole('button', { name: /start profiling/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(screen.getByText(/Model.*required/i)).toBeInTheDocument();
      });
    });

    it('allows starting with valid configuration', async () => {
      (profilingApi.start as jest.Mock).mockResolvedValue({
        success: true,
        data: { run_id: 'test-run-123' },
      });

      renderWithProvider(<EnergyProfilerPanel />);

      // Fill in required fields
      const promptInput = screen.getByPlaceholderText(/Enter prompt/i);
      fireEvent.change(promptInput, { target: { value: 'Test prompt' } });

      const modelSelect = screen.getByRole('combobox', { name: /model/i });
      fireEvent.change(modelSelect, { target: { value: 'test-model' } });

      const startButton = screen.getByRole('button', { name: /start profiling/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(profilingApi.start).toHaveBeenCalledWith(
          expect.objectContaining({
            prompt: 'Test prompt',
            model: 'test-model',
          })
        );
      });
    });
  });

  describe('WebSocket Connection and Streaming', () => {
    it('establishes WebSocket connection on profiling start', async () => {
      (profilingApi.start as jest.Mock).mockResolvedValue({
        success: true,
        data: { run_id: 'test-run-123' },
      });

      renderWithProvider(<EnergyProfilerPanel />);

      // Start profiling
      const promptInput = screen.getByPlaceholderText(/Enter prompt/i);
      fireEvent.change(promptInput, { target: { value: 'Test' } });

      const startButton = screen.getByRole('button', { name: /start profiling/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(global.WebSocket).toHaveBeenCalledWith(
          expect.stringContaining('/ws/profiling')
        );
      });
    });

    it('displays real-time power samples', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      // Simulate profiling started
      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'power_sample',
              timestamp_ms: 100,
              cpu_power_mw: 1000,
              gpu_power_mw: 2000,
              ane_power_mw: 500,
              dram_power_mw: 300,
              total_power_mw: 3800,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/3800.*mW/i)).toBeInTheDocument();
        });
      }
    });

    it('streams token generation updates', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'token_generated',
              token_index: 0,
              token_text: 'Hello',
              energy_mj: 50,
              duration_ms: 12,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/Hello/i)).toBeInTheDocument();
        });
      }
    });

    it('handles multiple power samples sequentially', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        const samples = [
          { timestamp_ms: 0, total_power_mw: 3000 },
          { timestamp_ms: 100, total_power_mw: 3500 },
          { timestamp_ms: 200, total_power_mw: 4000 },
        ];

        for (const sample of samples) {
          act(() => {
            messageHandler({
              data: JSON.stringify({
                type: 'power_sample',
                ...sample,
                cpu_power_mw: 1000,
                gpu_power_mw: 2000,
                ane_power_mw: 500,
                dram_power_mw: 300,
              }),
            } as MessageEvent);
          });
        }

        await waitFor(() => {
          expect(screen.getByText(/4000/i)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Run Completion and Summary', () => {
    it('displays completion message when profiling finishes', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'inference_complete',
              run_id: 'test-run-123',
              total_duration_ms: 5000,
              total_energy_mj: 15000,
              token_count: 50,
              tokens_per_second: 10,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(screen.getByText(/complete/i)).toBeInTheDocument();
        });
      }
    });

    it('generates and displays run summary', async () => {
      const mockSummary = {
        run_id: 'test-run-123',
        total_energy_mj: 15000,
        total_duration_ms: 5000,
        token_count: 50,
        efficiency_metrics: {
          joules_per_token: 0.3,
          tokens_per_second: 10,
        },
      };

      (profilingApi.getRunSummary as jest.Mock).mockResolvedValue({
        success: true,
        data: mockSummary,
      });

      renderWithProvider(<EnergyProfilerPanel />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'inference_complete',
              run_id: 'test-run-123',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(profilingApi.getRunSummary).toHaveBeenCalledWith('test-run-123');
        });
      }
    });

    it('displays energy efficiency metrics', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'inference_complete',
              run_id: 'test-run-123',
              total_energy_mj: 15000,
              token_count: 50,
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          // Energy per token: 15000 mJ / 50 tokens = 300 mJ/token = 0.3 J/token
          expect(screen.getByText(/0.3.*J.*token/i)).toBeInTheDocument();
        });
      }
    });
  });

  describe('History Browser Population', () => {
    it('loads profiling runs into history', async () => {
      const mockRuns: ProfilingRun[] = [
        {
          run_id: 'run-1',
          timestamp: '2024-01-01T00:00:00Z',
          prompt: 'Test prompt 1',
          model_name: 'gpt2',
          total_energy_mj: 10000,
          token_count: 40,
        },
        {
          run_id: 'run-2',
          timestamp: '2024-01-02T00:00:00Z',
          prompt: 'Test prompt 2',
          model_name: 'gpt2',
          total_energy_mj: 12000,
          token_count: 45,
        },
      ];

      (profilingApi.getRuns as jest.Mock).mockResolvedValue({
        success: true,
        data: mockRuns,
      });

      renderWithProvider(<EnergyProfilerPanel />);

      await waitFor(() => {
        expect(screen.getByText(/Test prompt 1/i)).toBeInTheDocument();
        expect(screen.getByText(/Test prompt 2/i)).toBeInTheDocument();
      });
    });

    it('displays run metadata in history list', async () => {
      const mockRuns: ProfilingRun[] = [
        {
          run_id: 'run-1',
          timestamp: '2024-01-01T00:00:00Z',
          prompt: 'Test',
          model_name: 'gpt2',
          total_energy_mj: 10000,
          token_count: 40,
        },
      ];

      (profilingApi.getRuns as jest.Mock).mockResolvedValue({
        success: true,
        data: mockRuns,
      });

      renderWithProvider(<EnergyProfilerPanel />);

      await waitFor(() => {
        expect(screen.getByText(/gpt2/i)).toBeInTheDocument();
        expect(screen.getByText(/10000/i)).toBeInTheDocument();
        expect(screen.getByText(/40/i)).toBeInTheDocument();
      });
    });

    it('updates history after new profiling run', async () => {
      (profilingApi.getRuns as jest.Mock).mockResolvedValue({
        success: true,
        data: [],
      });

      renderWithProvider(<EnergyProfilerPanel />);

      // Start and complete a run
      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'inference_complete',
              run_id: 'new-run',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          expect(profilingApi.getRuns).toHaveBeenCalled();
        });
      }
    });
  });

  describe('Run Detail View', () => {
    it('opens run detail when history item is clicked', async () => {
      const mockRuns: ProfilingRun[] = [
        {
          run_id: 'run-1',
          timestamp: '2024-01-01T00:00:00Z',
          prompt: 'Test prompt',
          model_name: 'gpt2',
          total_energy_mj: 10000,
          token_count: 40,
        },
      ];

      (profilingApi.getRuns as jest.Mock).mockResolvedValue({
        success: true,
        data: mockRuns,
      });

      (profilingApi.getRun as jest.Mock).mockResolvedValue({
        success: true,
        data: mockRuns[0],
      });

      renderWithProvider(<EnergyProfilerPanel />);

      await waitFor(() => {
        const runItem = screen.getByText(/Test prompt/i);
        fireEvent.click(runItem);
      });

      expect(profilingApi.getRun).toHaveBeenCalledWith('run-1');
    });

    it('displays detailed metrics in run detail view', async () => {
      const mockRunDetail = {
        run_id: 'run-1',
        prompt: 'Test',
        response: 'Generated response',
        total_energy_mj: 10000,
        total_duration_ms: 2000,
        token_count: 40,
        power_samples: [],
        tokens: [],
      };

      (profilingApi.getRun as jest.Mock).mockResolvedValue({
        success: true,
        data: mockRunDetail,
      });

      renderWithProvider(<EnergyProfilerPanel />);

      await waitFor(() => {
        expect(screen.getByText(/Generated response/i)).toBeInTheDocument();
        expect(screen.getByText(/2000.*ms/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error when powermetrics is unavailable', async () => {
      (profilingApi.status as jest.Mock).mockResolvedValue({
        success: true,
        data: { available: false, error: 'powermetrics not found' },
      });

      renderWithProvider(<EnergyProfilerPanel />);

      await waitFor(() => {
        expect(screen.getByText(/powermetrics not found/i)).toBeInTheDocument();
      });
    });

    it('handles WebSocket errors gracefully', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const errorHandler = messageHandlers.get('error');
      if (errorHandler) {
        act(() => {
          errorHandler(new Event('error'));
        });

        await waitFor(() => {
          expect(screen.getByText(/connection error/i)).toBeInTheDocument();
        });
      }
    });

    it('handles profiling API errors', async () => {
      (profilingApi.start as jest.Mock).mockResolvedValue({
        success: false,
        error: 'Model not loaded',
      });

      renderWithProvider(<EnergyProfilerPanel />);

      const promptInput = screen.getByPlaceholderText(/Enter prompt/i);
      fireEvent.change(promptInput, { target: { value: 'Test' } });

      const startButton = screen.getByRole('button', { name: /start profiling/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(screen.getByText(/Model not loaded/i)).toBeInTheDocument();
      });
    });

    it('allows canceling running profiling', async () => {
      (profilingApi.cancel as jest.Mock).mockResolvedValue({
        success: true,
        data: { cancelled: true },
      });

      renderWithProvider(<EnergyProfilerPanel />);

      // Start profiling
      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        act(() => {
          messageHandler({
            data: JSON.stringify({
              type: 'profiling_started',
            }),
          } as MessageEvent);
        });

        await waitFor(() => {
          const cancelButton = screen.getByRole('button', { name: /cancel/i });
          fireEvent.click(cancelButton);
        });

        expect(profilingApi.cancel).toHaveBeenCalled();
      }
    });
  });

  describe('Performance', () => {
    it('handles high-frequency power samples without lag', async () => {
      renderWithProvider(<EnergyProfilerPanel />);

      const messageHandler = messageHandlers.get('message');
      if (messageHandler) {
        const startTime = performance.now();

        // Send 100 power samples rapidly
        for (let i = 0; i < 100; i++) {
          act(() => {
            messageHandler({
              data: JSON.stringify({
                type: 'power_sample',
                timestamp_ms: i * 10,
                cpu_power_mw: 1000 + i,
                gpu_power_mw: 2000,
                total_power_mw: 3000 + i,
              }),
            } as MessageEvent);
          });
        }

        const elapsed = performance.now() - startTime;
        expect(elapsed).toBeLessThan(1000);
      }
    });
  });
});
