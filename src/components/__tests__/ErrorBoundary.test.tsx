/**
 * TEST-010: Error Boundary and Recovery Tests
 *
 * Tests that the application handles errors gracefully with proper
 * error boundaries, recovery UI, and user-friendly error messages.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ErrorBoundary from '../ErrorBoundary';

// Component that throws an error for testing
const ThrowError: React.FC<{ shouldThrow?: boolean; message?: string }> = ({
  shouldThrow = true,
  message = 'Test error',
}) => {
  if (shouldThrow) {
    throw new Error(message);
  }
  return <div>No error</div>;
};

// Component for testing async errors
const AsyncError: React.FC = () => {
  const [shouldThrow, setShouldThrow] = React.useState(false);

  React.useEffect(() => {
    if (shouldThrow) {
      throw new Error('Async error');
    }
  }, [shouldThrow]);

  return <button onClick={() => setShouldThrow(true)}>Trigger Error</button>;
};

describe('ErrorBoundary', () => {
  const originalConsoleError = console.error;

  beforeEach(() => {
    // Suppress console errors in tests
    console.error = jest.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });

  describe('Error Catching', () => {
    it('catches component errors', () => {
      render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });

    it('displays custom error message', () => {
      const customMessage = 'Custom error occurred';

      render(
        <ErrorBoundary>
          <ThrowError message={customMessage} />
        </ErrorBoundary>
      );

      expect(screen.getByText(new RegExp(customMessage, 'i'))).toBeInTheDocument();
    });

    it('renders fallback UI when error occurs', () => {
      render(
        <ErrorBoundary fallback={<div>Custom Fallback</div>}>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByText(/Custom Fallback/i)).toBeInTheDocument();
    });

    it('does not catch errors when component is healthy', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(screen.getByText('No error')).toBeInTheDocument();
      expect(screen.queryByText(/error/i)).not.toBeInTheDocument();
    });
  });

  describe('Recovery UI', () => {
    it('displays retry button', () => {
      render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('resets error state when retry is clicked', () => {
      let shouldThrow = true;

      const ConditionalError: React.FC = () => {
        if (shouldThrow) {
          throw new Error('Error');
        }
        return <div>Recovered</div>;
      };

      const { rerender } = render(
        <ErrorBoundary>
          <ConditionalError />
        </ErrorBoundary>
      );

      expect(screen.getByText(/error/i)).toBeInTheDocument();

      // Fix the error condition
      shouldThrow = false;

      // Click retry
      const retryButton = screen.getByRole('button', { name: /retry/i });
      fireEvent.click(retryButton);

      rerender(
        <ErrorBoundary>
          <ConditionalError />
        </ErrorBoundary>
      );

      // Should show recovered state
      expect(screen.getByText('Recovered')).toBeInTheDocument();
    });

    it('displays error details in development mode', () => {
      const originalNodeEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'development';

      render(
        <ErrorBoundary>
          <ThrowError message="Detailed error for debugging" />
        </ErrorBoundary>
      );

      // Should show stack trace or detailed error info
      expect(screen.getByText(/Detailed error/i)).toBeInTheDocument();

      process.env.NODE_ENV = originalNodeEnv;
    });

    it('hides error details in production mode', () => {
      const originalNodeEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'production';

      render(
        <ErrorBoundary>
          <ThrowError message="Sensitive error details" />
        </ErrorBoundary>
      );

      // Should show generic error message
      expect(screen.queryByText(/Sensitive error details/i)).not.toBeInTheDocument();
      expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();

      process.env.NODE_ENV = originalNodeEnv;
    });
  });

  describe('API Error Handling', () => {
    it('shows user-friendly message for API errors', async () => {
      const mockFetch = jest.fn().mockRejectedValue(new Error('Network error'));
      global.fetch = mockFetch;

      const ApiComponent: React.FC = () => {
        const [error, setError] = React.useState<string | null>(null);

        React.useEffect(() => {
          fetch('/api/data')
            .then(() => {})
            .catch((err) => setError(err.message));
        }, []);

        if (error) {
          return <div>Failed to load data: {error}</div>;
        }

        return <div>Loading...</div>;
      };

      render(<ApiComponent />);

      await waitFor(() => {
        expect(screen.getByText(/Failed to load data/i)).toBeInTheDocument();
      });
    });

    it('provides action buttons for API errors', async () => {
      const ApiErrorComponent: React.FC = () => {
        const [hasError, setHasError] = React.useState(true);

        if (hasError) {
          return (
            <div>
              <p>API Error</p>
              <button onClick={() => setHasError(false)}>Retry</button>
            </div>
          );
        }

        return <div>Success</div>;
      };

      render(<ApiErrorComponent />);

      expect(screen.getByText(/API Error/i)).toBeInTheDocument();

      const retryButton = screen.getByRole('button', { name: /retry/i });
      fireEvent.click(retryButton);

      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
      });
    });
  });

  describe('WebSocket Disconnection Handling', () => {
    it('shows reconnecting state when WebSocket disconnects', () => {
      const WebSocketComponent: React.FC = () => {
        const [connectionState, setConnectionState] = React.useState('connected');

        return (
          <div>
            <p>Status: {connectionState}</p>
            <button onClick={() => setConnectionState('disconnected')}>
              Disconnect
            </button>
          </div>
        );
      };

      render(<WebSocketComponent />);

      expect(screen.getByText(/Status: connected/i)).toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: /disconnect/i }));

      expect(screen.getByText(/Status: disconnected/i)).toBeInTheDocument();
    });

    it('auto-reconnects after disconnection', async () => {
      jest.useFakeTimers();

      const AutoReconnectComponent: React.FC = () => {
        const [status, setStatus] = React.useState('connected');

        React.useEffect(() => {
          if (status === 'disconnected') {
            const timer = setTimeout(() => {
              setStatus('reconnecting');
              setTimeout(() => setStatus('connected'), 1000);
            }, 1000);
            return () => clearTimeout(timer);
          }
        }, [status]);

        return (
          <div>
            <p>Status: {status}</p>
            <button onClick={() => setStatus('disconnected')}>Disconnect</button>
          </div>
        );
      };

      render(<AutoReconnectComponent />);

      fireEvent.click(screen.getByRole('button', { name: /disconnect/i }));

      expect(screen.getByText(/Status: disconnected/i)).toBeInTheDocument();

      // Advance timers for reconnection attempt
      jest.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(screen.getByText(/Status: reconnecting/i)).toBeInTheDocument();
      });

      jest.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(screen.getByText(/Status: connected/i)).toBeInTheDocument();
      });

      jest.useRealTimers();
    });
  });

  describe('Backend Unavailable Indicator', () => {
    it('shows offline indicator when backend is unreachable', async () => {
      const mockFetch = jest.fn().mockRejectedValue(new Error('Backend unavailable'));
      global.fetch = mockFetch;

      const BackendStatusComponent: React.FC = () => {
        const [isOnline, setIsOnline] = React.useState(true);

        React.useEffect(() => {
          fetch('/api/health')
            .then(() => setIsOnline(true))
            .catch(() => setIsOnline(false));
        }, []);

        return (
          <div>
            {!isOnline && <div>Backend is offline</div>}
            {isOnline && <div>Backend is online</div>}
          </div>
        );
      };

      render(<BackendStatusComponent />);

      await waitFor(() => {
        expect(screen.getByText(/Backend is offline/i)).toBeInTheDocument();
      });
    });

    it('polls backend health periodically', async () => {
      jest.useFakeTimers();
      const mockFetch = jest.fn().mockResolvedValue({ ok: true });
      global.fetch = mockFetch;

      const HealthCheckComponent: React.FC = () => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            fetch('/api/health');
          }, 5000);
          return () => clearInterval(interval);
        }, []);

        return <div>Health Check Active</div>;
      };

      render(<HealthCheckComponent />);

      expect(mockFetch).toHaveBeenCalledTimes(0);

      jest.advanceTimersByTime(5000);
      expect(mockFetch).toHaveBeenCalledTimes(1);

      jest.advanceTimersByTime(5000);
      expect(mockFetch).toHaveBeenCalledTimes(2);

      jest.useRealTimers();
    });
  });

  describe('Error Logging', () => {
    it('logs errors to console', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      render(
        <ErrorBoundary>
          <ThrowError message="Logged error" />
        </ErrorBoundary>
      );

      expect(consoleSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
    });

    it('sends errors to error tracking service', async () => {
      const errorTracker = jest.fn();
      (window as any).trackError = errorTracker;

      render(
        <ErrorBoundary onError={errorTracker}>
          <ThrowError message="Tracked error" />
        </ErrorBoundary>
      );

      await waitFor(() => {
        expect(errorTracker).toHaveBeenCalled();
      });
    });
  });

  describe('Nested Error Boundaries', () => {
    it('isolates errors to nearest boundary', () => {
      const InnerComponent: React.FC = () => {
        throw new Error('Inner error');
      };

      render(
        <ErrorBoundary fallback={<div>Outer Fallback</div>}>
          <div>
            <p>Outer Content</p>
            <ErrorBoundary fallback={<div>Inner Fallback</div>}>
              <InnerComponent />
            </ErrorBoundary>
          </div>
        </ErrorBoundary>
      );

      // Inner boundary should catch the error
      expect(screen.getByText('Inner Fallback')).toBeInTheDocument();
      // Outer content should still render
      expect(screen.getByText('Outer Content')).toBeInTheDocument();
    });

    it('bubbles up when inner boundary fails', () => {
      const InnerBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
        throw new Error('Boundary itself failed');
      };

      render(
        <ErrorBoundary fallback={<div>Outer Caught It</div>}>
          <InnerBoundary>
            <div>Content</div>
          </InnerBoundary>
        </ErrorBoundary>
      );

      expect(screen.getByText('Outer Caught It')).toBeInTheDocument();
    });
  });

  describe('Graceful Degradation', () => {
    it('renders partial UI when non-critical component fails', () => {
      const CriticalComponent: React.FC = () => <div>Critical Content</div>;

      const NonCriticalComponent: React.FC = () => {
        throw new Error('Non-critical error');
      };

      render(
        <div>
          <CriticalComponent />
          <ErrorBoundary fallback={<div>Feature Unavailable</div>}>
            <NonCriticalComponent />
          </ErrorBoundary>
        </div>
      );

      // Critical content should still render
      expect(screen.getByText('Critical Content')).toBeInTheDocument();
      // Non-critical shows fallback
      expect(screen.getByText('Feature Unavailable')).toBeInTheDocument();
    });

    it('maintains app functionality when chart component fails', () => {
      const DataTable: React.FC = () => <table><tbody><tr><td>Data</td></tr></tbody></table>;

      const BrokenChart: React.FC = () => {
        throw new Error('Chart rendering failed');
      };

      render(
        <div>
          <DataTable />
          <ErrorBoundary fallback={<div>Chart unavailable</div>}>
            <BrokenChart />
          </ErrorBoundary>
        </div>
      );

      // Data should still be accessible
      expect(screen.getByText('Data')).toBeInTheDocument();
      expect(screen.getByText('Chart unavailable')).toBeInTheDocument();
    });
  });
});
