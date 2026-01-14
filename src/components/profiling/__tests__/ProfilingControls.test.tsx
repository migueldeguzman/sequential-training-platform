import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ProfilingControls } from '../ProfilingControls';
import { ProfilingProvider } from '../ProfilingContext';

// Mock the API
jest.mock('@/lib/api', () => ({
  api: {
    profiledGenerate: jest.fn(),
  },
}));

// Mock the WebSocket
jest.mock('@/lib/profilingWebsocket', () => ({
  useProfilingWebSocket: () => ({
    connect: jest.fn(),
    disconnect: jest.fn(),
    connectionState: 'disconnected' as const,
    error: null,
  }),
}));

const renderWithContext = (component: React.ReactElement) => {
  return render(<ProfilingProvider>{component}</ProfilingProvider>);
};

describe('ProfilingControls', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the profiling controls form', () => {
    renderWithContext(<ProfilingControls />);

    expect(screen.getByText('Profiling Controls')).toBeInTheDocument();
    expect(screen.getByLabelText(/Model Path/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Prompt/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Profiling Depth/i)).toBeInTheDocument();
  });

  it('shows Ready status when not profiling', () => {
    renderWithContext(<ProfilingControls />);

    expect(screen.getByText('Ready')).toBeInTheDocument();
  });

  it('disables form inputs when profiling is running', async () => {
    const user = userEvent.setup();
    renderWithContext(<ProfilingControls />);

    const promptInput = screen.getByLabelText(/Prompt/i);
    await user.type(promptInput, 'Test prompt');

    const startButton = screen.getByRole('button', { name: /Start Profiling/i });
    await user.click(startButton);

    // After starting, inputs should be disabled
    await waitFor(() => {
      const modelInput = screen.getByLabelText(/Model Path/i);
      expect(modelInput).toBeDisabled();
    });
  });

  it('validates prompt is not empty before starting', async () => {
    const user = userEvent.setup();
    const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});

    renderWithContext(<ProfilingControls />);

    const startButton = screen.getByRole('button', { name: /Start Profiling/i });
    await user.click(startButton);

    expect(alertMock).toHaveBeenCalledWith('Please enter a prompt');
    alertMock.mockRestore();
  });

  it('accepts user input for all form fields', async () => {
    const user = userEvent.setup();
    renderWithContext(<ProfilingControls />);

    // Model path
    const modelInput = screen.getByLabelText(/Model Path/i);
    await user.type(modelInput, 'test-model');
    expect(modelInput).toHaveValue('test-model');

    // Prompt
    const promptInput = screen.getByLabelText(/Prompt/i);
    await user.type(promptInput, 'Test prompt');
    expect(promptInput).toHaveValue('Test prompt');

    // Tags
    const tagsInput = screen.getByLabelText(/Tags/i);
    await user.type(tagsInput, 'test, experiment');
    expect(tagsInput).toHaveValue('test, experiment');

    // Experiment name
    const expInput = screen.getByLabelText(/Experiment Name/i);
    await user.type(expInput, 'Test Experiment');
    expect(expInput).toHaveValue('Test Experiment');
  });

  it('toggles profiling depth between module and deep', async () => {
    const user = userEvent.setup();
    renderWithContext(<ProfilingControls />);

    const depthSelect = screen.getByLabelText(/Profiling Depth/i);
    expect(depthSelect).toHaveValue('module');

    await user.selectOptions(depthSelect, 'deep');
    expect(depthSelect).toHaveValue('deep');
  });

  it('adjusts temperature with slider', async () => {
    const user = userEvent.setup();
    renderWithContext(<ProfilingControls />);

    const tempSlider = screen.getByLabelText(/Temperature/i);
    expect(tempSlider).toHaveValue('0.7');

    await user.clear(tempSlider);
    await user.type(tempSlider, '0.9');
    expect(tempSlider).toHaveValue('0.9');
  });

  it('adjusts max length with input', async () => {
    const user = userEvent.setup();
    renderWithContext(<ProfilingControls />);

    const maxLengthInput = screen.getByLabelText(/Max Length/i);
    expect(maxLengthInput).toHaveValue(100);

    await user.clear(maxLengthInput);
    await user.type(maxLengthInput, '200');
    expect(maxLengthInput).toHaveValue(200);
  });

  it('renders Start Profiling button when not running', () => {
    renderWithContext(<ProfilingControls />);

    const startButton = screen.getByRole('button', { name: /Start Profiling/i });
    expect(startButton).toBeInTheDocument();
    expect(startButton).not.toBeDisabled();
  });
});
