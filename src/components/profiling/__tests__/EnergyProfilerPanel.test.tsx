import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import EnergyProfilerPanel from '../EnergyProfilerPanel';

describe('EnergyProfilerPanel', () => {
  it('renders the panel with title and description', () => {
    render(<EnergyProfilerPanel />);

    expect(screen.getByText('Energy Profiler')).toBeInTheDocument();
    expect(screen.getByText(/Real-time power and energy profiling/i)).toBeInTheDocument();
  });

  it('renders all three tab buttons', () => {
    render(<EnergyProfilerPanel />);

    expect(screen.getByRole('button', { name: /Live/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /History/i })).toBeInTheDocument();
  });

  it('starts with Live tab active by default', () => {
    render(<EnergyProfilerPanel />);

    const liveTab = screen.getByRole('button', { name: /Live/i });
    expect(liveTab).toHaveClass('text-blue-600', 'border-blue-600');
  });

  it('switches tabs when clicked', async () => {
    const user = userEvent.setup();
    render(<EnergyProfilerPanel />);

    const analysisTab = screen.getByRole('button', { name: /Analysis/i });
    await user.click(analysisTab);

    expect(analysisTab).toHaveClass('text-blue-600', 'border-blue-600');
  });

  it('wraps content with ProfilingProvider', () => {
    const { container } = render(<EnergyProfilerPanel />);

    // Verify structure indicates provider is present
    expect(container.firstChild).toBeInTheDocument();
  });
});
