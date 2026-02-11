import React from 'react';
import { render, screen } from '@testing-library/react';
import PowerTimeSeriesChart from '../PowerTimeSeriesChart';
import type { PowerSample } from '@/types';

describe('PowerTimeSeriesChart', () => {
  const mockPowerSamples: PowerSample[] = [
    {
      timestamp: 0,
      cpu_power_mw: 1000,
      gpu_power_mw: 2000,
      ane_power_mw: 500,
      dram_power_mw: 300,
      total_power_mw: 3800,
    },
    {
      timestamp: 100,
      cpu_power_mw: 1200,
      gpu_power_mw: 2500,
      ane_power_mw: 600,
      dram_power_mw: 350,
      total_power_mw: 4650,
    },
    {
      timestamp: 200,
      cpu_power_mw: 1100,
      gpu_power_mw: 2300,
      ane_power_mw: 550,
      dram_power_mw: 320,
      total_power_mw: 4270,
    },
  ];

  it('renders canvas element', () => {
    render(<PowerTimeSeriesChart samples={mockPowerSamples} />);

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
    expect(canvas.tagName).toBe('CANVAS');
  });

  it('renders with empty samples', () => {
    render(<PowerTimeSeriesChart samples={[]} />);

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
  });

  it('handles width and height props', () => {
    render(<PowerTimeSeriesChart samples={mockPowerSamples} width={800} height={400} />);

    const canvas = screen.getByRole('img') as HTMLCanvasElement;
    expect(canvas.width).toBe(800);
    expect(canvas.height).toBe(400);
  });

  it('updates chart when samples change', () => {
    const { rerender } = render(<PowerTimeSeriesChart samples={mockPowerSamples} />);

    const newSamples: PowerSample[] = [
      ...mockPowerSamples,
      {
        timestamp: 300,
        cpu_power_mw: 1300,
        gpu_power_mw: 2600,
        ane_power_mw: 650,
        dram_power_mw: 380,
        total_power_mw: 4930,
      },
    ];

    rerender(<PowerTimeSeriesChart samples={newSamples} />);

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
  });

  it('handles live data updates', () => {
    const { rerender } = render(<PowerTimeSeriesChart samples={[mockPowerSamples[0]]} />);

    // Simulate live streaming by adding samples one at a time
    for (let i = 1; i < mockPowerSamples.length; i++) {
      rerender(<PowerTimeSeriesChart samples={mockPowerSamples.slice(0, i + 1)} />);
    }

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
  });

  it('handles large datasets efficiently', () => {
    // Generate 1000 samples
    const largeSamples: PowerSample[] = Array.from({ length: 1000 }, (_, i) => ({
      timestamp: i * 10,
      cpu_power_mw: 1000 + Math.random() * 500,
      gpu_power_mw: 2000 + Math.random() * 1000,
      ane_power_mw: 500 + Math.random() * 200,
      dram_power_mw: 300 + Math.random() * 100,
      total_power_mw: 3800 + Math.random() * 1800,
    }));

    const startTime = performance.now();
    render(<PowerTimeSeriesChart samples={largeSamples} />);
    const elapsed = performance.now() - startTime;

    expect(elapsed).toBeLessThan(100); // Should render in < 100ms
  });

  it('handles responsive resize', () => {
    const { rerender } = render(<PowerTimeSeriesChart samples={mockPowerSamples} width={600} height={300} />);

    // Simulate window resize
    rerender(<PowerTimeSeriesChart samples={mockPowerSamples} width={800} height={400} />);

    const canvas = screen.getByRole('img') as HTMLCanvasElement;
    expect(canvas.width).toBe(800);
    expect(canvas.height).toBe(400);

    const canvas = screen.getByRole('img') as HTMLCanvasElement;
    expect(canvas.width).toBe(800);
    expect(canvas.height).toBe(400);
  });

  it('applies custom className', () => {
    render(<PowerTimeSeriesChart samples={mockPowerSamples} className="custom-class" />);

    const canvas = screen.getByRole('img');
    expect(canvas).toHaveClass('custom-class');
  });

  it('renders legend with component labels', () => {
    render(<PowerTimeSeriesChart samples={mockPowerSamples} />);

    expect(screen.getByText(/CPU/i)).toBeInTheDocument();
    expect(screen.getByText(/GPU/i)).toBeInTheDocument();
    expect(screen.getByText(/ANE/i)).toBeInTheDocument();
    expect(screen.getByText(/DRAM/i)).toBeInTheDocument();
    expect(screen.getByText(/Total/i)).toBeInTheDocument();
  });

  it('updates when samples change', () => {
    const { rerender } = render(<PowerTimeSeriesChart samples={mockPowerSamples} />);

    const newSamples: PowerSample[] = [
      ...mockPowerSamples,
      {
        timestamp: 300,
        cpu_power_mw: 1300,
        gpu_power_mw: 2600,
        ane_power_mw: 650,
        dram_power_mw: 370,
        total_power_mw: 4920,
      },
    ];

    rerender(<PowerTimeSeriesChart samples={newSamples} />);

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
  });
});
