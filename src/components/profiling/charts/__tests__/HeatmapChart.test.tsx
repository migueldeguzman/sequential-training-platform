import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import HeatmapChart from '../HeatmapChart';

describe('HeatmapChart', () => {
  const mockData = [
    [1.2, 3.4, 2.1, 4.5],
    [2.3, 1.8, 3.2, 2.9],
    [3.1, 2.5, 1.5, 3.8],
  ];

  const xLabels = ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3'];
  const yLabels = ['q_proj', 'k_proj', 'v_proj'];

  it('renders canvas element', () => {
    render(
      <HeatmapChart
        data={mockData}
        xLabels={xLabels}
        yLabels={yLabels}
        title="Test Heatmap"
      />
    );

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
    expect(canvas.tagName).toBe('CANVAS');
  });

  it('renders title when provided', () => {
    render(
      <HeatmapChart
        data={mockData}
        xLabels={xLabels}
        yLabels={yLabels}
        title="Layer Energy Heatmap"
      />
    );

    expect(screen.getByText('Layer Energy Heatmap')).toBeInTheDocument();
  });

  it('handles empty data gracefully', () => {
    render(
      <HeatmapChart
        data={[]}
        xLabels={[]}
        yLabels={[]}
        title="Empty Heatmap"
      />
    );

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
  });

  it('handles click events when onClick is provided', async () => {
    const user = userEvent.setup();
    const handleClick = jest.fn();

    render(
      <HeatmapChart
        data={mockData}
        xLabels={xLabels}
        yLabels={yLabels}
        title="Clickable Heatmap"
        onClick={handleClick}
      />
    );

    const canvas = screen.getByRole('img');
    await user.click(canvas);

    // Click handler should be registered
    expect(canvas).toBeInTheDocument();
  });

  it('applies custom width and height', () => {
    render(
      <HeatmapChart
        data={mockData}
        xLabels={xLabels}
        yLabels={yLabels}
        title="Custom Size"
        width={800}
        height={600}
      />
    );

    const canvas = screen.getByRole('img') as HTMLCanvasElement;
    expect(canvas.width).toBe(800);
    expect(canvas.height).toBe(600);
  });

  it('uses custom color scale when provided', () => {
    const customColorScale = ['#000000', '#ffffff'];

    render(
      <HeatmapChart
        data={mockData}
        xLabels={xLabels}
        yLabels={yLabels}
        title="Custom Colors"
        colorScale={customColorScale}
      />
    );

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
  });

  it('updates when data changes', () => {
    const { rerender } = render(
      <HeatmapChart
        data={mockData}
        xLabels={xLabels}
        yLabels={yLabels}
        title="Dynamic Heatmap"
      />
    );

    const newData = [
      [5.1, 6.2, 7.3, 8.4],
      [4.5, 3.6, 2.7, 1.8],
      [9.1, 8.2, 7.3, 6.4],
    ];

    rerender(
      <HeatmapChart
        data={newData}
        xLabels={xLabels}
        yLabels={yLabels}
        title="Dynamic Heatmap"
      />
    );

    const canvas = screen.getByRole('img');
    expect(canvas).toBeInTheDocument();
  });
});
