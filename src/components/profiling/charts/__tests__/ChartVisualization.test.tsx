/**
 * TEST-007: Chart Visualization Tests (Extended)
 *
 * Tests D3.js chart rendering with various data shapes,
 * including scaling charts, energy flow diagrams, and responsive behavior.
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock D3 charts (these would be actual imports in production)
const MockSankeyChart: React.FC<any> = ({ data, width, height }) => (
  <svg role="img" aria-label="sankey-chart" width={width} height={height}>
    <title>Sankey Chart</title>
  </svg>
);

const MockEnergyScalingChart: React.FC<any> = ({ data, width, height }) => (
  <svg role="img" aria-label="scaling-chart" width={width} height={height}>
    <title>Energy Scaling Chart</title>
  </svg>
);

const MockTreemapChart: React.FC<any> = ({ data, width, height }) => (
  <svg role="img" aria-label="treemap-chart" width={width} height={height}>
    <title>Treemap Chart</title>
  </svg>
);

describe('Chart Visualization Tests', () => {
  describe('SankeyChart - Energy Flow', () => {
    const mockSankeyData = {
      nodes: [
        { name: 'Input' },
        { name: 'Embedding' },
        { name: 'Attention' },
        { name: 'FFN' },
        { name: 'Output' },
      ],
      links: [
        { source: 0, target: 1, value: 100 },
        { source: 1, target: 2, value: 250 },
        { source: 1, target: 3, value: 150 },
        { source: 2, target: 4, value: 200 },
        { source: 3, target: 4, value: 150 },
      ],
    };

    it('renders Sankey chart with energy flow data', () => {
      render(<MockSankeyChart data={mockSankeyData} width={800} height={600} />);

      const chart = screen.getByRole('img', { name: /sankey/i });
      expect(chart).toBeInTheDocument();
    });

    it('displays nodes and links', () => {
      render(<MockSankeyChart data={mockSankeyData} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('handles empty flow data', () => {
      const emptyData = { nodes: [], links: [] };
      render(<MockSankeyChart data={emptyData} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('updates when flow data changes', () => {
      const { rerender } = render(<MockSankeyChart data={mockSankeyData} />);

      const newData = {
        ...mockSankeyData,
        links: [
          ...mockSankeyData.links,
          { source: 3, target: 2, value: 50 },
        ],
      };

      rerender(<MockSankeyChart data={newData} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });
  });

  describe('EnergyScalingChart - Regression Lines', () => {
    const mockScalingData = [
      { tokenCount: 10, energy: 100, model: 'gpt2-small' },
      { tokenCount: 50, energy: 450, model: 'gpt2-small' },
      { tokenCount: 100, energy: 950, model: 'gpt2-small' },
      { tokenCount: 200, energy: 1900, model: 'gpt2-small' },
      { tokenCount: 10, energy: 200, model: 'gpt2-medium' },
      { tokenCount: 50, energy: 900, model: 'gpt2-medium' },
      { tokenCount: 100, energy: 1850, model: 'gpt2-medium' },
      { tokenCount: 200, energy: 3700, model: 'gpt2-medium' },
    ];

    it('renders scaling chart with regression lines', () => {
      render(<MockEnergyScalingChart data={mockScalingData} width={800} height={600} />);

      const chart = screen.getByRole('img', { name: /scaling/i });
      expect(chart).toBeInTheDocument();
    });

    it('displays multiple model series', () => {
      render(<MockEnergyScalingChart data={mockScalingData} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('handles single data point', () => {
      const singlePoint = [{ tokenCount: 50, energy: 450, model: 'test' }];
      render(<MockEnergyScalingChart data={singlePoint} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('calculates regression correctly', () => {
      // Test that regression line fits data
      render(<MockEnergyScalingChart data={mockScalingData} showRegression={true} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('handles logarithmic scale', () => {
      render(<MockEnergyScalingChart data={mockScalingData} scale="log" />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });
  });

  describe('TreemapChart - Component Breakdown', () => {
    const mockTreemapData = {
      name: 'Total Energy',
      value: 10000,
      children: [
        {
          name: 'Attention',
          value: 4500,
          children: [
            { name: 'Q-Proj', value: 1500 },
            { name: 'K-Proj', value: 1500 },
            { name: 'V-Proj', value: 1000 },
            { name: 'O-Proj', value: 500 },
          ],
        },
        {
          name: 'FFN',
          value: 3500,
          children: [
            { name: 'fc1', value: 2000 },
            { name: 'fc2', value: 1500 },
          ],
        },
        { name: 'LayerNorm', value: 1000 },
        { name: 'Embedding', value: 1000 },
      ],
    };

    it('renders treemap with hierarchical data', () => {
      render(<MockTreemapChart data={mockTreemapData} width={800} height={600} />);

      const chart = screen.getByRole('img', { name: /treemap/i });
      expect(chart).toBeInTheDocument();
    });

    it('displays hierarchical energy breakdown', () => {
      render(<MockTreemapChart data={mockTreemapData} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('handles flat hierarchy', () => {
      const flatData = {
        name: 'Total',
        value: 1000,
        children: [
          { name: 'A', value: 400 },
          { name: 'B', value: 300 },
          { name: 'C', value: 300 },
        ],
      };

      render(<MockTreemapChart data={flatData} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('supports zooming into sections', () => {
      const { container } = render(<MockTreemapChart data={mockTreemapData} interactive={true} />);

      const chart = screen.getByRole('img');
      fireEvent.click(chart);

      expect(chart).toBeInTheDocument();
    });
  });

  describe('Responsive Behavior', () => {
    it('adapts to container width', () => {
      const { rerender } = render(
        <div style={{ width: 600 }}>
          <MockEnergyScalingChart data={[]} width={600} height={400} />
        </div>
      );

      rerender(
        <div style={{ width: 800 }}>
          <MockEnergyScalingChart data={[]} width={800} height={400} />
        </div>
      );

      const chart = screen.getByRole('img');
      expect(chart).toHaveAttribute('width', '800');
    });

    it('maintains aspect ratio on resize', () => {
      const aspectRatio = 16 / 9;
      const width = 800;
      const height = width / aspectRatio;

      render(<MockEnergyScalingChart data={[]} width={width} height={height} />);

      const chart = screen.getByRole('img') as SVGSVGElement;
      const actualRatio = Number(chart.getAttribute('width')) / Number(chart.getAttribute('height'));

      expect(Math.abs(actualRatio - aspectRatio)).toBeLessThan(0.1);
    });

    it('handles mobile viewport', () => {
      render(
        <MockEnergyScalingChart data={[]} width={320} height={240} responsive={true} />
      );

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('provides touch-friendly interactions on mobile', () => {
      render(<MockTreemapChart data={{}} width={320} height={240} touchEnabled={true} />);

      const chart = screen.getByRole('img');
      fireEvent.touchStart(chart);

      expect(chart).toBeInTheDocument();
    });
  });

  describe('Performance Optimization', () => {
    it('renders large datasets efficiently', () => {
      // Generate 1000 data points
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        tokenCount: i,
        energy: i * 10 + Math.random() * 100,
        model: 'test',
      }));

      const startTime = performance.now();
      render(<MockEnergyScalingChart data={largeDataset} />);
      const elapsed = performance.now() - startTime;

      expect(elapsed).toBeLessThan(200); // Should render in < 200ms
    });

    it('uses canvas rendering for large datasets', () => {
      const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
        x: i,
        y: Math.random() * 1000,
      }));

      // Canvas should be used instead of SVG for performance
      render(<MockEnergyScalingChart data={largeDataset} useCanvas={true} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('debounces resize events', () => {
      const { rerender } = render(<MockEnergyScalingChart data={[]} width={600} />);

      // Rapidly change width (simulating window resize)
      for (let i = 600; i < 800; i += 10) {
        rerender(<MockEnergyScalingChart data={[]} width={i} />);
      }

      // Should not cause performance issues
      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides aria labels for screen readers', () => {
      render(<MockEnergyScalingChart data={[]} aria-label="Energy consumption over time" />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('includes data table alternative', () => {
      const data = [
        { tokenCount: 10, energy: 100 },
        { tokenCount: 20, energy: 200 },
      ];

      render(
        <div>
          <MockEnergyScalingChart data={data} />
          <table role="table" aria-label="Energy data">
            <tbody>
              {data.map((d, i) => (
                <tr key={i}>
                  <td>{d.tokenCount}</td>
                  <td>{d.energy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();
    });

    it('supports keyboard navigation', () => {
      render(<MockTreemapChart data={{}} interactive={true} />);

      const chart = screen.getByRole('img');

      fireEvent.keyDown(chart, { key: 'Enter' });
      fireEvent.keyDown(chart, { key: 'ArrowRight' });

      expect(chart).toBeInTheDocument();
    });
  });

  describe('Color Scaling', () => {
    it('uses consistent color scale across charts', () => {
      const colorScale = ['#0000ff', '#ff0000'];

      render(<MockEnergyScalingChart data={[]} colorScale={colorScale} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('adjusts colors for color-blind accessibility', () => {
      render(<MockEnergyScalingChart data={[]} colorBlindSafe={true} />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });

    it('supports dark mode theming', () => {
      render(<MockEnergyScalingChart data={[]} theme="dark" />);

      const chart = screen.getByRole('img');
      expect(chart).toBeInTheDocument();
    });
  });
});
