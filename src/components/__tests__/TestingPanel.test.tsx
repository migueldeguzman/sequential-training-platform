/**
 * TEST-005: Model Loading and Inference Tests
 *
 * Tests model loading states and generation workflows including
 * single generation, loop generation, and batch generation.
 */

import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import TestingPanel from '../TestingPanel';
import { modelsApi, inferenceApi } from '@/lib/api';
import type { Model, InferenceResult } from '@/types';

// Mock the API modules
jest.mock('@/lib/api', () => ({
  modelsApi: {
    list: jest.fn(),
    get: jest.fn(),
  },
  inferenceApi: {
    status: jest.fn(),
    load: jest.fn(),
    unload: jest.fn(),
    generate: jest.fn(),
    generateLoop: jest.fn(),
    generateBatch: jest.fn(),
    export: jest.fn(),
  },
}));

describe('TestingPanel', () => {
  const mockModels: Model[] = [
    {
      name: 'gpt2-small',
      path: '/models/gpt2-small',
      size: '500 MB',
      architecture: 'GPT-2',
      parameters: '124M',
      modifiedAt: '2024-01-01T00:00:00Z',
    },
    {
      name: 'gpt2-medium',
      path: '/models/gpt2-medium',
      size: '1.5 GB',
      architecture: 'GPT-2',
      parameters: '355M',
      modifiedAt: '2024-01-02T00:00:00Z',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();

    // Setup default mock implementations
    (modelsApi.list as jest.Mock).mockResolvedValue({
      success: true,
      data: mockModels,
    });

    (inferenceApi.status as jest.Mock).mockResolvedValue({
      success: true,
      data: {
        loaded: false,
        model: null,
      },
    });
  });

  describe('Model Dropdown Population', () => {
    it('loads and displays available models', async () => {
      render(<TestingPanel />);

      await waitFor(() => {
        expect(modelsApi.list).toHaveBeenCalled();
      });

      // Check if models are in the dropdown
      await waitFor(() => {
        expect(screen.getByText(/gpt2-small/i)).toBeInTheDocument();
      });
    });

    it('handles empty model list', async () => {
      (modelsApi.list as jest.Mock).mockResolvedValue({
        success: true,
        data: [],
      });

      render(<TestingPanel />);

      await waitFor(() => {
        expect(screen.getByText(/No models found/i)).toBeInTheDocument();
      });
    });

    it('displays model metadata in dropdown', async () => {
      render(<TestingPanel />);

      await waitFor(() => {
        expect(screen.getByText(/124M/i)).toBeInTheDocument();
        expect(screen.getByText(/355M/i)).toBeInTheDocument();
      });
    });
  });

  describe('Loading/Unloading States', () => {
    it('loads a model when selected and load button clicked', async () => {
      (inferenceApi.load as jest.Mock).mockResolvedValue({
        success: true,
        data: { loaded: true, model: 'gpt2-small' },
      });

      render(<TestingPanel />);

      await waitFor(() => {
        expect(screen.getByText(/gpt2-small/i)).toBeInTheDocument();
      });

      // Select model
      const modelSelect = screen.getByRole('combobox');
      fireEvent.change(modelSelect, { target: { value: 'gpt2-small' } });

      // Click load button
      const loadButton = screen.getByRole('button', { name: /load/i });
      fireEvent.click(loadButton);

      await waitFor(() => {
        expect(inferenceApi.load).toHaveBeenCalledWith(
          expect.objectContaining({ modelName: 'gpt2-small' })
        );
      });
    });

    it('shows loading indicator while model loads', async () => {
      (inferenceApi.load as jest.Mock).mockImplementation(
        () =>
          new Promise((resolve) =>
            setTimeout(() => resolve({ success: true, data: {} }), 100)
          )
      );

      render(<TestingPanel />);

      await waitFor(() => {
        const modelSelect = screen.getByRole('combobox');
        fireEvent.change(modelSelect, { target: { value: 'gpt2-small' } });
      });

      const loadButton = screen.getByRole('button', { name: /load/i });
      fireEvent.click(loadButton);

      expect(screen.getByText(/Loading/i)).toBeInTheDocument();
    });

    it('displays loaded model status', async () => {
      (inferenceApi.status as jest.Mock).mockResolvedValue({
        success: true,
        data: {
          loaded: true,
          model: 'gpt2-small',
        },
      });

      render(<TestingPanel />);

      await waitFor(() => {
        expect(screen.getByText(/gpt2-small.*loaded/i)).toBeInTheDocument();
      });
    });

    it('unloads model when unload button clicked', async () => {
      (inferenceApi.status as jest.Mock).mockResolvedValue({
        success: true,
        data: { loaded: true, model: 'gpt2-small' },
      });

      (inferenceApi.unload as jest.Mock).mockResolvedValue({
        success: true,
        data: { loaded: false },
      });

      render(<TestingPanel />);

      await waitFor(() => {
        const unloadButton = screen.getByRole('button', { name: /unload/i });
        fireEvent.click(unloadButton);
      });

      expect(inferenceApi.unload).toHaveBeenCalled();
    });

    it('handles model loading errors', async () => {
      (inferenceApi.load as jest.Mock).mockResolvedValue({
        success: false,
        error: 'Out of memory',
      });

      render(<TestingPanel />);

      await waitFor(() => {
        const modelSelect = screen.getByRole('combobox');
        fireEvent.change(modelSelect, { target: { value: 'gpt2-small' } });
      });

      const loadButton = screen.getByRole('button', { name: /load/i });
      fireEvent.click(loadButton);

      await waitFor(() => {
        expect(screen.getByText(/Out of memory/i)).toBeInTheDocument();
      });
    });
  });

  describe('Single Generation Flow', () => {
    beforeEach(() => {
      (inferenceApi.status as jest.Mock).mockResolvedValue({
        success: true,
        data: { loaded: true, model: 'gpt2-small' },
      });
    });

    it('generates text from prompt', async () => {
      const mockResult: InferenceResult = {
        text: 'This is generated text.',
        tokens: 5,
        duration_ms: 150,
        tokens_per_second: 33.3,
      };

      (inferenceApi.generate as jest.Mock).mockResolvedValue({
        success: true,
        data: mockResult,
      });

      render(<TestingPanel />);

      // Wait for model status
      await waitFor(() => {
        expect(screen.getByText(/loaded/i)).toBeInTheDocument();
      });

      // Enter prompt
      const promptInput = screen.getByPlaceholderText(/Enter prompt/i);
      fireEvent.change(promptInput, { target: { value: 'Hello world' } });

      // Click generate
      const generateButton = screen.getByRole('button', { name: /^generate$/i });
      fireEvent.click(generateButton);

      await waitFor(() => {
        expect(inferenceApi.generate).toHaveBeenCalledWith(
          expect.objectContaining({ prompt: 'Hello world' })
        );
        expect(screen.getByText(/This is generated text/i)).toBeInTheDocument();
      });
    });

    it('displays generation metrics', async () => {
      const mockResult: InferenceResult = {
        text: 'Generated',
        tokens: 10,
        duration_ms: 300,
        tokens_per_second: 33.3,
      };

      (inferenceApi.generate as jest.Mock).mockResolvedValue({
        success: true,
        data: mockResult,
      });

      render(<TestingPanel />);

      await waitFor(() => {
        const generateButton = screen.getByRole('button', { name: /^generate$/i });
        fireEvent.click(generateButton);
      });

      await waitFor(() => {
        expect(screen.getByText(/33.3.*tokens.*second/i)).toBeInTheDocument();
        expect(screen.getByText(/300.*ms/i)).toBeInTheDocument();
      });
    });

    it('handles generation errors', async () => {
      (inferenceApi.generate as jest.Mock).mockResolvedValue({
        success: false,
        error: 'Generation failed',
      });

      render(<TestingPanel />);

      await waitFor(() => {
        const generateButton = screen.getByRole('button', { name: /^generate$/i });
        fireEvent.click(generateButton);
      });

      await waitFor(() => {
        expect(screen.getByText(/Generation failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('Loop Generation with Progress', () => {
    beforeEach(() => {
      (inferenceApi.status as jest.Mock).mockResolvedValue({
        success: true,
        data: { loaded: true, model: 'gpt2-small' },
      });
    });

    it('runs loop generation multiple times', async () => {
      (inferenceApi.generateLoop as jest.Mock).mockResolvedValue({
        success: true,
        data: {
          results: [
            { text: 'Result 1', tokens: 5, duration_ms: 100 },
            { text: 'Result 2', tokens: 5, duration_ms: 100 },
            { text: 'Result 3', tokens: 5, duration_ms: 100 },
          ],
          total_duration_ms: 300,
          avg_tokens_per_second: 50,
        },
      });

      render(<TestingPanel />);

      // Set loop count
      const loopInput = screen.getByLabelText(/Loop count/i);
      fireEvent.change(loopInput, { target: { value: '3' } });

      // Start loop generation
      const loopButton = screen.getByRole('button', { name: /loop/i });
      fireEvent.click(loopButton);

      await waitFor(() => {
        expect(inferenceApi.generateLoop).toHaveBeenCalledWith(
          expect.objectContaining({ count: 3 })
        );
      });
    });

    it('displays progress during loop generation', async () => {
      (inferenceApi.generateLoop as jest.Mock).mockImplementation(
        () =>
          new Promise((resolve) => {
            setTimeout(
              () =>
                resolve({
                  success: true,
                  data: { results: [], total_duration_ms: 0 },
                }),
              100
            );
          })
      );

      render(<TestingPanel />);

      const loopButton = screen.getByRole('button', { name: /loop/i });
      fireEvent.click(loopButton);

      expect(screen.getByText(/Generating/i)).toBeInTheDocument();
    });

    it('displays all loop results', async () => {
      (inferenceApi.generateLoop as jest.Mock).mockResolvedValue({
        success: true,
        data: {
          results: [
            { text: 'Result 1', tokens: 5, duration_ms: 100 },
            { text: 'Result 2', tokens: 6, duration_ms: 110 },
          ],
          total_duration_ms: 210,
        },
      });

      render(<TestingPanel />);

      const loopButton = screen.getByRole('button', { name: /loop/i });
      fireEvent.click(loopButton);

      await waitFor(() => {
        expect(screen.getByText(/Result 1/i)).toBeInTheDocument();
        expect(screen.getByText(/Result 2/i)).toBeInTheDocument();
      });
    });
  });

  describe('Batch Generation Results Display', () => {
    beforeEach(() => {
      (inferenceApi.status as jest.Mock).mockResolvedValue({
        success: true,
        data: { loaded: true, model: 'gpt2-small' },
      });
    });

    it('generates from multiple prompts', async () => {
      (inferenceApi.generateBatch as jest.Mock).mockResolvedValue({
        success: true,
        data: {
          results: [
            { prompt: 'Prompt 1', text: 'Response 1', tokens: 5 },
            { prompt: 'Prompt 2', text: 'Response 2', tokens: 6 },
          ],
          total_duration_ms: 300,
        },
      });

      render(<TestingPanel />);

      // Enter batch prompts
      const batchInput = screen.getByPlaceholderText(/Enter prompts/i);
      fireEvent.change(batchInput, {
        target: { value: 'Prompt 1\nPrompt 2' },
      });

      // Start batch generation
      const batchButton = screen.getByRole('button', { name: /batch/i });
      fireEvent.click(batchButton);

      await waitFor(() => {
        expect(inferenceApi.generateBatch).toHaveBeenCalled();
        expect(screen.getByText(/Response 1/i)).toBeInTheDocument();
        expect(screen.getByText(/Response 2/i)).toBeInTheDocument();
      });
    });

    it('displays batch generation statistics', async () => {
      (inferenceApi.generateBatch as jest.Mock).mockResolvedValue({
        success: true,
        data: {
          results: [
            { prompt: 'P1', text: 'R1', tokens: 5 },
            { prompt: 'P2', text: 'R2', tokens: 6 },
          ],
          total_duration_ms: 300,
          avg_tokens_per_second: 36.7,
        },
      });

      render(<TestingPanel />);

      const batchButton = screen.getByRole('button', { name: /batch/i });
      fireEvent.click(batchButton);

      await waitFor(() => {
        expect(screen.getByText(/36.7/i)).toBeInTheDocument();
        expect(screen.getByText(/300.*ms/i)).toBeInTheDocument();
      });
    });
  });

  describe('Export Functionality', () => {
    it('exports results to file', async () => {
      (inferenceApi.export as jest.Mock).mockResolvedValue({
        success: true,
        data: { filePath: '/exports/results.json' },
      });

      // Mock results in state
      (inferenceApi.generate as jest.Mock).mockResolvedValue({
        success: true,
        data: { text: 'Test result', tokens: 5, duration_ms: 100 },
      });

      render(<TestingPanel />);

      // Generate some results first
      await waitFor(() => {
        const generateButton = screen.getByRole('button', { name: /^generate$/i });
        fireEvent.click(generateButton);
      });

      await waitFor(() => {
        const exportButton = screen.getByRole('button', { name: /export/i });
        fireEvent.click(exportButton);
      });

      expect(inferenceApi.export).toHaveBeenCalled();
    });

    it('disables export when no results', () => {
      render(<TestingPanel />);

      const exportButton = screen.queryByRole('button', { name: /export/i });
      if (exportButton) {
        expect(exportButton).toBeDisabled();
      }
    });
  });

  describe('Generation Configuration', () => {
    it('allows adjusting max_tokens', async () => {
      render(<TestingPanel />);

      const maxTokensInput = screen.getByLabelText(/Max tokens/i);
      fireEvent.change(maxTokensInput, { target: { value: '100' } });

      const generateButton = screen.getByRole('button', { name: /^generate$/i });
      fireEvent.click(generateButton);

      await waitFor(() => {
        expect(inferenceApi.generate).toHaveBeenCalledWith(
          expect.objectContaining({ max_tokens: 100 })
        );
      });
    });

    it('allows adjusting temperature', async () => {
      render(<TestingPanel />);

      const tempInput = screen.getByLabelText(/Temperature/i);
      fireEvent.change(tempInput, { target: { value: '0.8' } });

      const generateButton = screen.getByRole('button', { name: /^generate$/i });
      fireEvent.click(generateButton);

      await waitFor(() => {
        expect(inferenceApi.generate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.8 })
        );
      });
    });

    it('validates configuration parameters', async () => {
      render(<TestingPanel />);

      const tempInput = screen.getByLabelText(/Temperature/i);
      fireEvent.change(tempInput, { target: { value: '2.0' } }); // Invalid

      await waitFor(() => {
        expect(screen.getByText(/Invalid temperature/i)).toBeInTheDocument();
      });
    });
  });
});
