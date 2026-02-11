"""
Unit tests for LayerProfiler class.

Tests cover:
- Hook registration on mock model
- Timing capture
- Activation statistics calculation
- Hook cleanup
- Context manager behavior
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import time
import sys
import os
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from profiling.layer_profiler import LayerProfiler, ComponentTiming
from profiling.model_detector import ComponentPaths


class MockModule(torch.nn.Module):
    """Mock PyTorch module for testing."""
    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, x):
        self.forward_count += 1
        return x


class TestLayerProfiler(unittest.TestCase):
    """Test LayerProfiler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model with nested structure
        self.model = Mock()
        self.model.config = Mock()
        self.model.config.num_hidden_layers = 2

        # Create mock layers with proper structure
        self.layers = []
        for i in range(2):
            layer = Mock()

            # Attention components
            layer.self_attn = Mock()
            layer.self_attn.q_proj = MockModule()
            layer.self_attn.k_proj = MockModule()
            layer.self_attn.v_proj = MockModule()
            layer.self_attn.o_proj = MockModule()

            # MLP components
            layer.mlp = Mock()
            layer.mlp.gate_proj = MockModule()
            layer.mlp.up_proj = MockModule()
            layer.mlp.down_proj = MockModule()

            # Layer norms
            layer.input_layernorm = MockModule()
            layer.post_attention_layernorm = MockModule()

            self.layers.append(layer)

        self.model.model = Mock()
        self.model.model.layers = self.layers

    def _create_component_paths(self) -> ComponentPaths:
        """Create mock ComponentPaths for testing."""
        return ComponentPaths(
            architecture="test",
            num_layers=2,
            layers_path="model.layers",
            q_proj="self_attn.q_proj",
            k_proj="self_attn.k_proj",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            gate_proj="mlp.gate_proj",
            up_proj="mlp.up_proj",
            down_proj="mlp.down_proj",
            input_layernorm="input_layernorm",
            post_attention_layernorm="post_attention_layernorm",
            norm_type="rmsnorm"
        )

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_initialization(self, mock_detector_class):
        """Test LayerProfiler initialization."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler
        profiler = LayerProfiler(
            self.model,
            capture_activations=True,
            sparsity_threshold=1e-5
        )

        # Verify initialization
        self.assertEqual(profiler.model, self.model)
        self.assertTrue(profiler.capture_activations)
        self.assertEqual(profiler.sparsity_threshold, 1e-5)
        self.assertEqual(profiler.component_paths.num_layers, 2)
        self.assertFalse(profiler._hooks_registered)
        self.assertEqual(len(profiler.hook_handles), 0)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_hook_registration(self, mock_detector_class):
        """Test hook registration on model components."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler and register hooks
        profiler = LayerProfiler(self.model, capture_activations=False)
        profiler.register_hooks()

        # Verify hooks registered flag
        self.assertTrue(profiler._hooks_registered)

        # Verify hooks registered on all components
        # Each component gets 2 hooks (pre + post)
        # 9 components per layer * 2 layers * 2 hooks = 36 hooks
        expected_hooks = 9 * 2 * 2  # 9 components, 2 layers, 2 hooks each
        self.assertEqual(len(profiler.hook_handles), expected_hooks)

        # Verify warning on double registration
        with self.assertLogs('backend.profiling.layer_profiler', level='WARNING') as cm:
            profiler.register_hooks()
            self.assertTrue(any('already registered' in msg for msg in cm.output))

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_timing_capture(self, mock_detector_class):
        """Test accurate timing capture during forward pass."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler without activation capture for cleaner test
        profiler = LayerProfiler(self.model, capture_activations=False)
        profiler.register_hooks()

        # Run forward pass on a component
        test_input = torch.randn(1, 10)
        layer = self.layers[0]
        q_proj = layer.self_attn.q_proj

        # Execute forward pass
        start = time.perf_counter()
        output = q_proj(test_input)
        end = time.perf_counter()
        actual_duration_ms = (end - start) * 1000.0

        # Get captured timings
        timings = profiler.get_timings()

        # Verify timing was captured
        self.assertGreater(len(timings), 0)

        # Find the q_proj timing for layer 0
        q_proj_timings = [t for t in timings if t.component_name == 'q_proj' and t.layer_idx == 0]
        self.assertEqual(len(q_proj_timings), 1)

        timing = q_proj_timings[0]
        self.assertEqual(timing.component_name, 'q_proj')
        self.assertEqual(timing.layer_idx, 0)
        self.assertGreater(timing.duration_ms, 0)
        # Duration should be somewhat close to actual (within 10ms for overhead)
        self.assertLess(abs(timing.duration_ms - actual_duration_ms), 10.0)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_activation_statistics_capture(self, mock_detector_class):
        """Test activation statistics calculation."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler with activation capture enabled
        profiler = LayerProfiler(
            self.model,
            capture_activations=True,
            sparsity_threshold=0.5
        )
        profiler.register_hooks()

        # Create a test tensor with known statistics
        # Half zeros (below threshold), half ones (above threshold)
        test_output = torch.cat([
            torch.zeros(1, 5),
            torch.ones(1, 5)
        ], dim=1)

        # Patch the component to return our test tensor
        layer = self.layers[0]
        original_forward = layer.self_attn.q_proj.forward

        def patched_forward(x):
            original_forward(x)
            return test_output

        layer.self_attn.q_proj.forward = patched_forward

        # Run forward pass
        test_input = torch.randn(1, 10)
        _ = layer.self_attn.q_proj(test_input)

        # Get timings
        timings = profiler.get_timings()
        q_proj_timings = [t for t in timings if t.component_name == 'q_proj' and t.layer_idx == 0]
        self.assertEqual(len(q_proj_timings), 1)

        timing = q_proj_timings[0]

        # Verify activation statistics
        self.assertIsNotNone(timing.activation_mean)
        self.assertIsNotNone(timing.activation_std)
        self.assertIsNotNone(timing.activation_max)
        self.assertIsNotNone(timing.activation_sparsity)

        # Verify values are reasonable
        # Mean of [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] is 0.5
        self.assertAlmostEqual(timing.activation_mean, 0.5, places=5)
        # Max is 1
        self.assertAlmostEqual(timing.activation_max, 1.0, places=5)
        # Sparsity: 5 values below 0.5 out of 10 = 0.5
        self.assertAlmostEqual(timing.activation_sparsity, 0.5, places=5)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_activation_capture_disabled(self, mock_detector_class):
        """Test that activation statistics are not captured when disabled."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler with activation capture disabled
        profiler = LayerProfiler(self.model, capture_activations=False)
        profiler.register_hooks()

        # Run forward pass
        test_input = torch.randn(1, 10)
        _ = self.layers[0].self_attn.q_proj(test_input)

        # Get timings
        timings = profiler.get_timings()
        q_proj_timings = [t for t in timings if t.component_name == 'q_proj' and t.layer_idx == 0]
        self.assertEqual(len(q_proj_timings), 1)

        timing = q_proj_timings[0]

        # Verify activation statistics are None
        self.assertIsNone(timing.activation_mean)
        self.assertIsNone(timing.activation_std)
        self.assertIsNone(timing.activation_max)
        self.assertIsNone(timing.activation_sparsity)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_reset_clears_timings(self, mock_detector_class):
        """Test that reset() clears captured timings."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler and register hooks
        profiler = LayerProfiler(self.model, capture_activations=False)
        profiler.register_hooks()

        # Run forward pass to capture timings
        test_input = torch.randn(1, 10)
        _ = self.layers[0].self_attn.q_proj(test_input)

        # Verify timings were captured
        timings = profiler.get_timings()
        self.assertGreater(len(timings), 0)

        # Reset
        profiler.reset()

        # Verify timings are cleared
        timings_after = profiler.get_timings()
        self.assertEqual(len(timings_after), 0)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_hook_cleanup(self, mock_detector_class):
        """Test that detach() properly removes hooks."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler and register hooks
        profiler = LayerProfiler(self.model, capture_activations=False)
        profiler.register_hooks()

        # Verify hooks exist
        initial_hook_count = len(profiler.hook_handles)
        self.assertGreater(initial_hook_count, 0)
        self.assertTrue(profiler._hooks_registered)

        # Detach hooks
        profiler.detach()

        # Verify hooks removed
        self.assertEqual(len(profiler.hook_handles), 0)
        self.assertFalse(profiler._hooks_registered)

        # Verify timings cleared
        timings = profiler.get_timings()
        self.assertEqual(len(timings), 0)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_context_manager_entry(self, mock_detector_class):
        """Test context manager __enter__ registers hooks."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler
        profiler = LayerProfiler(self.model, capture_activations=False)

        # Verify hooks not registered initially
        self.assertFalse(profiler._hooks_registered)

        # Enter context
        with profiler as p:
            # Verify hooks registered
            self.assertTrue(p._hooks_registered)
            self.assertGreater(len(p.hook_handles), 0)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_context_manager_exit(self, mock_detector_class):
        """Test context manager __exit__ removes hooks."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler
        profiler = LayerProfiler(self.model, capture_activations=False)

        # Use context manager
        with profiler as p:
            # Run some operations
            test_input = torch.randn(1, 10)
            _ = self.layers[0].self_attn.q_proj(test_input)

        # After exit, hooks should be removed
        self.assertFalse(profiler._hooks_registered)
        self.assertEqual(len(profiler.hook_handles), 0)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_context_manager_exception_cleanup(self, mock_detector_class):
        """Test that context manager cleans up even on exception."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler
        profiler = LayerProfiler(self.model, capture_activations=False)

        # Use context manager with exception
        with self.assertRaises(ValueError):
            with profiler as p:
                # Verify hooks registered
                self.assertTrue(p._hooks_registered)
                # Raise exception
                raise ValueError("Test exception")

        # After exception, hooks should still be cleaned up
        self.assertFalse(profiler._hooks_registered)
        self.assertEqual(len(profiler.hook_handles), 0)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_get_timings_returns_copy(self, mock_detector_class):
        """Test that get_timings() returns a copy, not reference."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler
        profiler = LayerProfiler(self.model, capture_activations=False)
        profiler.register_hooks()

        # Run forward pass
        test_input = torch.randn(1, 10)
        _ = self.layers[0].self_attn.q_proj(test_input)

        # Get timings twice
        timings1 = profiler.get_timings()
        timings2 = profiler.get_timings()

        # Verify they are equal but not the same object
        self.assertEqual(len(timings1), len(timings2))
        self.assertIsNot(timings1, timings2)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_multiple_layers_profiled(self, mock_detector_class):
        """Test that multiple layers are profiled correctly."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler
        profiler = LayerProfiler(self.model, capture_activations=False)
        profiler.register_hooks()

        # Run forward pass on components from both layers
        test_input = torch.randn(1, 10)
        _ = self.layers[0].self_attn.q_proj(test_input)
        _ = self.layers[1].self_attn.q_proj(test_input)

        # Get timings
        timings = profiler.get_timings()

        # Find timings for both layers
        layer0_timings = [t for t in timings if t.layer_idx == 0 and t.component_name == 'q_proj']
        layer1_timings = [t for t in timings if t.layer_idx == 1 and t.component_name == 'q_proj']

        self.assertEqual(len(layer0_timings), 1)
        self.assertEqual(len(layer1_timings), 1)

    @patch('backend.profiling.layer_profiler.ModelArchitectureDetector')
    def test_tuple_output_handling(self, mock_detector_class):
        """Test handling of tuple outputs from modules."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = self._create_component_paths()
        mock_detector_class.return_value = mock_detector

        # Create profiler with activation capture
        profiler = LayerProfiler(self.model, capture_activations=True)
        profiler.register_hooks()

        # Patch component to return tuple (simulating attention output)
        test_output = torch.ones(1, 10)
        layer = self.layers[0]
        original_forward = layer.self_attn.q_proj.forward

        def patched_forward(x):
            original_forward(x)
            return (test_output, None)  # Tuple output

        layer.self_attn.q_proj.forward = patched_forward

        # Run forward pass
        test_input = torch.randn(1, 10)
        _ = layer.self_attn.q_proj(test_input)

        # Get timings
        timings = profiler.get_timings()
        q_proj_timings = [t for t in timings if t.component_name == 'q_proj' and t.layer_idx == 0]

        # Should still capture statistics from first element of tuple
        self.assertEqual(len(q_proj_timings), 1)
        timing = q_proj_timings[0]
        self.assertIsNotNone(timing.activation_mean)
        self.assertAlmostEqual(timing.activation_mean, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
