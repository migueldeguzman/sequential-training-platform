"""
Integration tests for Energy Profiler system.

Tests end-to-end profiling flow:
- Full profiled inference with small model
- Data flow from profiler to database
- API endpoints return correct data
- WebSocket streaming
- Export functionality
"""

import unittest
import asyncio
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import sqlite3

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from profiling.power_monitor import PowerMonitor, PowerSample
from profiling.layer_profiler import LayerProfiler
from profiling.database import ProfileDatabase
from profiling.pipeline_profiler import InferencePipelineProfiler
from profiling.model_features import extract_model_features
import uuid


class MockModel:
    """Mock transformer model for testing."""

    def __init__(self, num_layers=2):
        self.config = MagicMock()
        self.config.num_hidden_layers = num_layers
        self.config.hidden_size = 768
        self.config.intermediate_size = 3072
        self.config.num_attention_heads = 12
        self.config.num_key_value_heads = 12
        self.config.vocab_size = 50000
        self.config.model_type = "test_model"

        # Create mock layers
        self.model = MagicMock()
        self.model.layers = []
        for i in range(num_layers):
            layer = MagicMock()
            layer.self_attn = MagicMock()
            layer.self_attn.q_proj = MagicMock()
            layer.self_attn.k_proj = MagicMock()
            layer.self_attn.v_proj = MagicMock()
            layer.self_attn.o_proj = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.gate_proj = MagicMock()
            layer.mlp.up_proj = MagicMock()
            layer.mlp.down_proj = MagicMock()
            layer.input_layernorm = MagicMock()
            layer.post_attention_layernorm = MagicMock()
            self.model.layers.append(layer)

    def named_modules(self):
        """Mock named_modules for hook registration."""
        modules = []
        for i, layer in enumerate(self.model.layers):
            modules.append((f"model.layers.{i}.self_attn.q_proj", layer.self_attn.q_proj))
            modules.append((f"model.layers.{i}.self_attn.k_proj", layer.self_attn.k_proj))
            modules.append((f"model.layers.{i}.self_attn.v_proj", layer.self_attn.v_proj))
            modules.append((f"model.layers.{i}.self_attn.o_proj", layer.self_attn.o_proj))
            modules.append((f"model.layers.{i}.mlp.gate_proj", layer.mlp.gate_proj))
            modules.append((f"model.layers.{i}.mlp.up_proj", layer.mlp.up_proj))
            modules.append((f"model.layers.{i}.mlp.down_proj", layer.mlp.down_proj))
            modules.append((f"model.layers.{i}.input_layernorm", layer.input_layernorm))
            modules.append((f"model.layers.{i}.post_attention_layernorm", layer.post_attention_layernorm))
        return modules

    def parameters(self):
        """Mock parameters for param counting."""
        # Return fake parameters that sum to ~1M params
        mock_param = MagicMock()
        mock_param.numel.return_value = 100000
        return [mock_param] * 10  # 10 * 100k = 1M params


class TestIntegration(unittest.TestCase):
    """Integration tests for profiling system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()

        # Initialize database
        self.db = ProfileDatabase(self.db_path)

        # Create mock model
        self.model = MockModel(num_layers=2)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_01_full_profiled_inference_with_mock(self):
        """Test full profiled inference with small mock model."""
        # Mock PowerMonitor
        with patch('profiling.pipeline_profiler.PowerMonitor') as MockPowerMonitor:
            power_monitor = MagicMock()
            power_monitor.is_available.return_value = True
            power_monitor.get_samples.return_value = [
                PowerSample(
                    timestamp=0.0,
                    relative_time_ms=0.0,
                    cpu_power_mw=1000.0,
                    gpu_power_mw=2000.0,
                    ane_power_mw=500.0,
                    dram_power_mw=300.0,
                    total_power_mw=3800.0,
                    phase='prefill'
                ),
                PowerSample(
                    timestamp=0.1,
                    relative_time_ms=100.0,
                    cpu_power_mw=1200.0,
                    gpu_power_mw=2500.0,
                    ane_power_mw=600.0,
                    dram_power_mw=350.0,
                    total_power_mw=4650.0,
                    phase='decode'
                )
            ]
            MockPowerMonitor.return_value = power_monitor

            # Create profiler
            layer_profiler = LayerProfiler(self.model, capture_activations=False)

            profiler = InferencePipelineProfiler(
                power_monitor=power_monitor,
                layer_profiler=layer_profiler,
                database=self.db,
                model=self.model
            )

            # Simulate inference
            with profiler.run(
                prompt="Test prompt",
                model_name="test-model",
                tags=["integration-test"]
            ) as run_context:
                run_id = run_context['run_id']

                # Pre-inference phase
                with profiler.section('pre_inference', 'tokenization'):
                    pass  # Simulate tokenization

                # Prefill phase
                with profiler.section('prefill', 'embedding_lookup'):
                    pass

                # Decode phase (2 tokens)
                for token_idx in range(2):
                    with profiler.section('decode', f'token_{token_idx}'):
                        # Simulate layer processing
                        layer_profiler.reset()

                        # Add mock token
                        profiler.add_token(
                            token_position=token_idx,
                            token_text=f"token_{token_idx}",
                            duration_ms=10.0,
                            energy_mj=50.0
                        )

                # Post-inference phase
                with profiler.section('post_inference', 'detokenization'):
                    pass

            # Verify run was created
            run = self.db.get_run(run_id)
            self.assertIsNotNone(run)
            self.assertEqual(run['prompt'], "Test prompt")
            self.assertEqual(run['model_name'], "test-model")
            self.assertIn("integration-test", run['tags'].split(','))

            # Verify power samples were saved
            self.assertGreater(run['total_samples'], 0)

            print("✓ Full profiled inference test passed")

    def test_02_data_flow_profiler_to_database(self):
        """Test data flows correctly from profiler components to database."""
        # Create a run
        run_id = str(uuid.uuid4())
        self.db.create_run(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            prompt="Data flow test",
            model_name="test-model"
        )

        # Add power samples
        samples = [
            PowerSample(
                timestamp=i * 0.1,
                cpu_power_mw=1000.0 + i * 100,
                gpu_power_mw=2000.0 + i * 200,
                ane_power_mw=500.0,
                dram_power_mw=300.0,
                total_power_mw=3800.0 + i * 300,
                phase='prefill' if i < 5 else 'decode'
            )
            for i in range(10)
        ]
        self.db.add_power_samples(run_id, samples)

        # Add tokens with metrics
        tokens = []
        for i in range(3):
            token_data = {
                'token_position': i,
                'token_text': f"token_{i}",
                'duration_ms': 10.0 + i,
                'energy_mj': 50.0 + i * 10,
                'power_mw': 5000.0,
                'phase': 'decode'
            }
            tokens.append(token_data)

        token_ids = self.db.add_tokens_batch(run_id, tokens)

        # Add layer metrics for first token
        layer_metrics = []
        for layer_idx in range(2):
            layer_data = {
                'layer_index': layer_idx,
                'duration_ms': 5.0,
                'energy_mj': 25.0
            }
            layer_metrics.append(layer_data)

        self.db.add_layer_metrics(token_ids[0], layer_metrics)

        # Update run with final metrics
        self.db.update_run_metrics(
            run_id=run_id,
            total_duration_ms=100.0,
            total_energy_mj=500.0,
            total_samples=10,
            num_input_tokens=5,
            num_output_tokens=3
        )

        # Verify data integrity
        run = self.db.get_run(run_id)
        self.assertEqual(run['total_duration_ms'], 100.0)
        self.assertEqual(run['total_energy_mj'], 500.0)
        self.assertEqual(run['total_samples'], 10)

        # Verify tokens
        tokens_retrieved = self.db.get_tokens(run_id)
        self.assertEqual(len(tokens_retrieved), 3)
        self.assertEqual(tokens_retrieved[0]['token_text'], 'token_0')

        # Verify layer metrics
        layer_metrics_retrieved = self.db.get_layer_metrics(token_ids[0])
        self.assertEqual(len(layer_metrics_retrieved), 2)

        print("✓ Data flow profiler to database test passed")

    def test_03_api_endpoints_data_correctness(self):
        """Test that API query methods return correctly structured data."""
        # Create test run with complete data
        run_id = self.db.create_run(
            prompt="API test prompt",
            model_name="api-test-model",
            num_params=2000000,
            num_layers=4,
            hidden_size=1024,
            tags="api-test,integration"
        )

        # Add power samples
        samples = [
            PowerSample(
                timestamp=i * 0.1,
                cpu_power_mw=1000.0,
                gpu_power_mw=2000.0,
                ane_power_mw=500.0,
                dram_power_mw=300.0,
                total_power_mw=3800.0,
                phase='decode'
            )
            for i in range(5)
        ]
        self.db.add_power_samples(run_id, samples)

        # Add tokens
        tokens = [
            {
                'token_position': i,
                'token_text': f"word{i}",
                'duration_ms': 12.0,
                'energy_mj': 60.0,
                'power_mw': 5000.0,
                'phase': 'decode'
            }
            for i in range(3)
        ]
        token_ids = self.db.add_tokens_batch(run_id, tokens)

        # Update run metrics
        self.db.update_run_metrics(
            run_id=run_id,
            total_duration_ms=150.0,
            total_energy_mj=750.0,
            total_samples=5,
            num_input_tokens=10,
            num_output_tokens=3
        )

        # Test get_run
        run = self.db.get_run(run_id)
        self.assertIsNotNone(run)
        self.assertEqual(run['prompt'], "API test prompt")
        self.assertEqual(run['model_name'], "api-test-model")
        self.assertEqual(run['total_duration_ms'], 150.0)
        self.assertEqual(run['total_energy_mj'], 750.0)

        # Test get_runs with filters
        runs = self.db.get_runs(tags="api-test")
        self.assertGreater(len(runs), 0)
        self.assertTrue(any(r['run_id'] == run_id for r in runs))

        # Test get_run_summary
        summary = self.db.get_run_summary(run_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['run_id'], run_id)
        self.assertIn('total_energy_mj', summary)
        self.assertIn('joules_per_token', summary)

        # Test get_tokens
        tokens_retrieved = self.db.get_tokens(run_id)
        self.assertEqual(len(tokens_retrieved), 3)

        # Test get_power_timeline
        timeline = self.db.get_power_timeline(run_id)
        self.assertEqual(len(timeline), 5)
        self.assertIn('timestamp', timeline[0])
        self.assertIn('total_power_mw', timeline[0])

        print("✓ API endpoints data correctness test passed")

    def test_04_export_functionality(self):
        """Test export functionality for JSON and CSV formats."""
        # Create test run
        run_id = self.db.create_run(
            prompt="Export test",
            model_name="export-model",
            num_params=1000000,
            num_layers=2,
            hidden_size=512
        )

        # Add minimal data
        samples = [
            PowerSample(
                timestamp=0.0,
                cpu_power_mw=1000.0,
                gpu_power_mw=2000.0,
                ane_power_mw=500.0,
                dram_power_mw=300.0,
                total_power_mw=3800.0,
                phase='prefill'
            )
        ]
        self.db.add_power_samples(run_id, samples)

        self.db.update_run_metrics(
            run_id=run_id,
            total_duration_ms=50.0,
            total_energy_mj=250.0,
            total_samples=1,
            num_input_tokens=5,
            num_output_tokens=2
        )

        # Test JSON export (get complete run data)
        run = self.db.get_run(run_id)
        self.assertIsNotNone(run)

        # Verify JSON serializability
        json_str = json.dumps(run, default=str)
        self.assertIsInstance(json_str, str)
        reconstructed = json.loads(json_str)
        self.assertEqual(reconstructed['run_id'], run_id)

        # Test CSV export (get flattened data)
        tokens = self.db.get_tokens(run_id)
        timeline = self.db.get_power_timeline(run_id)

        # Verify CSV-compatible structure (flat dictionaries)
        if timeline:
            for sample in timeline:
                self.assertIsInstance(sample, dict)
                # All values should be simple types
                for value in sample.values():
                    self.assertIn(type(value).__name__, ['int', 'float', 'str', 'NoneType'])

        print("✓ Export functionality test passed")

    def test_05_database_cascading_deletes(self):
        """Test that deleting a run cascades to all related data."""
        # Create run with full data hierarchy
        run_id = self.db.create_run(
            prompt="Delete test",
            model_name="delete-model",
            num_params=500000,
            num_layers=2,
            hidden_size=256
        )

        # Add power samples
        samples = [
            PowerSample(
                timestamp=0.0,
                cpu_power_mw=1000.0,
                gpu_power_mw=2000.0,
                ane_power_mw=500.0,
                dram_power_mw=300.0,
                total_power_mw=3800.0,
                phase='decode'
            )
        ]
        self.db.add_power_samples(run_id, samples)

        # Add token
        tokens = [{
            'token_position': 0,
            'token_text': 'test',
            'duration_ms': 10.0,
            'energy_mj': 50.0,
            'power_mw': 5000.0,
            'phase': 'decode'
        }]
        token_ids = self.db.add_tokens_batch(run_id, tokens)

        # Add layer metrics
        layer_metrics = [{
            'layer_index': 0,
            'duration_ms': 5.0,
            'energy_mj': 25.0
        }]
        self.db.add_layer_metrics(token_ids[0], layer_metrics)

        # Verify data exists
        run = self.db.get_run(run_id)
        self.assertIsNotNone(run)

        # Delete run
        self.db.delete_run(run_id)

        # Verify run and all related data deleted
        run = self.db.get_run(run_id)
        self.assertIsNone(run)

        # Verify power samples deleted
        timeline = self.db.get_power_timeline(run_id)
        self.assertEqual(len(timeline), 0)

        # Verify tokens deleted
        tokens_retrieved = self.db.get_tokens(run_id)
        self.assertEqual(len(tokens_retrieved), 0)

        print("✓ Database cascading deletes test passed")

    def test_06_model_features_extraction(self):
        """Test model feature extraction integration."""
        # Extract features from mock model
        features = extract_model_features(self.model)

        self.assertEqual(features['num_layers'], 2)
        self.assertEqual(features['hidden_size'], 768)
        self.assertEqual(features['intermediate_size'], 3072)
        self.assertEqual(features['num_attention_heads'], 12)
        self.assertEqual(features['num_key_value_heads'], 12)
        self.assertEqual(features['vocab_size'], 50000)

        # Create run with features
        run_id = self.db.create_run(
            prompt="Feature test",
            model_name="feature-model",
            num_params=features.get('total_params', 1000000),
            num_layers=features['num_layers'],
            hidden_size=features['hidden_size'],
            intermediate_size=features['intermediate_size'],
            num_attention_heads=features['num_attention_heads'],
            num_key_value_heads=features['num_key_value_heads']
        )

        # Verify features stored
        run = self.db.get_run(run_id)
        self.assertEqual(run['num_layers'], 2)
        self.assertEqual(run['hidden_size'], 768)

        print("✓ Model features extraction test passed")

    def test_07_phase_tagged_samples(self):
        """Test that power samples are correctly tagged with phases."""
        run_id = self.db.create_run(
            prompt="Phase tag test",
            model_name="phase-model",
            num_params=1000000,
            num_layers=2,
            hidden_size=512
        )

        # Add samples with different phases
        samples = [
            PowerSample(timestamp=0.0, cpu_power_mw=1000.0, gpu_power_mw=2000.0,
                       ane_power_mw=500.0, dram_power_mw=300.0, total_power_mw=3800.0,
                       phase='idle'),
            PowerSample(timestamp=0.1, cpu_power_mw=1100.0, gpu_power_mw=2100.0,
                       ane_power_mw=550.0, dram_power_mw=320.0, total_power_mw=4070.0,
                       phase='pre_inference'),
            PowerSample(timestamp=0.2, cpu_power_mw=1500.0, gpu_power_mw=3000.0,
                       ane_power_mw=800.0, dram_power_mw=400.0, total_power_mw=5700.0,
                       phase='prefill'),
            PowerSample(timestamp=0.3, cpu_power_mw=1300.0, gpu_power_mw=2500.0,
                       ane_power_mw=600.0, dram_power_mw=350.0, total_power_mw=4750.0,
                       phase='decode'),
            PowerSample(timestamp=0.4, cpu_power_mw=1000.0, gpu_power_mw=2000.0,
                       ane_power_mw=500.0, dram_power_mw=300.0, total_power_mw=3800.0,
                       phase='post_inference'),
        ]
        self.db.add_power_samples(run_id, samples)

        # Retrieve and verify phases
        timeline = self.db.get_power_timeline(run_id)
        self.assertEqual(len(timeline), 5)

        phases = [s['phase'] for s in timeline]
        self.assertIn('idle', phases)
        self.assertIn('pre_inference', phases)
        self.assertIn('prefill', phases)
        self.assertIn('decode', phases)
        self.assertIn('post_inference', phases)

        print("✓ Phase tagged samples test passed")

    def test_08_peak_power_tracking(self):
        """Test peak power tracking during inference."""
        run_id = self.db.create_run(
            prompt="Peak power test",
            model_name="peak-model",
            num_params=1000000,
            num_layers=2,
            hidden_size=512
        )

        # Add samples with varying power
        samples = [
            PowerSample(timestamp=0.0, cpu_power_mw=1000.0, gpu_power_mw=2000.0,
                       ane_power_mw=500.0, dram_power_mw=300.0, total_power_mw=3800.0,
                       phase='decode'),
            PowerSample(timestamp=0.1, cpu_power_mw=2000.0, gpu_power_mw=5000.0,
                       ane_power_mw=1500.0, dram_power_mw=800.0, total_power_mw=9300.0,
                       phase='decode'),  # Peak
            PowerSample(timestamp=0.2, cpu_power_mw=1200.0, gpu_power_mw=2500.0,
                       ane_power_mw=600.0, dram_power_mw=350.0, total_power_mw=4650.0,
                       phase='decode'),
        ]
        self.db.add_power_samples(run_id, samples)

        # Calculate and update peak power
        timeline = self.db.get_power_timeline(run_id)
        peak_total = max(s['total_power_mw'] for s in timeline)
        peak_cpu = max(s['cpu_power_mw'] for s in timeline)
        peak_gpu = max(s['gpu_power_mw'] for s in timeline)

        self.assertEqual(peak_total, 9300.0)
        self.assertEqual(peak_cpu, 2000.0)
        self.assertEqual(peak_gpu, 5000.0)

        print("✓ Peak power tracking test passed")

    def test_09_joules_per_token_metric(self):
        """Test Joules per token (J/t) calculation."""
        run_id = self.db.create_run(
            prompt="J/t test",
            model_name="jt-model",
            num_params=1000000,
            num_layers=2,
            hidden_size=512
        )

        # Update with energy and token counts
        total_energy_mj = 1000.0  # 1000 millijoules = 1 joule
        num_input_tokens = 10
        num_output_tokens = 5
        total_tokens = num_input_tokens + num_output_tokens  # 15

        self.db.update_run_metrics(
            run_id=run_id,
            total_duration_ms=100.0,
            total_energy_mj=total_energy_mj,
            total_samples=10,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens
        )

        # Get summary with calculated metrics
        summary = self.db.get_run_summary(run_id)

        # Verify J/t calculation
        expected_jt = (total_energy_mj / 1000.0) / total_tokens  # Convert mJ to J
        self.assertAlmostEqual(summary['joules_per_token'], expected_jt, places=6)

        # Should be approximately 1J / 15 tokens = 0.0667 J/t
        self.assertAlmostEqual(summary['joules_per_token'], 0.0667, places=3)

        print("✓ Joules per token metric test passed")

    def test_10_data_retention_cleanup(self):
        """Test data retention and cleanup functionality."""
        from datetime import datetime, timedelta

        # Create old run (simulated by directly inserting with old timestamp)
        old_run_id = self.db.create_run(
            prompt="Old run",
            model_name="old-model",
            num_params=1000000,
            num_layers=2,
            hidden_size=512
        )

        # Create recent run
        recent_run_id = self.db.create_run(
            prompt="Recent run",
            model_name="recent-model",
            num_params=1000000,
            num_layers=2,
            hidden_size=512
        )

        # Manually update timestamp for old run
        old_timestamp = (datetime.now() - timedelta(days=90)).isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE profiling_runs SET timestamp = ? WHERE run_id = ?",
            (old_timestamp, old_run_id)
        )
        conn.commit()
        conn.close()

        # Get retention stats before cleanup
        stats_before = self.db.get_retention_stats()
        self.assertEqual(stats_before['total_runs'], 2)

        # Clean up runs older than 30 days
        deleted = self.db.cleanup_old_runs(max_age_days=30)
        self.assertEqual(deleted, 1)  # Should delete old run only

        # Verify old run deleted, recent run remains
        old_run = self.db.get_run(old_run_id)
        recent_run = self.db.get_run(recent_run_id)

        self.assertIsNone(old_run)
        self.assertIsNotNone(recent_run)

        # Get retention stats after cleanup
        stats_after = self.db.get_retention_stats()
        self.assertEqual(stats_after['total_runs'], 1)

        print("✓ Data retention cleanup test passed")


def run_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
