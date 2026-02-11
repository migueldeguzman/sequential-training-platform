"""
Unit tests for ProfileDatabase class

Tests schema creation, CRUD operations, query methods,
batch inserts, and cascading deletes.
"""

import unittest
import tempfile
import os
import sys
import sqlite3
import time
import uuid
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from profiling.database import ProfileDatabase, init_database


class TestProfileDatabase(unittest.TestCase):
    """Test suite for ProfileDatabase class"""

    def setUp(self):
        """Create a temporary database for each test"""
        # Create temporary file for test database
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Initialize database
        self.db = ProfileDatabase(self.db_path)
        self.db.connect()

        # Enable foreign key constraints (required for cascade delete and FK checks)
        self.db.conn.execute("PRAGMA foreign_keys = ON")

    def tearDown(self):
        """Clean up temporary database after each test"""
        self.db.close()
        # Remove temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_schema_creation(self):
        """Test that all tables and indexes are created correctly"""
        cursor = self.db.conn.cursor()

        # Check that all tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            'profiling_runs',
            'power_samples',
            'pipeline_sections',
            'tokens',
            'layer_metrics',
            'component_metrics',
            'deep_operation_metrics'
        ]

        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} should exist")

        # Check that indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]

        # Check some key indexes
        key_indexes = [
            'idx_power_samples_run_id',
            'idx_pipeline_sections_run_id',
            'idx_tokens_run_id',
            'idx_layer_metrics_token_id',
            'idx_component_metrics_layer_metric_id'
        ]

        for index in key_indexes:
            self.assertIn(index, indexes, f"Index {index} should exist")

    def test_create_run(self):
        """Test creating a profiling run record"""
        run_id = "test_run_001"
        timestamp = datetime.now().isoformat()
        model_name = "test-model-7b"
        prompt = "What is the meaning of life?"

        row_id = self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name=model_name,
            prompt=prompt,
            response="42",
            experiment_name="unit_test",
            tags="test,philosophy",
            profiling_depth="module"
        )

        self.assertIsNotNone(row_id)
        self.assertGreater(row_id, 0)

        # Verify the run was created
        run = self.db.get_run(run_id)
        self.assertIsNotNone(run)
        self.assertEqual(run['run_id'], run_id)
        self.assertEqual(run['model_name'], model_name)
        self.assertEqual(run['prompt'], prompt)
        self.assertEqual(run['response'], "42")
        self.assertEqual(run['experiment_name'], "unit_test")
        self.assertEqual(run['tags'], "test,philosophy")
        self.assertEqual(run['profiling_depth'], "module")
        self.assertEqual(run['status'], "running")

    def test_update_run_metrics(self):
        """Test updating run with final metrics"""
        run_id = "test_run_002"
        timestamp = datetime.now().isoformat()

        # Create initial run
        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Update with metrics
        self.db.update_run_metrics(
            run_id=run_id,
            total_duration_ms=1500.5,
            total_energy_mj=2500.3,
            token_count=50,
            tokens_per_second=33.33,
            input_token_count=10,
            output_token_count=40,
            prefill_energy_mj=500.1,
            decode_energy_mj=2000.2,
            energy_per_input_token_mj=50.01,
            energy_per_output_token_mj=50.005,
            input_output_energy_ratio=11.0,
            peak_power_mw=15000.5,
            peak_power_cpu_mw=3000.1,
            peak_power_gpu_mw=10000.2,
            peak_power_ane_mw=1500.1,
            peak_power_dram_mw=500.1,
            peak_power_timestamp_ms=750.0,
            baseline_power_mw=2000.5,
            baseline_cpu_power_mw=500.1,
            baseline_gpu_power_mw=1000.2,
            baseline_ane_power_mw=300.1,
            baseline_dram_power_mw=200.1,
            baseline_sample_count=20,
            status="completed"
        )

        # Verify updates
        run = self.db.get_run(run_id)
        self.assertEqual(run['total_duration_ms'], 1500.5)
        self.assertEqual(run['total_energy_mj'], 2500.3)
        self.assertEqual(run['token_count'], 50)
        self.assertEqual(run['tokens_per_second'], 33.33)
        self.assertEqual(run['input_token_count'], 10)
        self.assertEqual(run['output_token_count'], 40)
        self.assertEqual(run['prefill_energy_mj'], 500.1)
        self.assertEqual(run['decode_energy_mj'], 2000.2)
        self.assertEqual(run['peak_power_mw'], 15000.5)
        self.assertEqual(run['baseline_power_mw'], 2000.5)
        self.assertEqual(run['status'], "completed")

    def test_add_power_samples_batch(self):
        """Test batch inserting power samples"""
        run_id = "test_run_003"
        timestamp = datetime.now().isoformat()

        # Create run first
        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Create sample power samples
        samples = [
            {
                "timestamp_ms": 0.0,
                "cpu_power_mw": 1000.0,
                "gpu_power_mw": 5000.0,
                "ane_power_mw": 500.0,
                "dram_power_mw": 300.0,
                "total_power_mw": 6800.0,
                "phase": "idle"
            },
            {
                "timestamp_ms": 100.0,
                "cpu_power_mw": 2000.0,
                "gpu_power_mw": 8000.0,
                "ane_power_mw": 1000.0,
                "dram_power_mw": 500.0,
                "total_power_mw": 11500.0,
                "phase": "prefill"
            },
            {
                "timestamp_ms": 200.0,
                "cpu_power_mw": 1500.0,
                "gpu_power_mw": 7000.0,
                "ane_power_mw": 800.0,
                "dram_power_mw": 400.0,
                "total_power_mw": 9700.0,
                "phase": "decode"
            }
        ]

        # Insert samples
        self.db.add_power_samples(run_id, samples)

        # Verify samples were inserted
        timeline = self.db.get_power_timeline(run_id)
        self.assertEqual(len(timeline), 3)

        # Check first sample
        self.assertEqual(timeline[0]['timestamp_ms'], 0.0)
        self.assertEqual(timeline[0]['cpu_power_mw'], 1000.0)
        self.assertEqual(timeline[0]['phase'], "idle")

        # Check second sample
        self.assertEqual(timeline[1]['timestamp_ms'], 100.0)
        self.assertEqual(timeline[1]['total_power_mw'], 11500.0)
        self.assertEqual(timeline[1]['phase'], "prefill")

    def test_add_power_samples_foreign_key_constraint(self):
        """Test that adding samples for non-existent run fails"""
        samples = [{
            "timestamp_ms": 0.0,
            "cpu_power_mw": 1000.0,
            "gpu_power_mw": 5000.0,
            "ane_power_mw": 500.0,
            "dram_power_mw": 300.0,
            "total_power_mw": 6800.0,
            "phase": "idle"
        }]

        with self.assertRaises(sqlite3.IntegrityError):
            self.db.add_power_samples("nonexistent_run", samples)

    def test_add_pipeline_sections(self):
        """Test adding pipeline section records"""
        run_id = "test_run_004"
        timestamp = datetime.now().isoformat()

        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Add individual section
        section_id = self.db.add_pipeline_section(
            run_id=run_id,
            phase="pre_inference",
            section_name="tokenization",
            start_time_ms=0.0,
            end_time_ms=10.0,
            duration_ms=10.0,
            energy_mj=50.0,
            avg_power_mw=5000.0
        )

        self.assertIsNotNone(section_id)
        self.assertGreater(section_id, 0)

        # Add batch sections
        sections = [
            {
                "phase": "prefill",
                "section_name": "embedding_lookup",
                "start_time_ms": 10.0,
                "end_time_ms": 20.0,
                "duration_ms": 10.0,
                "energy_mj": 100.0,
                "avg_power_mw": 10000.0
            },
            {
                "phase": "prefill",
                "section_name": "layers",
                "start_time_ms": 20.0,
                "end_time_ms": 100.0,
                "duration_ms": 80.0,
                "energy_mj": 800.0,
                "avg_power_mw": 10000.0
            }
        ]

        self.db.add_pipeline_sections_batch(run_id, sections)
        self.db.commit_transaction()

        # Verify sections
        all_sections = self.db.get_pipeline_sections(run_id)
        self.assertEqual(len(all_sections), 3)
        self.assertEqual(all_sections[0]['section_name'], "tokenization")
        self.assertEqual(all_sections[1]['section_name'], "embedding_lookup")

    def test_add_tokens_batch(self):
        """Test batch inserting tokens"""
        run_id = "test_run_005"
        timestamp = datetime.now().isoformat()

        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Create tokens
        tokens = [
            {
                "token_index": 0,
                "token_text": "Hello",
                "phase": "prefill",
                "start_time_ms": 0.0,
                "end_time_ms": 10.0,
                "duration_ms": 10.0,
                "energy_mj": 50.0,
                "avg_power_mw": 5000.0,
                "is_input_token": True
            },
            {
                "token_index": 1,
                "token_text": "world",
                "phase": "decode",
                "start_time_ms": 10.0,
                "end_time_ms": 20.0,
                "duration_ms": 10.0,
                "energy_mj": 60.0,
                "avg_power_mw": 6000.0,
                "is_input_token": False
            }
        ]

        token_ids = self.db.add_tokens_batch(run_id, tokens)
        self.db.commit_transaction()

        self.assertEqual(len(token_ids), 2)
        self.assertGreater(token_ids[0], 0)
        self.assertGreater(token_ids[1], 0)

        # Verify tokens
        retrieved_tokens = self.db.get_tokens(run_id)
        self.assertEqual(len(retrieved_tokens), 2)
        self.assertEqual(retrieved_tokens[0]['token_text'], "Hello")
        self.assertEqual(retrieved_tokens[0]['is_input_token'], 1)  # SQLite stores bool as int
        self.assertEqual(retrieved_tokens[1]['token_text'], "world")
        self.assertEqual(retrieved_tokens[1]['is_input_token'], 0)

    def test_add_layer_metrics_batch(self):
        """Test batch inserting layer metrics"""
        run_id = "test_run_006"
        timestamp = datetime.now().isoformat()

        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Add token first
        token_id = self.db.add_token(
            run_id=run_id,
            token_index=0,
            token_text="test",
            phase="decode",
            start_time_ms=0.0,
            end_time_ms=100.0,
            duration_ms=100.0
        )

        # Add layer metrics
        layer_metrics = [
            {
                "layer_index": 0,
                "duration_ms": 5.0,
                "energy_mj": 50.0,
                "avg_power_mw": 10000.0
            },
            {
                "layer_index": 1,
                "duration_ms": 6.0,
                "energy_mj": 60.0,
                "avg_power_mw": 10000.0
            },
            {
                "layer_index": 2,
                "duration_ms": 5.5,
                "energy_mj": 55.0,
                "avg_power_mw": 10000.0
            }
        ]

        self.db.add_layer_metrics(token_id, layer_metrics)
        self.db.commit_transaction()

        # Verify layer metrics
        retrieved_metrics = self.db.get_layer_metrics(token_id)
        self.assertEqual(len(retrieved_metrics), 3)
        self.assertEqual(retrieved_metrics[0]['layer_index'], 0)
        self.assertEqual(retrieved_metrics[0]['duration_ms'], 5.0)
        self.assertEqual(retrieved_metrics[1]['layer_index'], 1)

    def test_add_component_metrics_batch(self):
        """Test batch inserting component metrics"""
        run_id = "test_run_007"
        timestamp = datetime.now().isoformat()

        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Add token and layer metric
        token_id = self.db.add_token(
            run_id=run_id,
            token_index=0,
            token_text="test",
            phase="decode",
            start_time_ms=0.0,
            end_time_ms=100.0,
            duration_ms=100.0
        )

        layer_metrics = [{
            "layer_index": 0,
            "duration_ms": 10.0,
            "energy_mj": 100.0,
            "avg_power_mw": 10000.0
        }]

        self.db.add_layer_metrics(token_id, layer_metrics)
        self.db.commit_transaction()

        layer_metric_id = self.db.get_layer_metrics(token_id)[0]['id']

        # Add component metrics
        component_metrics = [
            {
                "component_name": "q_proj",
                "duration_ms": 2.0,
                "energy_mj": 20.0,
                "avg_power_mw": 10000.0,
                "activation_mean": 0.5,
                "activation_std": 0.25,
                "activation_max": 2.0,
                "activation_sparsity": 0.1
            },
            {
                "component_name": "k_proj",
                "duration_ms": 2.0,
                "energy_mj": 20.0,
                "avg_power_mw": 10000.0,
                "activation_mean": 0.6,
                "activation_std": 0.3,
                "activation_max": 2.5,
                "activation_sparsity": 0.15
            }
        ]

        self.db.add_component_metrics(layer_metric_id, component_metrics)
        self.db.commit_transaction()

        # Verify component metrics
        retrieved_metrics = self.db.get_component_metrics(layer_metric_id)
        self.assertEqual(len(retrieved_metrics), 2)
        self.assertEqual(retrieved_metrics[0]['component_name'], "k_proj")  # Ordered by name
        self.assertEqual(retrieved_metrics[0]['activation_mean'], 0.6)
        self.assertEqual(retrieved_metrics[1]['component_name'], "q_proj")

    def test_add_deep_operation_metrics_batch(self):
        """Test batch inserting deep operation metrics"""
        run_id = "test_run_008"
        timestamp = datetime.now().isoformat()

        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Create full hierarchy: run -> token -> layer -> component
        token_id = self.db.add_token(
            run_id=run_id,
            token_index=0,
            token_text="test",
            phase="decode",
            start_time_ms=0.0,
            end_time_ms=100.0,
            duration_ms=100.0
        )

        layer_metrics = [{
            "layer_index": 0,
            "duration_ms": 10.0,
            "energy_mj": 100.0,
            "avg_power_mw": 10000.0
        }]
        self.db.add_layer_metrics(token_id, layer_metrics)
        self.db.commit_transaction()

        layer_metric_id = self.db.get_layer_metrics(token_id)[0]['id']

        component_metrics = [{
            "component_name": "attention",
            "duration_ms": 5.0,
            "energy_mj": 50.0,
            "avg_power_mw": 10000.0
        }]
        self.db.add_component_metrics(layer_metric_id, component_metrics)
        self.db.commit_transaction()

        component_metric_id = self.db.get_component_metrics(layer_metric_id)[0]['id']

        # Add deep operation metrics
        deep_metrics = [
            {
                "operation_name": "qk_matmul",
                "duration_ms": 1.0,
                "energy_mj": 10.0,
                "avg_power_mw": 10000.0,
                "attention_entropy": 2.5,
                "attention_max_weight": 0.8,
                "attention_sparsity": 0.2
            },
            {
                "operation_name": "softmax",
                "duration_ms": 0.5,
                "energy_mj": 5.0,
                "avg_power_mw": 10000.0,
                "attention_entropy": 2.3,
                "attention_max_weight": 0.75,
                "attention_sparsity": 0.25
            }
        ]

        self.db.add_deep_operation_metrics(component_metric_id, deep_metrics)
        self.db.commit_transaction()

        # Verify deep metrics (need to query directly as no getter method exists)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT * FROM deep_operation_metrics WHERE component_metric_id = ?",
            (component_metric_id,)
        )
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 2)

    def test_get_runs_with_filters(self):
        """Test querying runs with various filters"""
        # Create multiple runs
        base_time = datetime.now()

        runs = [
            {
                "run_id": "run_001",
                "timestamp": base_time.isoformat(),
                "model_name": "model-7b",
                "prompt": "test prompt 1",
                "tags": "tag1,tag2"
            },
            {
                "run_id": "run_002",
                "timestamp": (base_time + timedelta(hours=1)).isoformat(),
                "model_name": "model-13b",
                "prompt": "test prompt 2",
                "tags": "tag2,tag3"
            },
            {
                "run_id": "run_003",
                "timestamp": (base_time + timedelta(hours=2)).isoformat(),
                "model_name": "model-7b",
                "prompt": "test prompt 3",
                "tags": "tag3,tag4",
                "experiment_name": "experiment_1"
            }
        ]

        for run in runs:
            self.db.create_run(**run)

        # Test: Get all runs
        all_runs = self.db.get_runs()
        self.assertEqual(len(all_runs), 3)

        # Test: Filter by model
        model_runs = self.db.get_runs(model="model-7b")
        self.assertEqual(len(model_runs), 2)
        self.assertTrue(all(r['model_name'] == "model-7b" for r in model_runs))

        # Test: Filter by tags
        tag_runs = self.db.get_runs(tags="tag2")
        self.assertEqual(len(tag_runs), 2)

        # Test: Filter by experiment
        exp_runs = self.db.get_runs(experiment="experiment_1")
        self.assertEqual(len(exp_runs), 1)
        self.assertEqual(exp_runs[0]['run_id'], "run_003")

        # Test: Filter by date range
        date_runs = self.db.get_runs(
            date_from=base_time.isoformat(),
            date_to=(base_time + timedelta(hours=1, minutes=30)).isoformat()
        )
        self.assertEqual(len(date_runs), 2)

        # Test: Pagination
        page1 = self.db.get_runs(limit=2, offset=0)
        self.assertEqual(len(page1), 2)
        page2 = self.db.get_runs(limit=2, offset=2)
        self.assertEqual(len(page2), 1)

    def test_get_run_summary(self):
        """Test getting aggregated run summary"""
        run_id = "test_run_009"
        timestamp = datetime.now().isoformat()

        # Create run with full data
        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Update with metrics
        self.db.update_run_metrics(
            run_id=run_id,
            total_duration_ms=1000.0,
            total_energy_mj=5000.0,
            token_count=10,
            input_token_count=5,
            output_token_count=5,
            prefill_energy_mj=1000.0,
            decode_energy_mj=4000.0
        )

        # Add power samples with phases
        samples = [
            {
                "timestamp_ms": 0.0,
                "cpu_power_mw": 2000.0,
                "gpu_power_mw": 8000.0,
                "ane_power_mw": 1000.0,
                "dram_power_mw": 500.0,
                "total_power_mw": 11500.0,
                "phase": "idle"
            },
            {
                "timestamp_ms": 100.0,
                "cpu_power_mw": 3000.0,
                "gpu_power_mw": 10000.0,
                "ane_power_mw": 1500.0,
                "dram_power_mw": 700.0,
                "total_power_mw": 15200.0,
                "phase": "prefill"
            },
            {
                "timestamp_ms": 200.0,
                "cpu_power_mw": 2500.0,
                "gpu_power_mw": 9000.0,
                "ane_power_mw": 1200.0,
                "dram_power_mw": 600.0,
                "total_power_mw": 13300.0,
                "phase": "decode"
            }
        ]
        self.db.add_power_samples(run_id, samples)

        # Get summary
        summary = self.db.get_run_summary(run_id)

        self.assertIsNotNone(summary)
        self.assertEqual(summary['run_id'], run_id)
        self.assertEqual(summary['token_count'], 10)

        # Check efficiency metrics are included
        self.assertIn('efficiency_metrics', summary)
        self.assertIn('joules_per_token', summary['efficiency_metrics'])
        self.assertIn('tokens_per_joule', summary['efficiency_metrics'])

        # Check component breakdown is included
        self.assertIn('component_energy_breakdown', summary)
        self.assertIn('phase_power_breakdown', summary)

    def test_search_by_prompt(self):
        """Test searching runs by prompt text"""
        # Create runs with different prompts
        runs = [
            {"run_id": "run_001", "prompt": "What is machine learning?"},
            {"run_id": "run_002", "prompt": "Explain neural networks"},
            {"run_id": "run_003", "prompt": "How does machine learning work?"}
        ]

        for run in runs:
            self.db.create_run(
                run_id=run["run_id"],
                timestamp=datetime.now().isoformat(),
                model_name="test-model",
                prompt=run["prompt"]
            )

        # Search for "machine learning"
        results = self.db.search_by_prompt("machine learning")
        self.assertEqual(len(results), 2)

        # Search for "neural"
        results = self.db.search_by_prompt("neural")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['run_id'], "run_002")

    def test_delete_run_cascade(self):
        """Test that deleting a run cascades to all related records"""
        run_id = "test_run_010"
        timestamp = datetime.now().isoformat()

        # Create run with full hierarchy of data
        self.db.create_run(
            run_id=run_id,
            timestamp=timestamp,
            model_name="test-model",
            prompt="test prompt"
        )

        # Add power samples
        samples = [{
            "timestamp_ms": 0.0,
            "cpu_power_mw": 1000.0,
            "gpu_power_mw": 5000.0,
            "ane_power_mw": 500.0,
            "dram_power_mw": 300.0,
            "total_power_mw": 6800.0,
            "phase": "idle"
        }]
        self.db.add_power_samples(run_id, samples)

        # Add pipeline sections
        sections = [{
            "phase": "prefill",
            "section_name": "test_section",
            "start_time_ms": 0.0,
            "end_time_ms": 10.0,
            "duration_ms": 10.0
        }]
        self.db.add_pipeline_sections_batch(run_id, sections)

        # Add tokens
        tokens = [{
            "token_index": 0,
            "token_text": "test",
            "phase": "decode",
            "start_time_ms": 0.0,
            "end_time_ms": 10.0,
            "duration_ms": 10.0
        }]
        token_ids = self.db.add_tokens_batch(run_id, tokens)
        self.db.commit_transaction()

        # Add layer metrics
        layer_metrics = [{
            "layer_index": 0,
            "duration_ms": 5.0
        }]
        self.db.add_layer_metrics(token_ids[0], layer_metrics)
        self.db.commit_transaction()

        # Verify all data exists
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM power_samples WHERE run_id = ?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor.execute("SELECT COUNT(*) FROM pipeline_sections WHERE run_id = ?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor.execute("SELECT COUNT(*) FROM tokens WHERE run_id = ?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor.execute("""
            SELECT COUNT(*) FROM layer_metrics
            WHERE token_id IN (SELECT id FROM tokens WHERE run_id = ?)
        """, (run_id,))
        self.assertEqual(cursor.fetchone()[0], 1)

        # Delete the run
        deleted = self.db.delete_run(run_id)
        self.assertTrue(deleted)

        # Verify all related data is deleted (CASCADE)
        cursor.execute("SELECT COUNT(*) FROM power_samples WHERE run_id = ?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], 0)

        cursor.execute("SELECT COUNT(*) FROM pipeline_sections WHERE run_id = ?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], 0)

        cursor.execute("SELECT COUNT(*) FROM tokens WHERE run_id = ?", (run_id,))
        self.assertEqual(cursor.fetchone()[0], 0)

        # Layer metrics should also be deleted due to cascade
        cursor.execute("SELECT COUNT(*) FROM layer_metrics")
        self.assertEqual(cursor.fetchone()[0], 0)

    def test_cleanup_old_runs(self):
        """Test cleanup of old runs by max_runs and max_age"""
        base_time = datetime.now()

        # Create runs with different timestamps
        runs = [
            {
                "run_id": f"run_{i:03d}",
                "timestamp": (base_time - timedelta(days=i)).isoformat(),
                "model_name": "test-model",
                "prompt": f"test prompt {i}"
            }
            for i in range(10)
        ]

        for run in runs:
            self.db.create_run(**run)

        # Test: Dry run with max_runs
        result = self.db.cleanup_old_runs(max_runs=5, dry_run=True)
        self.assertEqual(result['runs_to_delete'], 5)
        self.assertIsNone(result['db_size_after'])

        # Verify nothing was deleted
        all_runs = self.db.get_runs(limit=100)
        self.assertEqual(len(all_runs), 10)

        # Test: Actual cleanup with max_runs
        result = self.db.cleanup_old_runs(max_runs=7, dry_run=False)
        self.assertEqual(result['runs_to_delete'], 3)
        self.assertIsNotNone(result['db_size_after'])

        # Verify correct number of runs remain
        all_runs = self.db.get_runs(limit=100)
        self.assertEqual(len(all_runs), 7)

        # Test: Cleanup by age
        result = self.db.cleanup_old_runs(max_age_days=5, dry_run=False)
        # Should delete runs older than 5 days (days 5-9 from remaining)
        self.assertGreater(result['runs_to_delete'], 0)

    def test_get_retention_stats(self):
        """Test getting retention statistics"""
        base_time = datetime.now()

        # Create some runs
        for i in range(3):
            self.db.create_run(
                run_id=f"run_{i:03d}",
                timestamp=(base_time - timedelta(days=i)).isoformat(),
                model_name="test-model",
                prompt=f"test prompt {i}"
            )

        stats = self.db.get_retention_stats()

        self.assertEqual(stats['total_runs'], 3)
        self.assertIsNotNone(stats['oldest_run_date'])
        self.assertIsNotNone(stats['newest_run_date'])
        self.assertGreater(stats['db_size_bytes'], 0)
        self.assertGreater(stats['db_size_mb'], 0)

    def test_init_database_helper(self):
        """Test the init_database helper function"""
        # Create a new temp file for this test
        temp_db2 = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
        temp_db2.close()
        db_path2 = temp_db2.name

        try:
            # Use helper function
            db = init_database(db_path2)

            self.assertIsNotNone(db)
            self.assertIsNotNone(db.conn)

            # Verify database is functional
            run_id = "test_run_helper"
            db.create_run(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                model_name="test-model",
                prompt="test prompt"
            )

            run = db.get_run(run_id)
            self.assertIsNotNone(run)
            self.assertEqual(run['run_id'], run_id)

            db.close()
        finally:
            if os.path.exists(db_path2):
                os.unlink(db_path2)

    def test_connection_error_handling(self):
        """Test error handling for connection issues"""
        # Test with invalid path (read-only location)
        invalid_db = ProfileDatabase("/invalid/path/that/does/not/exist/db.sqlite")

        # Should raise either PermissionError or OSError depending on platform
        with self.assertRaises((PermissionError, OSError)):
            invalid_db.connect()


if __name__ == '__main__':
    unittest.main()
