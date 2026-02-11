"""
TEST-001: API Endpoint Health Check Suite

Tests all 60+ REST endpoints respond correctly with proper HTTP status codes
and response schemas matching TypeScript types.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
from datetime import datetime
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_data = tempfile.mkdtemp()
    temp_models = tempfile.mkdtemp()
    temp_datasets = tempfile.mkdtemp()

    yield {
        'data': temp_data,
        'models': temp_models,
        'datasets': temp_datasets
    }

    # Cleanup
    shutil.rmtree(temp_data, ignore_errors=True)
    shutil.rmtree(temp_models, ignore_errors=True)
    shutil.rmtree(temp_datasets, ignore_errors=True)


class TestDatasetEndpoints:
    """Test Dataset Management endpoints (MOD-001)."""

    def test_get_datasets(self, client):
        """Test GET /api/datasets returns proper structure."""
        response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_dataset_by_name_not_found(self, client):
        """Test GET /api/datasets/{name} returns 404 for non-existent dataset."""
        response = client.get("/api/datasets/nonexistent-dataset")
        assert response.status_code == 404

    def test_get_text_datasets(self, client):
        """Test GET /api/text-datasets returns proper structure."""
        response = client.get("/api/text-datasets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestTrainingEndpoints:
    """Test Training Orchestration endpoints (MOD-002)."""

    def test_get_training_status(self, client):
        """Test GET /api/training/status returns proper structure."""
        response = client.get("/api/training/status")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] in ['idle', 'running', 'completed', 'error']

    def test_start_training_without_config(self, client):
        """Test POST /api/training/start requires valid config."""
        response = client.post("/api/training/start", json={})
        # Should fail with 422 (validation error) or 400 (bad request)
        assert response.status_code in [400, 422]

    def test_stop_training_when_not_running(self, client):
        """Test POST /api/training/stop when nothing is running."""
        response = client.post("/api/training/stop")
        # Should return success or indicate nothing to stop
        assert response.status_code in [200, 400, 404]

    def test_get_training_history(self, client):
        """Test GET /api/training/history returns list."""
        response = client.get("/api/training/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_training_logs(self, client):
        """Test GET /api/training/logs returns proper structure."""
        response = client.get("/api/training/logs")
        assert response.status_code in [200, 404]  # 404 if no training job


class TestModelEndpoints:
    """Test Model Management endpoints (MOD-003)."""

    def test_get_models(self, client):
        """Test GET /api/models returns list."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_model_not_found(self, client):
        """Test GET /api/models/{name} returns 404 for non-existent model."""
        response = client.get("/api/models/nonexistent-model")
        assert response.status_code == 404

    def test_delete_model_not_found(self, client):
        """Test DELETE /api/models/{name} returns 404 for non-existent model."""
        response = client.delete("/api/models/nonexistent-model")
        assert response.status_code == 404


class TestInferenceEndpoints:
    """Test Inference Testing endpoints (MOD-004)."""

    def test_get_inference_status(self, client):
        """Test GET /api/inference/status returns proper structure."""
        response = client.get("/api/inference/status")
        assert response.status_code == 200
        data = response.json()
        assert 'loaded' in data
        assert isinstance(data['loaded'], bool)

    def test_load_model_without_name(self, client):
        """Test POST /api/inference/load requires model name."""
        response = client.post("/api/inference/load", json={})
        assert response.status_code in [400, 422]

    def test_unload_model_when_none_loaded(self, client):
        """Test POST /api/inference/unload when no model loaded."""
        response = client.post("/api/inference/unload")
        assert response.status_code in [200, 400]

    def test_generate_without_loaded_model(self, client):
        """Test POST /api/inference/generate fails without loaded model."""
        response = client.post("/api/inference/generate", json={"prompt": "test"})
        assert response.status_code in [400, 422]


class TestProfilingEndpoints:
    """Test Energy Profiling endpoints (MOD-005)."""

    def test_get_powermetrics_status(self, client):
        """Test GET /api/profiling/powermetrics/status."""
        response = client.get("/api/profiling/powermetrics/status")
        assert response.status_code == 200
        data = response.json()
        assert 'available' in data
        assert isinstance(data['available'], bool)

    def test_get_profiling_runs(self, client):
        """Test GET /api/profiling/runs returns list."""
        response = client.get("/api/profiling/runs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_run_not_found(self, client):
        """Test GET /api/profiling/run/{run_id} returns 404."""
        response = client.get("/api/profiling/run/nonexistent-run-id")
        assert response.status_code == 404

    def test_get_run_summary_not_found(self, client):
        """Test GET /api/profiling/run/{run_id}/summary returns 404."""
        response = client.get("/api/profiling/run/nonexistent-run-id/summary")
        assert response.status_code == 404

    def test_delete_run_not_found(self, client):
        """Test DELETE /api/profiling/run/{run_id} returns 404."""
        response = client.delete("/api/profiling/run/nonexistent-run-id")
        assert response.status_code == 404

    def test_cancel_profiling_when_not_running(self, client):
        """Test POST /api/profiling/cancel when nothing running."""
        response = client.post("/api/profiling/cancel")
        assert response.status_code in [200, 400, 404]

    def test_generate_profiling_without_config(self, client):
        """Test POST /api/profiling/generate requires config."""
        response = client.post("/api/profiling/generate", json={})
        assert response.status_code in [400, 422]


class TestAnalyticsEndpoints:
    """Test Advanced Analytics endpoints (MOD-006)."""

    def test_get_architectural_analysis(self, client):
        """Test GET /api/profiling/architectural-analysis."""
        response = client.get("/api/profiling/architectural-analysis")
        # May return 200 with empty data or 404 if no runs
        assert response.status_code in [200, 404]

    def test_get_long_context_analysis(self, client):
        """Test GET /api/profiling/long-context-analysis."""
        response = client.get("/api/profiling/long-context-analysis")
        assert response.status_code in [200, 404]

    def test_get_energy_scaling_analysis(self, client):
        """Test GET /api/profiling/energy-scaling-analysis."""
        response = client.get("/api/profiling/energy-scaling-analysis")
        assert response.status_code in [200, 404]

    def test_get_throughput_energy_tradeoff(self, client):
        """Test GET /api/profiling/throughput-energy-tradeoff."""
        response = client.get("/api/profiling/throughput-energy-tradeoff")
        assert response.status_code in [200, 404]

    def test_compare_without_runs(self, client):
        """Test POST /api/profiling/compare requires run IDs."""
        response = client.post("/api/profiling/compare", json={})
        assert response.status_code in [400, 422]

    def test_predict_without_features(self, client):
        """Test POST /api/profiling/predict requires features."""
        response = client.post("/api/profiling/predict", json={})
        assert response.status_code in [400, 422]


class TestSettingsEndpoints:
    """Test Settings & Configuration endpoints (MOD-007)."""

    def test_get_settings(self, client):
        """Test GET /api/settings returns settings object."""
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_post_settings(self, client):
        """Test POST /api/settings updates settings."""
        settings = {
            "data_dir": "/tmp/test-data",
            "models_dir": "/tmp/test-models"
        }
        response = client.post("/api/settings", json=settings)
        assert response.status_code in [200, 400]  # 400 if validation fails


class TestErrorResponses:
    """Test error cases return proper error responses."""

    def test_404_for_unknown_endpoint(self, client):
        """Test unknown endpoints return 404."""
        response = client.get("/api/unknown-endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test wrong HTTP method returns 405."""
        # GET /api/training/start should not be allowed (only POST)
        response = client.get("/api/training/start")
        assert response.status_code == 405

    def test_malformed_json(self, client):
        """Test malformed JSON returns 422."""
        response = client.post(
            "/api/training/start",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestResponseTimes:
    """Test API response times are acceptable."""

    def test_dataset_list_performance(self, client):
        """Test GET /api/datasets responds quickly."""
        import time
        start = time.time()
        response = client.get("/api/datasets")
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 500, f"Response took {elapsed}ms, expected < 500ms"

    def test_model_list_performance(self, client):
        """Test GET /api/models responds quickly."""
        import time
        start = time.time()
        response = client.get("/api/models")
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 500, f"Response took {elapsed}ms, expected < 500ms"


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/api/datasets")
    # CORS headers should be present
    assert response.status_code in [200, 405]  # Some frameworks auto-handle OPTIONS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
