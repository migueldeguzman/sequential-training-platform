"""
SQLite database management for Energy Profiler.

Provides complete schema creation and initialization for profiling data storage.
All profiling data clusters under the prompt that generated it.

Hierarchy: Prompt -> Phases -> Tokens -> Layers -> Components -> Operations
"""

import sqlite3
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ProfileDatabase:
    """SQLite database manager for profiling data."""

    def __init__(self, db_path: str = "backend/profiling.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Establish database connection and initialize schema."""
        # Ensure parent directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Initialize schema
        self._create_schema()

        logger.info(f"Connected to profiling database at {self.db_path}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Closed profiling database connection")

    def _create_schema(self):
        """Create all database tables and indexes."""
        cursor = self.conn.cursor()

        # profiling_runs table - Top-level run metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiling_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT,
                experiment_name TEXT,
                tags TEXT,
                profiling_depth TEXT NOT NULL,
                total_duration_ms REAL,
                total_energy_mj REAL,
                token_count INTEGER,
                tokens_per_second REAL,
                status TEXT DEFAULT 'running',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # power_samples table - Raw power measurements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS power_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp_ms REAL NOT NULL,
                cpu_power_mw REAL NOT NULL,
                gpu_power_mw REAL NOT NULL,
                ane_power_mw REAL NOT NULL,
                dram_power_mw REAL NOT NULL,
                total_power_mw REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES profiling_runs(run_id) ON DELETE CASCADE
            )
        """)

        # pipeline_sections table - Section-level timing and energy
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                section_name TEXT NOT NULL,
                start_time_ms REAL NOT NULL,
                end_time_ms REAL NOT NULL,
                duration_ms REAL NOT NULL,
                energy_mj REAL,
                avg_power_mw REAL,
                FOREIGN KEY (run_id) REFERENCES profiling_runs(run_id) ON DELETE CASCADE
            )
        """)

        # tokens table - Per-token metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                token_index INTEGER NOT NULL,
                token_text TEXT NOT NULL,
                phase TEXT NOT NULL,
                start_time_ms REAL NOT NULL,
                end_time_ms REAL NOT NULL,
                duration_ms REAL NOT NULL,
                energy_mj REAL,
                avg_power_mw REAL,
                FOREIGN KEY (run_id) REFERENCES profiling_runs(run_id) ON DELETE CASCADE
            )
        """)

        # layer_metrics table - Per-layer per-token metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layer_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id INTEGER NOT NULL,
                layer_index INTEGER NOT NULL,
                duration_ms REAL NOT NULL,
                energy_mj REAL,
                avg_power_mw REAL,
                FOREIGN KEY (token_id) REFERENCES tokens(id) ON DELETE CASCADE
            )
        """)

        # component_metrics table - Per-component (attention, MLP, etc.) metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS component_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer_metric_id INTEGER NOT NULL,
                component_name TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                energy_mj REAL,
                avg_power_mw REAL,
                activation_mean REAL,
                activation_std REAL,
                activation_max REAL,
                activation_sparsity REAL,
                FOREIGN KEY (layer_metric_id) REFERENCES layer_metrics(id) ON DELETE CASCADE
            )
        """)

        # deep_operation_metrics table - Lowest-level operation metrics (optional)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deep_operation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_metric_id INTEGER NOT NULL,
                operation_name TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                energy_mj REAL,
                avg_power_mw REAL,
                attention_entropy REAL,
                attention_max_weight REAL,
                attention_sparsity REAL,
                mlp_kill_ratio REAL,
                layernorm_variance_ratio REAL,
                FOREIGN KEY (component_metric_id) REFERENCES component_metrics(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for query performance

        # Index on run_id for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_power_samples_run_id
            ON power_samples(run_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pipeline_sections_run_id
            ON pipeline_sections(run_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tokens_run_id
            ON tokens(run_id)
        """)

        # Index on token_id for layer metrics
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_layer_metrics_token_id
            ON layer_metrics(token_id)
        """)

        # Index on layer_metric_id for component metrics
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_component_metrics_layer_metric_id
            ON component_metrics(layer_metric_id)
        """)

        # Index on component_metric_id for deep operation metrics
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_deep_operation_metrics_component_metric_id
            ON deep_operation_metrics(component_metric_id)
        """)

        # Composite indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiling_runs_timestamp
            ON profiling_runs(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiling_runs_model
            ON profiling_runs(model_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiling_runs_tags
            ON profiling_runs(tags)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tokens_phase_index
            ON tokens(run_id, phase, token_index)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_layer_metrics_layer_index
            ON layer_metrics(token_id, layer_index)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_component_metrics_name
            ON component_metrics(layer_metric_id, component_name)
        """)

        self.conn.commit()
        logger.info("Database schema initialized successfully")


def init_database(db_path: str = "backend/profiling.db") -> ProfileDatabase:
    """Initialize and return a connected ProfileDatabase instance.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Connected ProfileDatabase instance
    """
    db = ProfileDatabase(db_path)
    db.connect()
    return db
