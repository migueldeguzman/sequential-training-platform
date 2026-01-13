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

    def create_run(
        self,
        run_id: str,
        timestamp: str,
        model_name: str,
        prompt: str,
        response: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[str] = None,
        profiling_depth: str = "module",
    ) -> int:
        """Create a new profiling run record.

        Args:
            run_id: Unique identifier for the run
            timestamp: ISO format timestamp
            model_name: Name of the model being profiled
            prompt: Input prompt text
            response: Generated response text
            experiment_name: Optional experiment name
            tags: Comma-separated tags
            profiling_depth: 'module' or 'deep'

        Returns:
            Database row ID of created run
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO profiling_runs (
                run_id, timestamp, model_name, prompt, response,
                experiment_name, tags, profiling_depth, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (run_id, timestamp, model_name, prompt, response, experiment_name, tags, profiling_depth),
        )
        self.conn.commit()
        logger.info(f"Created profiling run {run_id}")
        return cursor.lastrowid

    def add_power_samples(self, run_id: str, samples: list[dict]) -> None:
        """Batch insert power samples for a run.

        Args:
            run_id: Run identifier
            samples: List of power sample dicts with keys:
                timestamp_ms, cpu_power_mw, gpu_power_mw, ane_power_mw,
                dram_power_mw, total_power_mw
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO power_samples (
                run_id, timestamp_ms, cpu_power_mw, gpu_power_mw,
                ane_power_mw, dram_power_mw, total_power_mw
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    s["timestamp_ms"],
                    s["cpu_power_mw"],
                    s["gpu_power_mw"],
                    s["ane_power_mw"],
                    s["dram_power_mw"],
                    s["total_power_mw"],
                )
                for s in samples
            ],
        )
        self.conn.commit()
        logger.debug(f"Added {len(samples)} power samples for run {run_id}")

    def add_pipeline_section(
        self,
        run_id: str,
        phase: str,
        section_name: str,
        start_time_ms: float,
        end_time_ms: float,
        duration_ms: float,
        energy_mj: Optional[float] = None,
        avg_power_mw: Optional[float] = None,
    ) -> int:
        """Add a pipeline section timing record.

        Args:
            run_id: Run identifier
            phase: Pipeline phase (pre_inference, prefill, decode, post_inference)
            section_name: Section name within phase
            start_time_ms: Start timestamp in milliseconds
            end_time_ms: End timestamp in milliseconds
            duration_ms: Section duration
            energy_mj: Energy consumed in millijoules
            avg_power_mw: Average power in milliwatts

        Returns:
            Database row ID of created section
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO pipeline_sections (
                run_id, phase, section_name, start_time_ms, end_time_ms,
                duration_ms, energy_mj, avg_power_mw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, phase, section_name, start_time_ms, end_time_ms, duration_ms, energy_mj, avg_power_mw),
        )
        self.conn.commit()
        logger.debug(f"Added pipeline section {phase}/{section_name} for run {run_id}")
        return cursor.lastrowid

    def add_token(
        self,
        run_id: str,
        token_index: int,
        token_text: str,
        phase: str,
        start_time_ms: float,
        end_time_ms: float,
        duration_ms: float,
        energy_mj: Optional[float] = None,
        avg_power_mw: Optional[float] = None,
    ) -> int:
        """Add a token with its metrics.

        Args:
            run_id: Run identifier
            token_index: Token position in sequence
            token_text: Decoded token text
            phase: Phase (prefill or decode)
            start_time_ms: Start timestamp
            end_time_ms: End timestamp
            duration_ms: Token generation duration
            energy_mj: Energy consumed
            avg_power_mw: Average power

        Returns:
            Database row ID of created token
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO tokens (
                run_id, token_index, token_text, phase, start_time_ms,
                end_time_ms, duration_ms, energy_mj, avg_power_mw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, token_index, token_text, phase, start_time_ms, end_time_ms, duration_ms, energy_mj, avg_power_mw),
        )
        self.conn.commit()
        logger.debug(f"Added token {token_index} for run {run_id}")
        return cursor.lastrowid

    def add_layer_metrics(self, token_id: int, metrics: list[dict]) -> None:
        """Batch insert layer metrics for a token.

        Args:
            token_id: Token database ID
            metrics: List of layer metric dicts with keys:
                layer_index, duration_ms, energy_mj, avg_power_mw
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO layer_metrics (
                token_id, layer_index, duration_ms, energy_mj, avg_power_mw
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    token_id,
                    m["layer_index"],
                    m["duration_ms"],
                    m.get("energy_mj"),
                    m.get("avg_power_mw"),
                )
                for m in metrics
            ],
        )
        self.conn.commit()
        logger.debug(f"Added {len(metrics)} layer metrics for token {token_id}")

    def add_component_metrics(self, layer_metric_id: int, metrics: list[dict]) -> None:
        """Batch insert component metrics for a layer.

        Args:
            layer_metric_id: Layer metric database ID
            metrics: List of component metric dicts with keys:
                component_name, duration_ms, energy_mj, avg_power_mw,
                activation_mean, activation_std, activation_max, activation_sparsity
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO component_metrics (
                layer_metric_id, component_name, duration_ms, energy_mj, avg_power_mw,
                activation_mean, activation_std, activation_max, activation_sparsity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    layer_metric_id,
                    m["component_name"],
                    m["duration_ms"],
                    m.get("energy_mj"),
                    m.get("avg_power_mw"),
                    m.get("activation_mean"),
                    m.get("activation_std"),
                    m.get("activation_max"),
                    m.get("activation_sparsity"),
                )
                for m in metrics
            ],
        )
        self.conn.commit()
        logger.debug(f"Added {len(metrics)} component metrics for layer {layer_metric_id}")

    def add_deep_operation_metrics(self, component_metric_id: int, metrics: list[dict]) -> None:
        """Batch insert deep operation metrics for a component.

        Args:
            component_metric_id: Component metric database ID
            metrics: List of deep operation metric dicts with keys:
                operation_name, duration_ms, energy_mj, avg_power_mw,
                attention_entropy, attention_max_weight, attention_sparsity,
                mlp_kill_ratio, layernorm_variance_ratio
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO deep_operation_metrics (
                component_metric_id, operation_name, duration_ms, energy_mj, avg_power_mw,
                attention_entropy, attention_max_weight, attention_sparsity,
                mlp_kill_ratio, layernorm_variance_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    component_metric_id,
                    m["operation_name"],
                    m["duration_ms"],
                    m.get("energy_mj"),
                    m.get("avg_power_mw"),
                    m.get("attention_entropy"),
                    m.get("attention_max_weight"),
                    m.get("attention_sparsity"),
                    m.get("mlp_kill_ratio"),
                    m.get("layernorm_variance_ratio"),
                )
                for m in metrics
            ],
        )
        self.conn.commit()
        logger.debug(f"Added {len(metrics)} deep operation metrics for component {component_metric_id}")

    def get_run(self, run_id: str) -> Optional[dict]:
        """Retrieve full run data by run_id.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with full run data or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM profiling_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def get_runs(
        self,
        model: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tags: Optional[str] = None,
        experiment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Retrieve list of profiling runs with optional filters.

        Args:
            model: Filter by model name
            date_from: Filter runs from this timestamp (ISO format)
            date_to: Filter runs up to this timestamp (ISO format)
            tags: Filter by comma-separated tags
            experiment: Filter by experiment name
            limit: Maximum number of results
            offset: Number of results to skip (for pagination)

        Returns:
            List of run dictionaries
        """
        cursor = self.conn.cursor()

        # Build dynamic query with filters
        query = "SELECT * FROM profiling_runs WHERE 1=1"
        params = []

        if model:
            query += " AND model_name = ?"
            params.append(model)

        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)

        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to)

        if tags:
            # Check if any of the provided tags match
            query += " AND tags LIKE ?"
            params.append(f"%{tags}%")

        if experiment:
            query += " AND experiment_name = ?"
            params.append(experiment)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_run_summary(self, run_id: str) -> Optional[dict]:
        """Get aggregated summary statistics for a run.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with summary statistics or None if not found
        """
        cursor = self.conn.cursor()

        # Get basic run info
        cursor.execute("SELECT * FROM profiling_runs WHERE run_id = ?", (run_id,))
        run = cursor.fetchone()

        if not run:
            return None

        summary = dict(run)

        # Get phase breakdown
        cursor.execute(
            """
            SELECT
                phase,
                SUM(duration_ms) as total_duration_ms,
                SUM(energy_mj) as total_energy_mj,
                AVG(avg_power_mw) as avg_power_mw,
                COUNT(*) as section_count
            FROM pipeline_sections
            WHERE run_id = ?
            GROUP BY phase
            """,
            (run_id,)
        )
        phases = cursor.fetchall()
        summary["phase_breakdown"] = [dict(phase) for phase in phases]

        # Get average metrics per layer
        cursor.execute(
            """
            SELECT
                layer_index,
                AVG(duration_ms) as avg_duration_ms,
                AVG(energy_mj) as avg_energy_mj,
                AVG(avg_power_mw) as avg_power_mw,
                COUNT(*) as token_count
            FROM layer_metrics
            WHERE token_id IN (SELECT id FROM tokens WHERE run_id = ?)
            GROUP BY layer_index
            ORDER BY layer_index
            """,
            (run_id,)
        )
        layers = cursor.fetchall()
        summary["layer_averages"] = [dict(layer) for layer in layers]

        # Get average metrics per component
        cursor.execute(
            """
            SELECT
                cm.component_name,
                AVG(cm.duration_ms) as avg_duration_ms,
                AVG(cm.energy_mj) as avg_energy_mj,
                AVG(cm.avg_power_mw) as avg_power_mw,
                AVG(cm.activation_mean) as avg_activation_mean,
                AVG(cm.activation_std) as avg_activation_std,
                AVG(cm.activation_max) as avg_activation_max,
                AVG(cm.activation_sparsity) as avg_activation_sparsity,
                COUNT(*) as occurrence_count
            FROM component_metrics cm
            JOIN layer_metrics lm ON cm.layer_metric_id = lm.id
            JOIN tokens t ON lm.token_id = t.id
            WHERE t.run_id = ?
            GROUP BY cm.component_name
            ORDER BY avg_energy_mj DESC
            """,
            (run_id,)
        )
        components = cursor.fetchall()
        summary["component_averages"] = [dict(comp) for comp in components]

        # Identify hottest components (top 10 by energy)
        if components:
            summary["hottest_components"] = [dict(comp) for comp in components[:10]]

        return summary

    def get_tokens(self, run_id: str) -> list[dict]:
        """Get all tokens with their metrics for a run.

        Args:
            run_id: Run identifier

        Returns:
            List of token dictionaries with metrics
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM tokens
            WHERE run_id = ?
            ORDER BY token_index
            """,
            (run_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_layer_metrics(self, token_id: int) -> list[dict]:
        """Get layer metrics for a specific token.

        Args:
            token_id: Token database ID

        Returns:
            List of layer metric dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM layer_metrics
            WHERE token_id = ?
            ORDER BY layer_index
            """,
            (token_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_component_metrics(self, layer_metric_id: int) -> list[dict]:
        """Get component metrics for a specific layer.

        Args:
            layer_metric_id: Layer metric database ID

        Returns:
            List of component metric dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM component_metrics
            WHERE layer_metric_id = ?
            ORDER BY component_name
            """,
            (layer_metric_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_power_timeline(self, run_id: str) -> list[dict]:
        """Get power sample timeline for a run.

        Args:
            run_id: Run identifier

        Returns:
            List of power sample dictionaries ordered by timestamp
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM power_samples
            WHERE run_id = ?
            ORDER BY timestamp_ms
            """,
            (run_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def search_by_prompt(self, query_string: str, limit: int = 50) -> list[dict]:
        """Search for runs by prompt text.

        Args:
            query_string: Text to search for in prompts
            limit: Maximum number of results

        Returns:
            List of run dictionaries matching the search
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM profiling_runs
            WHERE prompt LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{query_string}%", limit)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete a profiling run and all related data.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM profiling_runs WHERE run_id = ?", (run_id,))
        self.conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted profiling run {run_id}")
        else:
            logger.warning(f"Profiling run {run_id} not found for deletion")

        return deleted


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
