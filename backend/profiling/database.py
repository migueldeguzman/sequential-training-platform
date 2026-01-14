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
        """Establish database connection and initialize schema.

        Raises:
            sqlite3.OperationalError: If database file cannot be created or accessed
            PermissionError: If insufficient permissions to write to database location
            Exception: For other database initialization errors
        """
        try:
            # Ensure parent directory exists
            db_file = Path(self.db_path)
            try:
                db_file.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                raise PermissionError(
                    f"Insufficient permissions to create database directory: {db_file.parent}. "
                    f"Error: {str(e)}"
                )

            # Connect to database
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row  # Enable column access by name
            except sqlite3.OperationalError as e:
                raise sqlite3.OperationalError(
                    f"Failed to connect to database at {self.db_path}. "
                    f"Check if the path is valid and writable. Error: {str(e)}"
                )

            # Initialize schema
            try:
                self._create_schema()
            except Exception as e:
                logger.error(f"Failed to initialize database schema: {str(e)}")
                if self.conn:
                    self.conn.close()
                raise

            logger.info(f"Connected to profiling database at {self.db_path}")

        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

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
                input_token_count INTEGER,
                output_token_count INTEGER,
                prefill_energy_mj REAL,
                decode_energy_mj REAL,
                energy_per_input_token_mj REAL,
                energy_per_output_token_mj REAL,
                input_output_energy_ratio REAL,
                peak_power_mw REAL,
                peak_power_cpu_mw REAL,
                peak_power_gpu_mw REAL,
                peak_power_ane_mw REAL,
                peak_power_dram_mw REAL,
                peak_power_timestamp_ms REAL,
                baseline_power_mw REAL,
                baseline_cpu_power_mw REAL,
                baseline_gpu_power_mw REAL,
                baseline_ane_power_mw REAL,
                baseline_dram_power_mw REAL,
                baseline_sample_count INTEGER,
                kv_cache_size_mb REAL,
                kv_cache_utilization_pct REAL,
                kv_cache_memory_limit_mb REAL,
                context_length INTEGER,
                batch_size INTEGER DEFAULT 1,
                edp REAL,
                edp_per_token REAL,
                prefill_edp REAL,
                decode_edp REAL,
                electricity_price_per_kwh REAL DEFAULT 0.12,
                carbon_intensity_g_per_kwh REAL DEFAULT 400.0,
                cost_usd REAL,
                co2_grams REAL,
                precision TEXT,
                quantization_method TEXT,
                num_layers INTEGER,
                hidden_size INTEGER,
                intermediate_size INTEGER,
                num_attention_heads INTEGER,
                num_key_value_heads INTEGER,
                total_params INTEGER,
                attention_mechanism TEXT,
                is_moe BOOLEAN DEFAULT 0,
                num_experts INTEGER,
                num_active_experts INTEGER,
                architecture_type TEXT,
                inference_engine TEXT,
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
                phase TEXT DEFAULT 'idle',
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
                edp REAL,
                is_input_token BOOLEAN NOT NULL DEFAULT 0,
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

        # moe_expert_activations table - Per-token expert activation patterns for MoE models
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moe_expert_activations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id INTEGER NOT NULL,
                layer_index INTEGER NOT NULL,
                active_expert_ids TEXT NOT NULL,
                num_active_experts INTEGER NOT NULL,
                expert_weights TEXT,
                routing_entropy REAL,
                load_balance_loss REAL,
                FOREIGN KEY (token_id) REFERENCES tokens(id) ON DELETE CASCADE
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

        # Index on token_id for MoE expert activations
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_moe_expert_activations_token_id
            ON moe_expert_activations(token_id)
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
            CREATE INDEX IF NOT EXISTS idx_profiling_runs_inference_engine
            ON profiling_runs(inference_engine)
        """)

        # Migration: Add precision and quantization_method columns if they don't exist
        # Check if columns exist
        cursor.execute("PRAGMA table_info(profiling_runs)")
        columns = [row[1] for row in cursor.fetchall()]

        if "precision" not in columns:
            cursor.execute("ALTER TABLE profiling_runs ADD COLUMN precision TEXT")
            logger.info("Added precision column to profiling_runs table")

        if "quantization_method" not in columns:
            cursor.execute("ALTER TABLE profiling_runs ADD COLUMN quantization_method TEXT")
            logger.info("Added quantization_method column to profiling_runs table")

        if "inference_engine" not in columns:
            cursor.execute("ALTER TABLE profiling_runs ADD COLUMN inference_engine TEXT")
            logger.info("Added inference_engine column to profiling_runs table")

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
        batch_size: int = 1,
        electricity_price_per_kwh: float = 0.12,
        carbon_intensity_g_per_kwh: float = 400.0,
        precision: Optional[str] = None,
        quantization_method: Optional[str] = None,
        inference_engine: Optional[str] = None,
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
            batch_size: Batch size used for inference
            electricity_price_per_kwh: Cost of electricity in USD per kWh (default: $0.12)
            carbon_intensity_g_per_kwh: Carbon intensity in grams CO2 per kWh (default: 400g for US grid)
            precision: Model precision (FP32, FP16, BF16, FP8, INT8, INT4, MIXED)
            quantization_method: Quantization method (gptq, awq, gguf, bitsandbytes, None)
            inference_engine: Inference engine/backend used (transformers, mlx, vllm, etc.)

        Returns:
            Database row ID of created run
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO profiling_runs (
                run_id, timestamp, model_name, prompt, response,
                experiment_name, tags, profiling_depth, batch_size,
                electricity_price_per_kwh, carbon_intensity_g_per_kwh,
                precision, quantization_method, inference_engine, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (run_id, timestamp, model_name, prompt, response, experiment_name, tags, profiling_depth, batch_size,
             electricity_price_per_kwh, carbon_intensity_g_per_kwh, precision, quantization_method, inference_engine),
        )
        self.conn.commit()
        logger.info(f"Created profiling run {run_id} with batch_size={batch_size}, precision={precision}, quantization_method={quantization_method}, inference_engine={inference_engine}")
        return cursor.lastrowid

    def update_run_metrics(
        self,
        run_id: str,
        total_duration_ms: Optional[float] = None,
        total_energy_mj: Optional[float] = None,
        token_count: Optional[int] = None,
        tokens_per_second: Optional[float] = None,
        input_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        prefill_energy_mj: Optional[float] = None,
        decode_energy_mj: Optional[float] = None,
        energy_per_input_token_mj: Optional[float] = None,
        energy_per_output_token_mj: Optional[float] = None,
        input_output_energy_ratio: Optional[float] = None,
        peak_power_mw: Optional[float] = None,
        peak_power_cpu_mw: Optional[float] = None,
        peak_power_gpu_mw: Optional[float] = None,
        peak_power_ane_mw: Optional[float] = None,
        peak_power_dram_mw: Optional[float] = None,
        peak_power_timestamp_ms: Optional[float] = None,
        baseline_power_mw: Optional[float] = None,
        baseline_cpu_power_mw: Optional[float] = None,
        baseline_gpu_power_mw: Optional[float] = None,
        baseline_ane_power_mw: Optional[float] = None,
        baseline_dram_power_mw: Optional[float] = None,
        baseline_sample_count: Optional[int] = None,
        kv_cache_size_mb: Optional[float] = None,
        kv_cache_utilization_pct: Optional[float] = None,
        kv_cache_memory_limit_mb: Optional[float] = None,
        context_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        edp: Optional[float] = None,
        edp_per_token: Optional[float] = None,
        prefill_edp: Optional[float] = None,
        decode_edp: Optional[float] = None,
        cost_usd: Optional[float] = None,
        co2_grams: Optional[float] = None,
        status: str = "completed",
    ) -> None:
        """Update run with final metrics.

        Args:
            run_id: Run identifier
            total_duration_ms: Total inference duration
            total_energy_mj: Total energy consumed
            token_count: Total number of tokens generated
            tokens_per_second: Generation throughput
            input_token_count: Number of input/prompt tokens
            output_token_count: Number of output/generated tokens
            prefill_energy_mj: Energy consumed during prefill phase
            decode_energy_mj: Energy consumed during decode phase
            energy_per_input_token_mj: Average energy per input token
            energy_per_output_token_mj: Average energy per output token
            input_output_energy_ratio: Ratio of output to input token energy
            peak_power_mw: Peak total power during run
            peak_power_cpu_mw: Peak CPU power during run
            peak_power_gpu_mw: Peak GPU power during run
            peak_power_ane_mw: Peak ANE power during run
            peak_power_dram_mw: Peak DRAM power during run
            peak_power_timestamp_ms: Time when peak power occurred (relative to start)
            baseline_power_mw: Average idle baseline total power
            baseline_cpu_power_mw: Average idle baseline CPU power
            baseline_gpu_power_mw: Average idle baseline GPU power
            baseline_ane_power_mw: Average idle baseline ANE power
            baseline_dram_power_mw: Average idle baseline DRAM power
            baseline_sample_count: Number of samples used for baseline
            kv_cache_size_mb: KV cache memory usage in megabytes
            kv_cache_utilization_pct: KV cache utilization as percentage of limit
            kv_cache_memory_limit_mb: KV cache memory limit in megabytes
            context_length: Total context length (input + output tokens)
            batch_size: Batch size used for inference
            edp: Energy-Delay Product (total_energy_mj × total_duration_ms)
            edp_per_token: EDP normalized by token count
            prefill_edp: EDP for prefill phase only
            decode_edp: EDP for decode phase only
            cost_usd: Estimated electricity cost in USD
            co2_grams: Estimated CO2 emissions in grams
            status: Run status (default: 'completed')
        """
        cursor = self.conn.cursor()

        # Build dynamic UPDATE statement based on provided values
        updates = []
        values = []

        if total_duration_ms is not None:
            updates.append("total_duration_ms = ?")
            values.append(total_duration_ms)
        if total_energy_mj is not None:
            updates.append("total_energy_mj = ?")
            values.append(total_energy_mj)
        if token_count is not None:
            updates.append("token_count = ?")
            values.append(token_count)
        if tokens_per_second is not None:
            updates.append("tokens_per_second = ?")
            values.append(tokens_per_second)
        if input_token_count is not None:
            updates.append("input_token_count = ?")
            values.append(input_token_count)
        if output_token_count is not None:
            updates.append("output_token_count = ?")
            values.append(output_token_count)
        if prefill_energy_mj is not None:
            updates.append("prefill_energy_mj = ?")
            values.append(prefill_energy_mj)
        if decode_energy_mj is not None:
            updates.append("decode_energy_mj = ?")
            values.append(decode_energy_mj)
        if energy_per_input_token_mj is not None:
            updates.append("energy_per_input_token_mj = ?")
            values.append(energy_per_input_token_mj)
        if energy_per_output_token_mj is not None:
            updates.append("energy_per_output_token_mj = ?")
            values.append(energy_per_output_token_mj)
        if input_output_energy_ratio is not None:
            updates.append("input_output_energy_ratio = ?")
            values.append(input_output_energy_ratio)
        if peak_power_mw is not None:
            updates.append("peak_power_mw = ?")
            values.append(peak_power_mw)
        if peak_power_cpu_mw is not None:
            updates.append("peak_power_cpu_mw = ?")
            values.append(peak_power_cpu_mw)
        if peak_power_gpu_mw is not None:
            updates.append("peak_power_gpu_mw = ?")
            values.append(peak_power_gpu_mw)
        if peak_power_ane_mw is not None:
            updates.append("peak_power_ane_mw = ?")
            values.append(peak_power_ane_mw)
        if peak_power_dram_mw is not None:
            updates.append("peak_power_dram_mw = ?")
            values.append(peak_power_dram_mw)
        if peak_power_timestamp_ms is not None:
            updates.append("peak_power_timestamp_ms = ?")
            values.append(peak_power_timestamp_ms)
        if baseline_power_mw is not None:
            updates.append("baseline_power_mw = ?")
            values.append(baseline_power_mw)
        if baseline_cpu_power_mw is not None:
            updates.append("baseline_cpu_power_mw = ?")
            values.append(baseline_cpu_power_mw)
        if baseline_gpu_power_mw is not None:
            updates.append("baseline_gpu_power_mw = ?")
            values.append(baseline_gpu_power_mw)
        if baseline_ane_power_mw is not None:
            updates.append("baseline_ane_power_mw = ?")
            values.append(baseline_ane_power_mw)
        if baseline_dram_power_mw is not None:
            updates.append("baseline_dram_power_mw = ?")
            values.append(baseline_dram_power_mw)
        if baseline_sample_count is not None:
            updates.append("baseline_sample_count = ?")
            values.append(baseline_sample_count)
        if kv_cache_size_mb is not None:
            updates.append("kv_cache_size_mb = ?")
            values.append(kv_cache_size_mb)
        if kv_cache_utilization_pct is not None:
            updates.append("kv_cache_utilization_pct = ?")
            values.append(kv_cache_utilization_pct)
        if kv_cache_memory_limit_mb is not None:
            updates.append("kv_cache_memory_limit_mb = ?")
            values.append(kv_cache_memory_limit_mb)
        if context_length is not None:
            updates.append("context_length = ?")
            values.append(context_length)
        if batch_size is not None:
            updates.append("batch_size = ?")
            values.append(batch_size)
        if edp is not None:
            updates.append("edp = ?")
            values.append(edp)
        if edp_per_token is not None:
            updates.append("edp_per_token = ?")
            values.append(edp_per_token)
        if prefill_edp is not None:
            updates.append("prefill_edp = ?")
            values.append(prefill_edp)
        if decode_edp is not None:
            updates.append("decode_edp = ?")
            values.append(decode_edp)
        if cost_usd is not None:
            updates.append("cost_usd = ?")
            values.append(cost_usd)
        if co2_grams is not None:
            updates.append("co2_grams = ?")
            values.append(co2_grams)

        updates.append("status = ?")
        values.append(status)
        values.append(run_id)

        query = f"UPDATE profiling_runs SET {', '.join(updates)} WHERE run_id = ?"
        cursor.execute(query, values)
        self.conn.commit()
        logger.info(f"Updated profiling run {run_id} with final metrics")

    def add_power_samples(self, run_id: str, samples: list[dict]) -> None:
        """Batch insert power samples for a run.

        Args:
            run_id: Run identifier
            samples: List of power sample dicts with keys:
                timestamp_ms, cpu_power_mw, gpu_power_mw, ane_power_mw,
                dram_power_mw, total_power_mw, phase

        Raises:
            sqlite3.IntegrityError: If run_id doesn't exist (foreign key constraint)
            sqlite3.OperationalError: If database write fails
        """
        if not self.conn:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            cursor = self.conn.cursor()
            cursor.executemany(
                """
                INSERT INTO power_samples (
                    run_id, timestamp_ms, cpu_power_mw, gpu_power_mw,
                    ane_power_mw, dram_power_mw, total_power_mw, phase
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                        s.get("phase", "idle"),
                    )
                    for s in samples
                ],
            )
            self.conn.commit()
            logger.debug(f"Added {len(samples)} power samples for run {run_id}")
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            logger.error(f"Failed to add power samples: foreign key constraint violation for run {run_id}")
            raise sqlite3.IntegrityError(
                f"Cannot add power samples: run_id '{run_id}' does not exist. Error: {str(e)}"
            )
        except sqlite3.OperationalError as e:
            self.conn.rollback()
            logger.error(f"Database write failed for power samples: {str(e)}")
            raise
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Unexpected error adding power samples: {str(e)}")
            raise RuntimeError(f"Failed to add power samples: {str(e)}")

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
        # Defer commit for batch operations
        logger.debug(f"Added pipeline section {phase}/{section_name} for run {run_id}")
        return cursor.lastrowid

    def add_pipeline_sections_batch(self, run_id: str, sections: list[dict]) -> None:
        """Batch insert pipeline sections for a run.

        Args:
            run_id: Run identifier
            sections: List of section dicts with keys:
                phase, section_name, start_time_ms, end_time_ms, duration_ms, energy_mj, avg_power_mw
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO pipeline_sections (
                run_id, phase, section_name, start_time_ms, end_time_ms,
                duration_ms, energy_mj, avg_power_mw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    s["phase"],
                    s["section_name"],
                    s["start_time_ms"],
                    s["end_time_ms"],
                    s["duration_ms"],
                    s.get("energy_mj"),
                    s.get("avg_power_mw"),
                )
                for s in sections
            ],
        )
        logger.debug(f"Batch added {len(sections)} pipeline sections for run {run_id}")

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
        edp: Optional[float] = None,
        is_input_token: bool = False,
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
            edp: Energy-Delay Product (energy_mj × duration_ms)
            is_input_token: True if this is an input token (prefill), False if output (decode)

        Returns:
            Database row ID of created token
        """
        cursor = self.conn.cursor()

        # Calculate EDP if not provided
        if edp is None and energy_mj is not None and duration_ms is not None:
            edp = energy_mj * duration_ms

        cursor.execute(
            """
            INSERT INTO tokens (
                run_id, token_index, token_text, phase, start_time_ms,
                end_time_ms, duration_ms, energy_mj, avg_power_mw, edp, is_input_token
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, token_index, token_text, phase, start_time_ms, end_time_ms, duration_ms, energy_mj, avg_power_mw, edp, is_input_token),
        )
        # Defer commit for batch operations
        logger.debug(f"Added token {token_index} for run {run_id}")
        return cursor.lastrowid

    def add_tokens_batch(self, run_id: str, tokens: list[dict]) -> list[int]:
        """Batch insert tokens for a run.

        Args:
            run_id: Run identifier
            tokens: List of token dicts with keys:
                token_index, token_text, phase, start_time_ms, end_time_ms, duration_ms, energy_mj, avg_power_mw, edp, is_input_token

        Returns:
            List of database row IDs for created tokens
        """
        cursor = self.conn.cursor()
        token_ids = []
        for token in tokens:
            # Calculate EDP if not provided
            edp = token.get("edp")
            if edp is None:
                energy_mj = token.get("energy_mj")
                duration_ms = token.get("duration_ms")
                if energy_mj is not None and duration_ms is not None:
                    edp = energy_mj * duration_ms

            cursor.execute(
                """
                INSERT INTO tokens (
                    run_id, token_index, token_text, phase, start_time_ms,
                    end_time_ms, duration_ms, energy_mj, avg_power_mw, edp, is_input_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    token["token_index"],
                    token.get("token_text"),
                    token["phase"],
                    token["start_time_ms"],
                    token["end_time_ms"],
                    token["duration_ms"],
                    token.get("energy_mj"),
                    token.get("avg_power_mw"),
                    edp,
                    token.get("is_input_token", False),
                ),
            )
            token_ids.append(cursor.lastrowid)
        logger.debug(f"Batch added {len(tokens)} tokens for run {run_id}")
        return token_ids

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
        # Defer commit for batch operations
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
        # Defer commit for batch operations
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
        # Defer commit for batch operations
        logger.debug(f"Added {len(metrics)} deep operation metrics for component {component_metric_id}")

    def commit_transaction(self) -> None:
        """Commit all pending database operations.

        This should be called after batch operations to persist changes.
        Calling this explicitly allows for better control over when disk writes occur.
        """
        if self.conn:
            self.conn.commit()
            logger.debug("Committed database transaction")

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
        inference_engine: Optional[str] = None,
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
            inference_engine: Filter by inference engine/backend
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

        if inference_engine:
            query += " AND inference_engine = ?"
            params.append(inference_engine)

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

        # Get phase breakdown from pipeline sections
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

        # Get phase-specific energy from power samples (direct from phase tags)
        cursor.execute(
            """
            SELECT
                phase,
                COUNT(*) as sample_count,
                AVG(total_power_mw) as avg_power_mw,
                MAX(total_power_mw) as peak_power_mw,
                AVG(cpu_power_mw) as avg_cpu_power_mw,
                AVG(gpu_power_mw) as avg_gpu_power_mw,
                AVG(ane_power_mw) as avg_ane_power_mw,
                AVG(dram_power_mw) as avg_dram_power_mw
            FROM power_samples
            WHERE run_id = ?
            GROUP BY phase
            """,
            (run_id,)
        )
        power_phases = cursor.fetchall()
        summary["phase_power_breakdown"] = [dict(phase) for phase in power_phases]

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

        # Get input vs output token energy breakdown
        cursor.execute(
            """
            SELECT
                SUM(CASE WHEN is_input_token = 1 THEN energy_mj ELSE 0 END) as input_energy_mj,
                SUM(CASE WHEN is_input_token = 0 THEN energy_mj ELSE 0 END) as output_energy_mj,
                SUM(CASE WHEN is_input_token = 1 THEN 1 ELSE 0 END) as input_token_count,
                SUM(CASE WHEN is_input_token = 0 THEN 1 ELSE 0 END) as output_token_count
            FROM tokens
            WHERE run_id = ?
            """,
            (run_id,)
        )
        token_breakdown = cursor.fetchone()
        if token_breakdown:
            breakdown = dict(token_breakdown)
            # Calculate per-token metrics
            if breakdown["input_token_count"] and breakdown["input_token_count"] > 0:
                breakdown["energy_per_input_token_mj"] = breakdown["input_energy_mj"] / breakdown["input_token_count"]
            else:
                breakdown["energy_per_input_token_mj"] = 0

            if breakdown["output_token_count"] and breakdown["output_token_count"] > 0:
                breakdown["energy_per_output_token_mj"] = breakdown["output_energy_mj"] / breakdown["output_token_count"]
            else:
                breakdown["energy_per_output_token_mj"] = 0

            # Calculate ratio (output / input energy per token)
            if breakdown["energy_per_input_token_mj"] and breakdown["energy_per_input_token_mj"] > 0:
                breakdown["output_to_input_energy_ratio"] = breakdown["energy_per_output_token_mj"] / breakdown["energy_per_input_token_mj"]
            else:
                breakdown["output_to_input_energy_ratio"] = 0

            summary["token_energy_breakdown"] = breakdown

        # Calculate energy efficiency metrics (EP-076)
        efficiency_metrics = {}

        # Total energy per token (mJ/token)
        if summary.get("token_count") and summary["token_count"] > 0:
            efficiency_metrics["total_energy_per_token_mj"] = summary["total_energy_mj"] / summary["token_count"]
        else:
            efficiency_metrics["total_energy_per_token_mj"] = 0

        # Prefill energy per token
        if summary.get("input_token_count") and summary["input_token_count"] > 0:
            efficiency_metrics["prefill_energy_per_token_mj"] = summary.get("prefill_energy_mj", 0) / summary["input_token_count"]
        else:
            efficiency_metrics["prefill_energy_per_token_mj"] = 0

        # Decode energy per token
        if summary.get("output_token_count") and summary["output_token_count"] > 0:
            efficiency_metrics["decode_energy_per_token_mj"] = summary.get("decode_energy_mj", 0) / summary["output_token_count"]
        else:
            efficiency_metrics["decode_energy_per_token_mj"] = 0

        # Energy per million parameters (need model parameter count - placeholder for now)
        # This will be properly implemented when model features are extracted
        efficiency_metrics["energy_per_million_params_mj"] = None

        # Tokens per joule (efficiency score - higher is better)
        if summary.get("total_energy_mj") and summary["total_energy_mj"] > 0:
            # Convert mJ to J: divide by 1000
            total_energy_j = summary["total_energy_mj"] / 1000.0
            if summary.get("token_count") and summary["token_count"] > 0:
                efficiency_metrics["tokens_per_joule"] = summary["token_count"] / total_energy_j
            else:
                efficiency_metrics["tokens_per_joule"] = 0
        else:
            efficiency_metrics["tokens_per_joule"] = 0

        # Power utilization percentage (actual vs TDP - needs hardware TDP config)
        # For M4 Max, estimated TDP is ~90W for CPU+GPU combined
        # This is a placeholder - should be configurable
        M4_MAX_ESTIMATED_TDP_MW = 90000  # 90W in milliwatts

        # Get average total power across all samples
        cursor.execute(
            """
            SELECT AVG(total_power_mw) as avg_total_power_mw
            FROM power_samples
            WHERE run_id = ? AND phase != 'idle'
            """,
            (run_id,)
        )
        avg_power_row = cursor.fetchone()
        if avg_power_row and avg_power_row["avg_total_power_mw"]:
            avg_power_mw = avg_power_row["avg_total_power_mw"]
            efficiency_metrics["avg_power_mw"] = avg_power_mw
            efficiency_metrics["power_utilization_percentage"] = (avg_power_mw / M4_MAX_ESTIMATED_TDP_MW) * 100
        else:
            efficiency_metrics["avg_power_mw"] = 0
            efficiency_metrics["power_utilization_percentage"] = 0

        # Joules per token (standardized metric from TokenPowerBench)
        # Convert mJ/token to J/token
        if efficiency_metrics["total_energy_per_token_mj"]:
            efficiency_metrics["joules_per_token"] = efficiency_metrics["total_energy_per_token_mj"] / 1000.0
        else:
            efficiency_metrics["joules_per_token"] = 0

        # Joules per input token and output token
        if efficiency_metrics["prefill_energy_per_token_mj"]:
            efficiency_metrics["joules_per_input_token"] = efficiency_metrics["prefill_energy_per_token_mj"] / 1000.0
        else:
            efficiency_metrics["joules_per_input_token"] = 0

        if efficiency_metrics["decode_energy_per_token_mj"]:
            efficiency_metrics["joules_per_output_token"] = efficiency_metrics["decode_energy_per_token_mj"] / 1000.0
        else:
            efficiency_metrics["joules_per_output_token"] = 0

        summary["efficiency_metrics"] = efficiency_metrics

        # Calculate cost and carbon estimates (EP-089)
        cost_carbon_metrics = {}

        # Get electricity price and carbon intensity from run settings
        electricity_price = summary.get("electricity_price_per_kwh", 0.12)
        carbon_intensity = summary.get("carbon_intensity_g_per_kwh", 400.0)

        if summary.get("total_energy_mj"):
            # Convert mJ to kWh: mJ -> J -> Wh -> kWh
            # 1 mJ = 0.001 J
            # 1 Wh = 3600 J
            # 1 kWh = 1000 Wh
            total_energy_kwh = summary["total_energy_mj"] / 3600000.0

            # Calculate cost: energy (kWh) × price ($/kWh)
            cost_carbon_metrics["cost_usd"] = total_energy_kwh * electricity_price

            # Calculate CO2 emissions: energy (kWh) × carbon intensity (g/kWh)
            cost_carbon_metrics["co2_grams"] = total_energy_kwh * carbon_intensity

            # Also provide in kg for convenience
            cost_carbon_metrics["co2_kg"] = cost_carbon_metrics["co2_grams"] / 1000.0

            # Calculate per-token costs
            if summary.get("token_count") and summary["token_count"] > 0:
                cost_carbon_metrics["cost_per_token_usd"] = cost_carbon_metrics["cost_usd"] / summary["token_count"]
                cost_carbon_metrics["co2_per_token_grams"] = cost_carbon_metrics["co2_grams"] / summary["token_count"]
            else:
                cost_carbon_metrics["cost_per_token_usd"] = 0
                cost_carbon_metrics["co2_per_token_grams"] = 0

            # Store settings used for calculation
            cost_carbon_metrics["electricity_price_per_kwh"] = electricity_price
            cost_carbon_metrics["carbon_intensity_g_per_kwh"] = carbon_intensity

            # Provide context: equivalent CO2 comparisons
            # Average car emits ~404g CO2 per mile (EPA 2023)
            if cost_carbon_metrics["co2_grams"] > 0:
                cost_carbon_metrics["equivalent_car_miles"] = cost_carbon_metrics["co2_grams"] / 404.0
        else:
            cost_carbon_metrics["cost_usd"] = 0
            cost_carbon_metrics["co2_grams"] = 0
            cost_carbon_metrics["co2_kg"] = 0
            cost_carbon_metrics["cost_per_token_usd"] = 0
            cost_carbon_metrics["co2_per_token_grams"] = 0
            cost_carbon_metrics["electricity_price_per_kwh"] = electricity_price
            cost_carbon_metrics["carbon_intensity_g_per_kwh"] = carbon_intensity
            cost_carbon_metrics["equivalent_car_miles"] = 0

        summary["cost_carbon_metrics"] = cost_carbon_metrics

        # Calculate Energy-Delay Product (EDP) metrics (EP-088)
        edp_metrics = {}

        # Total EDP = total_energy_mj × total_duration_ms
        # Lower EDP is better (optimizes both energy and speed)
        if summary.get("total_energy_mj") and summary.get("total_duration_ms"):
            edp_metrics["edp"] = summary["total_energy_mj"] * summary["total_duration_ms"]

            # EDP per token
            if summary.get("token_count") and summary["token_count"] > 0:
                edp_metrics["edp_per_token"] = edp_metrics["edp"] / summary["token_count"]
            else:
                edp_metrics["edp_per_token"] = 0
        else:
            edp_metrics["edp"] = 0
            edp_metrics["edp_per_token"] = 0

        # Prefill EDP (prefill_energy_mj × prefill_duration_ms)
        # Need to get prefill duration from pipeline sections
        cursor.execute(
            """
            SELECT SUM(duration_ms) as prefill_duration_ms
            FROM pipeline_sections
            WHERE run_id = ? AND phase = 'prefill'
            """,
            (run_id,)
        )
        prefill_duration_row = cursor.fetchone()
        if prefill_duration_row and prefill_duration_row["prefill_duration_ms"] and summary.get("prefill_energy_mj"):
            prefill_duration_ms = prefill_duration_row["prefill_duration_ms"]
            edp_metrics["prefill_edp"] = summary["prefill_energy_mj"] * prefill_duration_ms
            edp_metrics["prefill_duration_ms"] = prefill_duration_ms
        else:
            edp_metrics["prefill_edp"] = 0
            edp_metrics["prefill_duration_ms"] = 0

        # Decode EDP (decode_energy_mj × decode_duration_ms)
        cursor.execute(
            """
            SELECT SUM(duration_ms) as decode_duration_ms
            FROM pipeline_sections
            WHERE run_id = ? AND phase = 'decode'
            """,
            (run_id,)
        )
        decode_duration_row = cursor.fetchone()
        if decode_duration_row and decode_duration_row["decode_duration_ms"] and summary.get("decode_energy_mj"):
            decode_duration_ms = decode_duration_row["decode_duration_ms"]
            edp_metrics["decode_edp"] = summary["decode_energy_mj"] * decode_duration_ms
            edp_metrics["decode_duration_ms"] = decode_duration_ms
        else:
            edp_metrics["decode_edp"] = 0
            edp_metrics["decode_duration_ms"] = 0

        summary["edp_metrics"] = edp_metrics

        # Calculate component energy breakdown (EP-090)
        # Get total energy per hardware component from power samples
        cursor.execute(
            """
            SELECT
                run_id,
                -- Calculate energy for each component (Power * Time)
                -- Each sample represents 100ms interval (0.1s)
                SUM(cpu_power_mw * 0.1) as cpu_energy_mj,
                SUM(gpu_power_mw * 0.1) as gpu_energy_mj,
                SUM(ane_power_mw * 0.1) as ane_energy_mj,
                SUM(dram_power_mw * 0.1) as dram_energy_mj,
                SUM(total_power_mw * 0.1) as total_energy_mj,
                AVG(cpu_power_mw) as avg_cpu_power_mw,
                AVG(gpu_power_mw) as avg_gpu_power_mw,
                AVG(ane_power_mw) as avg_ane_power_mw,
                AVG(dram_power_mw) as avg_dram_power_mw,
                MAX(cpu_power_mw) as peak_cpu_power_mw,
                MAX(gpu_power_mw) as peak_gpu_power_mw,
                MAX(ane_power_mw) as peak_ane_power_mw,
                MAX(dram_power_mw) as peak_dram_power_mw,
                COUNT(*) as sample_count
            FROM power_samples
            WHERE run_id = ? AND phase != 'idle'
            GROUP BY run_id
            """,
            (run_id,)
        )
        component_row = cursor.fetchone()

        if component_row:
            component_breakdown = dict(component_row)

            # Calculate percentages of total energy
            total_e = component_breakdown["total_energy_mj"]
            if total_e and total_e > 0:
                component_breakdown["cpu_energy_percentage"] = (component_breakdown["cpu_energy_mj"] / total_e) * 100
                component_breakdown["gpu_energy_percentage"] = (component_breakdown["gpu_energy_mj"] / total_e) * 100
                component_breakdown["ane_energy_percentage"] = (component_breakdown["ane_energy_mj"] / total_e) * 100
                component_breakdown["dram_energy_percentage"] = (component_breakdown["dram_energy_mj"] / total_e) * 100
            else:
                component_breakdown["cpu_energy_percentage"] = 0
                component_breakdown["gpu_energy_percentage"] = 0
                component_breakdown["ane_energy_percentage"] = 0
                component_breakdown["dram_energy_percentage"] = 0

            summary["component_energy_breakdown"] = component_breakdown
        else:
            summary["component_energy_breakdown"] = None

        # Get component breakdown by phase (for detailed analysis)
        cursor.execute(
            """
            SELECT
                phase,
                SUM(cpu_power_mw * 0.1) as cpu_energy_mj,
                SUM(gpu_power_mw * 0.1) as gpu_energy_mj,
                SUM(ane_power_mw * 0.1) as ane_energy_mj,
                SUM(dram_power_mw * 0.1) as dram_energy_mj,
                SUM(total_power_mw * 0.1) as total_energy_mj
            FROM power_samples
            WHERE run_id = ? AND phase != 'idle'
            GROUP BY phase
            """,
            (run_id,)
        )
        phase_components = cursor.fetchall()
        summary["component_energy_by_phase"] = [dict(pc) for pc in phase_components]

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

    def get_pipeline_sections(self, run_id: str) -> list[dict]:
        """Get all pipeline sections for a run, grouped by phase.

        Args:
            run_id: Run identifier

        Returns:
            List of pipeline section dictionaries ordered by start time
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM pipeline_sections
            WHERE run_id = ?
            ORDER BY start_time_ms
            """,
            (run_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_database_size_bytes(self) -> int:
        """Get the size of the database file in bytes.

        Returns:
            Database file size in bytes
        """
        db_file = Path(self.db_path)
        if db_file.exists():
            return db_file.stat().st_size
        return 0

    def get_run_count(self) -> int:
        """Get the total number of profiling runs.

        Returns:
            Total number of runs in database
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM profiling_runs")
        row = cursor.fetchone()
        return row["count"] if row else 0

    def cleanup_old_runs(self, max_runs: int = 0, max_age_days: int = 0, dry_run: bool = False) -> dict:
        """Clean up old profiling runs based on retention policies.

        Args:
            max_runs: Keep only this many most recent runs (0 = unlimited)
            max_age_days: Delete runs older than this many days (0 = unlimited)
            dry_run: If True, return what would be deleted without actually deleting

        Returns:
            Dictionary with cleanup statistics:
                - runs_to_delete: Number of runs that will be/were deleted
                - run_ids: List of run_ids that will be/were deleted
                - db_size_before: Database size before cleanup
                - db_size_after: Database size after cleanup (None if dry_run)
        """
        cursor = self.conn.cursor()
        runs_to_delete = []

        # Get database size before
        db_size_before = self.get_database_size_bytes()

        # Find runs to delete based on max_runs limit
        if max_runs > 0:
            cursor.execute(
                """
                SELECT run_id FROM profiling_runs
                ORDER BY timestamp DESC
                LIMIT -1 OFFSET ?
                """,
                (max_runs,)
            )
            excess_runs = cursor.fetchall()
            runs_to_delete.extend([row["run_id"] for row in excess_runs])

        # Find runs to delete based on age
        if max_age_days > 0:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cutoff_timestamp = cutoff_date.isoformat()

            cursor.execute(
                """
                SELECT run_id FROM profiling_runs
                WHERE timestamp < ?
                """,
                (cutoff_timestamp,)
            )
            old_runs = cursor.fetchall()
            # Combine with age-based deletion (using set to avoid duplicates)
            runs_to_delete = list(set(runs_to_delete + [row["run_id"] for row in old_runs]))

        result = {
            "runs_to_delete": len(runs_to_delete),
            "run_ids": runs_to_delete,
            "db_size_before": db_size_before,
            "db_size_after": None
        }

        # Perform actual deletion if not dry run
        if not dry_run and runs_to_delete:
            for run_id in runs_to_delete:
                self.delete_run(run_id)

            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            self.conn.commit()

            # Get size after cleanup
            result["db_size_after"] = self.get_database_size_bytes()
            result["space_freed_bytes"] = db_size_before - result["db_size_after"]

            logger.info(f"Cleanup completed: deleted {len(runs_to_delete)} runs, freed {result.get('space_freed_bytes', 0)} bytes")

        return result

    def get_retention_stats(self) -> dict:
        """Get statistics about database retention and storage.

        Returns:
            Dictionary with retention statistics:
                - total_runs: Total number of runs
                - oldest_run_date: Timestamp of oldest run
                - newest_run_date: Timestamp of newest run
                - db_size_bytes: Current database size
                - db_size_mb: Current database size in MB
        """
        cursor = self.conn.cursor()

        # Get total run count
        cursor.execute("SELECT COUNT(*) as count FROM profiling_runs")
        count_row = cursor.fetchone()
        total_runs = count_row["count"] if count_row else 0

        # Get oldest and newest run timestamps
        cursor.execute("SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM profiling_runs")
        date_row = cursor.fetchone()

        stats = {
            "total_runs": total_runs,
            "oldest_run_date": date_row["oldest"] if date_row else None,
            "newest_run_date": date_row["newest"] if date_row else None,
            "db_size_bytes": self.get_database_size_bytes(),
        }

        # Calculate size in MB for convenience
        stats["db_size_mb"] = stats["db_size_bytes"] / (1024 * 1024)

        return stats

    def get_long_context_analysis(self, run_id: Optional[str] = None, model_name: Optional[str] = None) -> dict:
        """Analyze how context length affects energy consumption and KV cache pressure.

        Args:
            run_id: Optional specific run to analyze. If None, analyzes all runs.
            model_name: Optional filter by model name

        Returns:
            Dictionary with long context analysis:
                - context_length_vs_energy: List of {context_length, energy_mj, energy_per_token}
                - kv_cache_stats: KV cache utilization statistics
                - saturation_point: Estimated context length where KV cache saturates (if detectable)
                - warnings: List of warnings for runs approaching memory limits
        """
        cursor = self.conn.cursor()

        # Build query with optional filters
        query = """
            SELECT
                run_id,
                context_length,
                input_token_count,
                output_token_count,
                total_energy_mj,
                total_duration_ms,
                kv_cache_size_mb,
                kv_cache_utilization_pct,
                kv_cache_memory_limit_mb,
                model_name
            FROM profiling_runs
            WHERE context_length IS NOT NULL
        """
        params = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY context_length ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return {
                "context_length_vs_energy": [],
                "kv_cache_stats": None,
                "saturation_point": None,
                "warnings": []
            }

        # Process data points
        data_points = []
        kv_cache_utilizations = []
        warnings = []

        for row in rows:
            run_data = dict(row)
            context_length = run_data["context_length"]
            total_energy = run_data["total_energy_mj"]
            token_count = run_data["input_token_count"] + run_data["output_token_count"]

            energy_per_token = total_energy / token_count if token_count > 0 else 0

            data_points.append({
                "run_id": run_data["run_id"],
                "context_length": context_length,
                "energy_mj": total_energy,
                "energy_per_token_mj": energy_per_token,
                "duration_ms": run_data["total_duration_ms"],
                "kv_cache_size_mb": run_data["kv_cache_size_mb"],
                "kv_cache_utilization_pct": run_data["kv_cache_utilization_pct"]
            })

            # Track KV cache utilization
            if run_data["kv_cache_utilization_pct"] is not None:
                kv_cache_utilizations.append(run_data["kv_cache_utilization_pct"])

            # Check for approaching memory limits
            if run_data["kv_cache_utilization_pct"] and run_data["kv_cache_utilization_pct"] > 80:
                warnings.append({
                    "run_id": run_data["run_id"],
                    "context_length": context_length,
                    "utilization_pct": run_data["kv_cache_utilization_pct"],
                    "message": f"KV cache utilization at {run_data['kv_cache_utilization_pct']:.1f}% - approaching memory limit"
                })

        # Calculate KV cache statistics
        kv_cache_stats = None
        if kv_cache_utilizations:
            kv_cache_stats = {
                "avg_utilization_pct": sum(kv_cache_utilizations) / len(kv_cache_utilizations),
                "max_utilization_pct": max(kv_cache_utilizations),
                "min_utilization_pct": min(kv_cache_utilizations)
            }

        # Detect saturation point (where energy per token starts increasing sharply)
        # Simple heuristic: find context length where energy/token increases by >20% from previous
        saturation_point = None
        if len(data_points) >= 3:
            for i in range(1, len(data_points)):
                prev_ept = data_points[i-1]["energy_per_token_mj"]
                curr_ept = data_points[i]["energy_per_token_mj"]

                if prev_ept > 0:
                    increase_pct = ((curr_ept - prev_ept) / prev_ept) * 100
                    if increase_pct > 20:
                        saturation_point = {
                            "context_length": data_points[i]["context_length"],
                            "energy_increase_pct": increase_pct,
                            "message": f"Energy per token increased by {increase_pct:.1f}% at context length {data_points[i]['context_length']}"
                        }
                        break

        return {
            "context_length_vs_energy": data_points,
            "kv_cache_stats": kv_cache_stats,
            "saturation_point": saturation_point,
            "warnings": warnings
        }


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
