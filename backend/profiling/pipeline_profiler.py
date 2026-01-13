"""
InferencePipelineProfiler - Orchestrates profiling of complete inference pipeline.

This module provides the main profiler that coordinates all profiling components:
- PowerMonitor: Samples system power during inference
- LayerProfiler: Captures layer and component timing/activations
- DeepAttentionProfiler: Captures deep operation metrics (optional)
- ProfileDatabase: Stores all profiling data

The profiler wraps the inference pipeline and provides section timing context managers
to measure energy consumption for each phase and operation.

Usage:
    profiler = InferencePipelineProfiler(
        power_monitor=PowerMonitor(),
        layer_profiler=LayerProfiler(model),
        deep_profiler=DeepAttentionProfiler(model),  # Optional
        database=ProfileDatabase()
    )

    with profiler.run(prompt="Hello world", model_name="llama-7b") as session:
        with session.section("tokenization", phase="pre_inference"):
            tokens = tokenizer.encode(prompt)

        with session.section("prefill", phase="prefill"):
            output = model(tokens)

        # ... rest of inference

    # Data is automatically saved to database
"""

import time
import uuid
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from .power_monitor import PowerMonitor, PowerSample
from .layer_profiler import LayerProfiler, ComponentTiming
from .deep_profiler import DeepAttentionProfiler, AttentionOperationMetrics, MLPOperationMetrics, LayerNormOperationMetrics
from .database import ProfileDatabase

logger = logging.getLogger(__name__)


@dataclass
class SectionTiming:
    """Timing and energy data for a pipeline section."""
    phase: str  # pre_inference, prefill, decode, post_inference
    section_name: str
    start_time: float
    end_time: float
    duration_ms: float
    energy_mj: Optional[float] = None
    avg_power_mw: Optional[float] = None
    power_samples: List[PowerSample] = field(default_factory=list)


@dataclass
class ProfilingSession:
    """Context for a single profiling run."""
    run_id: str
    start_time: float
    prompt: str
    model_name: str
    profiling_depth: str
    experiment_name: Optional[str] = None
    tags: Optional[str] = None

    # Collected data during the session
    sections: List[SectionTiming] = field(default_factory=list)
    response: Optional[str] = None

    # References to profiling components
    power_monitor: Optional[PowerMonitor] = None
    layer_profiler: Optional[LayerProfiler] = None
    deep_profiler: Optional[DeepAttentionProfiler] = None
    database: Optional[ProfileDatabase] = None


class InferencePipelineProfiler:
    """
    Main profiler orchestrating all profiling components.

    Coordinates power monitoring, layer profiling, deep operation profiling,
    and database storage for complete inference pipeline profiling.
    """

    def __init__(
        self,
        power_monitor: Optional[PowerMonitor] = None,
        layer_profiler: Optional[LayerProfiler] = None,
        deep_profiler: Optional[DeepAttentionProfiler] = None,
        database: Optional[ProfileDatabase] = None
    ):
        """
        Initialize the inference pipeline profiler.

        Args:
            power_monitor: PowerMonitor instance for system power sampling
            layer_profiler: LayerProfiler instance for layer/component metrics
            deep_profiler: DeepAttentionProfiler instance for operation-level metrics (optional)
            database: ProfileDatabase instance for storing profiling data
        """
        self.power_monitor = power_monitor
        self.layer_profiler = layer_profiler
        self.deep_profiler = deep_profiler
        self.database = database

        # Current active session
        self._current_session: Optional[ProfilingSession] = None

        logger.info("Initialized InferencePipelineProfiler")

    def _generate_run_id(self) -> str:
        """
        Generate a unique run ID.

        Returns:
            Unique identifier string (UUID4)
        """
        return str(uuid.uuid4())

    @contextmanager
    def run(
        self,
        prompt: str,
        model_name: str,
        profiling_depth: str = "module",
        experiment_name: Optional[str] = None,
        tags: Optional[str] = None
    ):
        """
        Context manager for a profiling run session.

        This manages the lifecycle of a complete profiling session:
        1. Generates unique run ID
        2. Starts power monitoring
        3. Yields session object for section timing
        4. Stops power monitoring
        5. Aggregates and saves all data to database

        Args:
            prompt: Input prompt being profiled
            model_name: Name of the model being profiled
            profiling_depth: 'module' or 'deep' profiling level
            experiment_name: Optional experiment name for organization
            tags: Optional comma-separated tags

        Yields:
            ProfilingSession object with section() context manager

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                with session.section("tokenization", "pre_inference"):
                    tokens = tokenize(prompt)
                with session.section("prefill", "prefill"):
                    output = model(tokens)
        """
        # Generate unique run ID
        run_id = self._generate_run_id()
        start_time = time.time()

        logger.info(f"Starting profiling run {run_id} for model {model_name}")

        # Create session object
        session = ProfilingSession(
            run_id=run_id,
            start_time=start_time,
            prompt=prompt,
            model_name=model_name,
            profiling_depth=profiling_depth,
            experiment_name=experiment_name,
            tags=tags,
            power_monitor=self.power_monitor,
            layer_profiler=self.layer_profiler,
            deep_profiler=self.deep_profiler,
            database=self.database
        )

        # Store as current session
        self._current_session = session

        # Start power monitoring
        if self.power_monitor:
            try:
                self.power_monitor.start()
                logger.info("Power monitoring started")
            except Exception as e:
                logger.error(f"Failed to start power monitoring: {e}")

        # Register layer profiler hooks if needed
        if self.layer_profiler and profiling_depth in ["module", "deep"]:
            try:
                self.layer_profiler.register_hooks()
                logger.info("Layer profiler hooks registered")
            except Exception as e:
                logger.error(f"Failed to register layer profiler hooks: {e}")

        # Patch deep profiler if needed
        if self.deep_profiler and profiling_depth == "deep":
            try:
                self.deep_profiler.patch()
                logger.info("Deep profiler patches applied")
            except Exception as e:
                logger.error(f"Failed to apply deep profiler patches: {e}")

        try:
            # Yield session to caller for profiling
            yield session

        finally:
            # Stop power monitoring
            if self.power_monitor and self.power_monitor.is_running():
                try:
                    self.power_monitor.stop()
                    logger.info("Power monitoring stopped")
                except Exception as e:
                    logger.error(f"Failed to stop power monitoring: {e}")

            # Detach layer profiler hooks
            if self.layer_profiler:
                try:
                    self.layer_profiler.detach()
                    logger.info("Layer profiler hooks removed")
                except Exception as e:
                    logger.error(f"Failed to detach layer profiler: {e}")

            # Unpatch deep profiler
            if self.deep_profiler and self.deep_profiler.is_patched:
                try:
                    self.deep_profiler.unpatch()
                    logger.info("Deep profiler patches removed")
                except Exception as e:
                    logger.error(f"Failed to unpatch deep profiler: {e}")

            # Save data to database
            if self.database:
                try:
                    self._save_run_to_database(session)
                    logger.info(f"Profiling run {run_id} saved to database")
                except Exception as e:
                    logger.error(f"Failed to save run to database: {e}")

            # Clear current session
            self._current_session = None

    def _save_run_to_database(self, session: ProfilingSession) -> None:
        """
        Save complete profiling run data to database.

        Args:
            session: ProfilingSession with collected data
        """
        if not self.database:
            logger.warning("No database configured, skipping save")
            return

        # Calculate total metrics
        end_time = time.time()
        total_duration_ms = (end_time - session.start_time) * 1000.0

        # Calculate total energy from power samples
        total_energy_mj = 0.0
        power_samples = []

        if self.power_monitor:
            samples = self.power_monitor.get_samples()
            power_samples = samples

            # Calculate energy: integrate power over time
            # Energy (mJ) = sum(power_mW * time_interval_ms)
            for i in range(len(samples) - 1):
                time_interval_ms = samples[i + 1].relative_time_ms - samples[i].relative_time_ms
                avg_power_mw = (samples[i].total_power_mw + samples[i + 1].total_power_mw) / 2.0
                energy_mj = avg_power_mw * time_interval_ms / 1000.0  # Convert to mJ
                total_energy_mj += energy_mj

        # Calculate tokens per second (placeholder - will be filled by caller)
        token_count = 0  # Will be updated by token-level profiling
        tokens_per_second = 0.0

        # Create run record
        timestamp = datetime.fromtimestamp(session.start_time).isoformat()

        self.database.create_run(
            run_id=session.run_id,
            timestamp=timestamp,
            model_name=session.model_name,
            prompt=session.prompt,
            response=session.response,
            experiment_name=session.experiment_name,
            tags=session.tags,
            profiling_depth=session.profiling_depth
        )

        # Save power samples
        if power_samples:
            power_sample_dicts = [
                {
                    "timestamp_ms": s.relative_time_ms,
                    "cpu_power_mw": s.cpu_power_mw,
                    "gpu_power_mw": s.gpu_power_mw,
                    "ane_power_mw": s.ane_power_mw,
                    "dram_power_mw": s.dram_power_mw,
                    "total_power_mw": s.total_power_mw
                }
                for s in power_samples
            ]
            self.database.add_power_samples(session.run_id, power_sample_dicts)

        # Save pipeline sections
        for section in session.sections:
            self.database.add_pipeline_section(
                run_id=session.run_id,
                phase=section.phase,
                section_name=section.section_name,
                start_time_ms=section.start_time * 1000.0,
                end_time_ms=section.end_time * 1000.0,
                duration_ms=section.duration_ms,
                energy_mj=section.energy_mj,
                avg_power_mw=section.avg_power_mw
            )

        logger.info(f"Saved run {session.run_id} with {len(power_samples)} power samples and {len(session.sections)} sections")

    # Pre-Inference Phase Profiling Helpers
    def profile_tokenization(self, session: ProfilingSession, tokenizer, prompt: str):
        """
        Profile tokenization step with automatic section timing.

        Args:
            session: Active ProfilingSession from run() context manager
            tokenizer: Tokenizer instance
            prompt: Input prompt text

        Returns:
            Tokenized output (tokens/input_ids)

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                tokens = profiler.profile_tokenization(session, tokenizer, prompt)
        """
        with session.section("tokenization", "pre_inference"):
            tokens = tokenizer.encode(prompt)
        return tokens

    def profile_tensor_transfer(self, session: ProfilingSession, tensor, device: str):
        """
        Profile tensor transfer to device with automatic section timing.

        Args:
            session: Active ProfilingSession from run() context manager
            tensor: Input tensor to transfer
            device: Target device (e.g., "mps", "cuda", "cpu")

        Returns:
            Transferred tensor

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                tokens = profiler.profile_tokenization(session, tokenizer, prompt)
                tokens = profiler.profile_tensor_transfer(session, tokens, "mps")
        """
        with session.section("tensor_transfer", "pre_inference"):
            transferred = tensor.to(device)
        return transferred

    def profile_kv_cache_init(self, session: ProfilingSession, init_func, *args, **kwargs):
        """
        Profile KV-cache initialization with automatic section timing.

        Args:
            session: Active ProfilingSession from run() context manager
            init_func: Function that initializes KV-cache
            *args: Positional arguments for init_func
            **kwargs: Keyword arguments for init_func

        Returns:
            Result from init_func

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                tokens = profiler.profile_tokenization(session, tokenizer, prompt)
                tokens = profiler.profile_tensor_transfer(session, tokens, "mps")
                cache = profiler.profile_kv_cache_init(session, model.init_kv_cache, batch_size=1)
        """
        with session.section("kv_cache_init", "pre_inference"):
            result = init_func(*args, **kwargs)
        return result

    # Prefill Phase Profiling Helpers
    def profile_embedding_lookup(self, session: ProfilingSession, embedding_func, *args, **kwargs):
        """
        Profile embedding lookup with automatic section timing.

        Args:
            session: Active ProfilingSession from run() context manager
            embedding_func: Function that performs embedding lookup (e.g., model.embed_tokens)
            *args: Positional arguments for embedding_func
            **kwargs: Keyword arguments for embedding_func

        Returns:
            Embedding tensor

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                tokens = profiler.profile_tokenization(session, tokenizer, prompt)
                embeddings = profiler.profile_embedding_lookup(session, model.embed_tokens, tokens)
        """
        with session.section("embedding_lookup", "prefill"):
            result = embedding_func(*args, **kwargs)
        return result

    def profile_position_embedding(self, session: ProfilingSession, position_func, *args, **kwargs):
        """
        Profile position embedding with automatic section timing.

        Args:
            session: Active ProfilingSession from run() context manager
            position_func: Function that adds positional encoding
            *args: Positional arguments for position_func
            **kwargs: Keyword arguments for position_func

        Returns:
            Result with position embeddings added

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                embeddings = profiler.profile_embedding_lookup(session, model.embed_tokens, tokens)
                embeddings = profiler.profile_position_embedding(session, model.add_position_embeddings, embeddings)
        """
        with session.section("position_embedding", "prefill"):
            result = position_func(*args, **kwargs)
        return result

    def profile_transformer_layers(self, session: ProfilingSession, layers_func, *args, **kwargs):
        """
        Profile all transformer layers during prefill with automatic section timing.

        This wraps the forward pass through all transformer layers. The LayerProfiler hooks
        will capture detailed metrics for each individual layer and component.

        Args:
            session: Active ProfilingSession from run() context manager
            layers_func: Function that runs transformer layers (e.g., model.layers forward)
            *args: Positional arguments for layers_func
            **kwargs: Keyword arguments for layers_func

        Returns:
            Output from transformer layers

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                embeddings = profiler.profile_embedding_lookup(session, model.embed_tokens, tokens)
                hidden_states = profiler.profile_transformer_layers(session, model.layers, embeddings)
        """
        with session.section("layers", "prefill"):
            result = layers_func(*args, **kwargs)
        return result

    def profile_final_layernorm(self, session: ProfilingSession, layernorm_func, *args, **kwargs):
        """
        Profile final layer normalization with automatic section timing.

        Args:
            session: Active ProfilingSession from run() context manager
            layernorm_func: Function that applies final layer norm
            *args: Positional arguments for layernorm_func
            **kwargs: Keyword arguments for layernorm_func

        Returns:
            Normalized output

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                hidden_states = profiler.profile_transformer_layers(session, model.layers, embeddings)
                normalized = profiler.profile_final_layernorm(session, model.norm, hidden_states)
        """
        with session.section("final_layernorm", "prefill"):
            result = layernorm_func(*args, **kwargs)
        return result

    def profile_lm_head(self, session: ProfilingSession, lm_head_func, *args, **kwargs):
        """
        Profile language model head projection with automatic section timing.

        This projects hidden states to vocabulary logits.

        Args:
            session: Active ProfilingSession from run() context manager
            lm_head_func: Function that projects to vocabulary (e.g., model.lm_head)
            *args: Positional arguments for lm_head_func
            **kwargs: Keyword arguments for lm_head_func

        Returns:
            Vocabulary logits

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                normalized = profiler.profile_final_layernorm(session, model.norm, hidden_states)
                logits = profiler.profile_lm_head(session, model.lm_head, normalized)
        """
        with session.section("lm_head", "prefill"):
            result = lm_head_func(*args, **kwargs)
        return result

    def profile_kv_cache_store(self, session: ProfilingSession, store_func, *args, **kwargs):
        """
        Profile KV-cache storage during prefill with automatic section timing.

        This captures the time to store keys and values from prefill for later decode steps.

        Args:
            session: Active ProfilingSession from run() context manager
            store_func: Function that stores KV-cache
            *args: Positional arguments for store_func
            **kwargs: Keyword arguments for store_func

        Returns:
            Result from store_func

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                logits = profiler.profile_lm_head(session, model.lm_head, normalized)
                profiler.profile_kv_cache_store(session, cache.store, keys, values)
        """
        with session.section("kv_cache_store", "prefill"):
            result = store_func(*args, **kwargs)
        return result

    def profile_prefill(
        self,
        session: ProfilingSession,
        model,
        input_ids,
        return_full_output: bool = False
    ):
        """
        Profile the entire prefill phase with automatic section breakdown.

        This is a convenience method that wraps a complete prefill forward pass and
        automatically profiles all major sections. For finer-grained control, use
        the individual profile_* methods instead.

        Args:
            session: Active ProfilingSession from run() context manager
            model: The transformer model
            input_ids: Input token IDs tensor
            return_full_output: If True, return full model output; if False, return only logits

        Returns:
            Model output (logits or full output depending on return_full_output)

        Example:
            with profiler.run("Hello", "llama-7b") as session:
                tokens = profiler.profile_tokenization(session, tokenizer, prompt)
                logits = profiler.profile_prefill(session, model, tokens)
        """
        with session.section("prefill_complete", "prefill"):
            # Reset layer profiler for fresh metrics
            if session.layer_profiler:
                session.layer_profiler.reset()

            # Run forward pass (hooks will capture layer/component metrics)
            output = model(input_ids, return_dict=True)

            # Get layer metrics if available
            if session.layer_profiler:
                layer_timings = session.layer_profiler.get_timings()
                logger.debug(f"Captured {len(layer_timings)} layer timings during prefill")

        return output if return_full_output else output.logits


# Add section() method to ProfilingSession
def _session_section(self, section_name: str, phase: str):
    """
    Context manager for timing a pipeline section.

    Automatically correlates timing with power samples to calculate energy consumption.

    Args:
        section_name: Name of the section (e.g., "tokenization", "prefill")
        phase: Pipeline phase (pre_inference, prefill, decode, post_inference)

    Example:
        with session.section("tokenization", "pre_inference"):
            tokens = tokenize(prompt)
    """
    @contextmanager
    def section_context():
        start_time = time.time()
        start_relative_ms = (start_time - self.start_time) * 1000.0

        logger.debug(f"Starting section {phase}/{section_name}")

        # Synchronize if using MPS (Apple Silicon)
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        except ImportError:
            pass

        try:
            yield

        finally:
            # Synchronize again after section
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
            except ImportError:
                pass

            end_time = time.time()
            end_relative_ms = (end_time - self.start_time) * 1000.0
            duration_ms = (end_time - start_time) * 1000.0

            # Calculate energy for this section from power samples
            energy_mj = None
            avg_power_mw = None
            section_power_samples = []

            if self.power_monitor:
                samples = self.power_monitor.get_samples()

                # Find samples within this section's time range
                for sample in samples:
                    if start_relative_ms <= sample.relative_time_ms <= end_relative_ms:
                        section_power_samples.append(sample)

                # Calculate energy from samples in this section
                if len(section_power_samples) > 1:
                    section_energy = 0.0
                    total_power_sum = 0.0

                    for i in range(len(section_power_samples) - 1):
                        time_interval_ms = section_power_samples[i + 1].relative_time_ms - section_power_samples[i].relative_time_ms
                        avg_power = (section_power_samples[i].total_power_mw + section_power_samples[i + 1].total_power_mw) / 2.0
                        section_energy += avg_power * time_interval_ms / 1000.0
                        total_power_sum += avg_power

                    energy_mj = section_energy
                    avg_power_mw = total_power_sum / (len(section_power_samples) - 1)
                elif len(section_power_samples) == 1:
                    # Single sample, estimate energy
                    energy_mj = section_power_samples[0].total_power_mw * duration_ms / 1000.0
                    avg_power_mw = section_power_samples[0].total_power_mw

            # Create section timing record
            section_timing = SectionTiming(
                phase=phase,
                section_name=section_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                energy_mj=energy_mj,
                avg_power_mw=avg_power_mw,
                power_samples=section_power_samples
            )

            # Store in session
            self.sections.append(section_timing)

            logger.debug(f"Completed section {phase}/{section_name}: {duration_ms:.2f}ms, {energy_mj:.2f}mJ" if energy_mj else f"Completed section {phase}/{section_name}: {duration_ms:.2f}ms")

    return section_context()


# Attach section method to ProfilingSession
ProfilingSession.section = _session_section
