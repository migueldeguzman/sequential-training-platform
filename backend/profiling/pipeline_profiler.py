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
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from .power_monitor import PowerMonitor, PowerSample
from .layer_profiler import LayerProfiler, ComponentTiming
from .deep_profiler import DeepAttentionProfiler, AttentionOperationMetrics, MLPOperationMetrics, LayerNormOperationMetrics
from .database import ProfileDatabase

logger = logging.getLogger(__name__)


# WebSocket callback types for streaming profiling events
PowerSampleCallback = Optional[callable]  # Callback signature: callback(sample: PowerSample)
SectionEventCallback = Optional[callable]  # Callback signature: callback(event_type: str, phase: str, section_name: str, timestamp: float, data: dict)
TokenCompleteCallback = Optional[callable]  # Callback signature: callback(token_data: dict)
LayerMetricsCallback = Optional[callable]  # Callback signature: callback(layer_metrics_data: dict)
ComponentMetricsCallback = Optional[callable]  # Callback signature: callback(component_metrics_data: dict)
InferenceCompleteCallback = Optional[callable]  # Callback signature: callback(inference_complete_data: dict)


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
    electricity_price_per_kwh: float = 0.12
    carbon_intensity_g_per_kwh: float = 400.0
    inference_engine: Optional[str] = None

    # Collected data during the session
    sections: List[SectionTiming] = field(default_factory=list)
    response: Optional[str] = None
    baseline_metrics: Optional[Dict[str, float]] = None

    # References to profiling components
    power_monitor: Optional[PowerMonitor] = None
    layer_profiler: Optional[LayerProfiler] = None
    deep_profiler: Optional[DeepAttentionProfiler] = None
    database: Optional[ProfileDatabase] = None

    # WebSocket streaming callbacks
    section_event_callback: SectionEventCallback = None
    token_complete_callback: TokenCompleteCallback = None
    layer_metrics_callback: LayerMetricsCallback = None
    component_metrics_callback: ComponentMetricsCallback = None
    inference_complete_callback: InferenceCompleteCallback = None


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
        database: Optional[ProfileDatabase] = None,
        power_sample_callback: PowerSampleCallback = None,
        section_event_callback: SectionEventCallback = None,
        token_complete_callback: TokenCompleteCallback = None,
        layer_metrics_callback: LayerMetricsCallback = None,
        component_metrics_callback: ComponentMetricsCallback = None,
        inference_complete_callback: InferenceCompleteCallback = None
    ):
        """
        Initialize the inference pipeline profiler.

        Args:
            power_monitor: PowerMonitor instance for system power sampling
            layer_profiler: LayerProfiler instance for layer/component metrics
            deep_profiler: DeepAttentionProfiler instance for operation-level metrics (optional)
            database: ProfileDatabase instance for storing profiling data
            power_sample_callback: Optional callback for streaming power samples (for WebSocket)
            section_event_callback: Optional callback for streaming section events (for WebSocket)
            token_complete_callback: Optional callback for streaming token completion events (for WebSocket)
            layer_metrics_callback: Optional callback for streaming layer metrics (for WebSocket)
            component_metrics_callback: Optional callback for streaming component metrics (for WebSocket)
            inference_complete_callback: Optional callback for streaming inference completion (for WebSocket)
        """
        self.power_monitor = power_monitor
        self.layer_profiler = layer_profiler
        self.deep_profiler = deep_profiler
        self.database = database
        self.power_sample_callback = power_sample_callback
        self.section_event_callback = section_event_callback
        self.token_complete_callback = token_complete_callback
        self.layer_metrics_callback = layer_metrics_callback
        self.component_metrics_callback = component_metrics_callback
        self.inference_complete_callback = inference_complete_callback

        # Current active session
        self._current_session: Optional[ProfilingSession] = None
        self._streaming_thread: Optional[threading.Thread] = None
        self._streaming_active = False

        logger.info("Initialized InferencePipelineProfiler")

    def _stream_power_samples(self) -> None:
        """
        Background thread to stream power samples via callback.

        Polls PowerMonitor for new samples at 100ms intervals and
        invokes the callback with each new sample.
        """
        if not self.power_monitor or not self.power_sample_callback:
            return

        last_sample_count = 0

        while self._streaming_active:
            try:
                # Get all samples collected so far
                if not self.power_monitor:
                    time.sleep(0.1)
                    continue

                samples = self.power_monitor.get_samples()

                # Stream any new samples
                if len(samples) > last_sample_count:
                    for sample in samples[last_sample_count:]:
                        try:
                            self.power_sample_callback(sample)
                        except Exception as e:
                            logger.error(f"Error in power sample callback: {e}")

                    last_sample_count = len(samples)

                # Wait for next check (100ms interval)
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in power sample streaming: {e}")
                break

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
        tags: Optional[str] = None,
        electricity_price_per_kwh: float = 0.12,
        carbon_intensity_g_per_kwh: float = 400.0,
        inference_engine: Optional[str] = None,
        model=None
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
            electricity_price_per_kwh: Cost of electricity in USD per kWh (default: $0.12)
            carbon_intensity_g_per_kwh: Carbon intensity in grams CO2 per kWh (default: 400g for US grid)
            inference_engine: Optional inference engine name (auto-detected if None)
            model: Optional model object for engine detection

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

        # Auto-detect inference engine if not provided
        if inference_engine is None:
            from profiling.engine_detector import detect_inference_engine
            inference_engine = detect_inference_engine(model)
            logger.info(f"Auto-detected inference engine: {inference_engine}")

        logger.info(f"Starting profiling run {run_id} for model {model_name} (engine: {inference_engine})")

        # Create session object
        session = ProfilingSession(
            run_id=run_id,
            start_time=start_time,
            prompt=prompt,
            model_name=model_name,
            profiling_depth=profiling_depth,
            experiment_name=experiment_name,
            tags=tags,
            electricity_price_per_kwh=electricity_price_per_kwh,
            carbon_intensity_g_per_kwh=carbon_intensity_g_per_kwh,
            inference_engine=inference_engine,
            power_monitor=self.power_monitor,
            layer_profiler=self.layer_profiler,
            deep_profiler=self.deep_profiler,
            database=self.database,
            section_event_callback=self.section_event_callback,
            token_complete_callback=self.token_complete_callback,
            layer_metrics_callback=self.layer_metrics_callback,
            component_metrics_callback=self.component_metrics_callback,
            inference_complete_callback=self.inference_complete_callback
        )

        # Store as current session
        self._current_session = session

        # Start power monitoring
        baseline_metrics = None
        if self.power_monitor:
            try:
                self.power_monitor.start()
                logger.info("Power monitoring started")

                # Measure idle baseline power before inference
                try:
                    baseline_metrics = self.power_monitor.measure_idle_baseline(duration_seconds=2.0)
                    logger.info(f"Idle baseline measured: {baseline_metrics['baseline_power_mw']:.2f} mW")
                except Exception as e:
                    logger.warning(f"Failed to measure idle baseline: {e}")
                    baseline_metrics = None

                # Start streaming thread if callback is provided
                if self.power_sample_callback:
                    self._streaming_active = True
                    self._streaming_thread = threading.Thread(
                        target=self._stream_power_samples,
                        daemon=True,
                        name="PowerSampleStreaming"
                    )
                    self._streaming_thread.start()
                    logger.info("Power sample streaming started")
            except Exception as e:
                logger.error(f"Failed to start power monitoring: {e}")

        # Store baseline in session for later database save
        session.baseline_metrics = baseline_metrics

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
            # Stop streaming thread
            if self._streaming_active:
                self._streaming_active = False
                if self._streaming_thread and self._streaming_thread.is_alive():
                    self._streaming_thread.join(timeout=1)
                logger.info("Power sample streaming stopped")

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

            # Emit inference_complete event
            if self.inference_complete_callback:
                try:
                    self._emit_inference_complete_event(session)
                except Exception as e:
                    logger.error(f"Failed to emit inference_complete event: {e}")

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

        # Calculate prefill vs decode energy from sections
        prefill_energy_mj = 0.0
        decode_energy_mj = 0.0
        input_token_count = 0
        output_token_count = 0

        for section in session.sections:
            if section.phase == "prefill" and section.energy_mj:
                prefill_energy_mj += section.energy_mj
                # Count input tokens from prefill phase (usually one section for all input tokens)
                if "prefill" in section.section_name or "embedding" in section.section_name:
                    input_token_count = 1  # Will be updated below
            elif section.phase == "decode" and section.energy_mj:
                decode_energy_mj += section.energy_mj
                # Count output tokens from decode phase
                if "token_" in section.section_name:
                    output_token_count += 1

        # Calculate per-token energy metrics
        energy_per_input_token_mj = prefill_energy_mj / input_token_count if input_token_count > 0 else 0.0
        energy_per_output_token_mj = decode_energy_mj / output_token_count if output_token_count > 0 else 0.0

        # Calculate ratio (paper shows output tokens are ~11x more energy than input)
        input_output_energy_ratio = energy_per_output_token_mj / energy_per_input_token_mj if energy_per_input_token_mj > 0 else 0.0

        # Calculate tokens per second
        token_count = input_token_count + output_token_count
        tokens_per_second = token_count / (total_duration_ms / 1000.0) if total_duration_ms > 0 else 0.0

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
            profiling_depth=session.profiling_depth,
            electricity_price_per_kwh=session.electricity_price_per_kwh,
            carbon_intensity_g_per_kwh=session.carbon_intensity_g_per_kwh,
            inference_engine=session.inference_engine
        )

        # Get peak power metrics
        peak_power_metrics = self.power_monitor.get_peak_power() if self.power_monitor else {}

        # Calculate cost and carbon emissions (EP-089)
        cost_usd = None
        co2_grams = None
        if total_energy_mj is not None and total_energy_mj > 0:
            # Convert mJ to kWh: mJ -> J -> Wh -> kWh
            # 1 mJ = 0.001 J
            # 1 Wh = 3600 J
            # 1 kWh = 1000 Wh
            total_energy_kwh = total_energy_mj / 3600000.0

            # Calculate cost: energy (kWh) × price ($/kWh)
            cost_usd = total_energy_kwh * session.electricity_price_per_kwh

            # Calculate CO2 emissions: energy (kWh) × carbon intensity (g/kWh)
            co2_grams = total_energy_kwh * session.carbon_intensity_g_per_kwh

            logger.debug(f"Calculated cost: ${cost_usd:.6f}, CO2: {co2_grams:.2f}g for run {session.run_id}")

        # Update run with calculated metrics
        self.database.update_run_metrics(
            run_id=session.run_id,
            total_duration_ms=total_duration_ms,
            total_energy_mj=total_energy_mj,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            prefill_energy_mj=prefill_energy_mj,
            decode_energy_mj=decode_energy_mj,
            energy_per_input_token_mj=energy_per_input_token_mj,
            energy_per_output_token_mj=energy_per_output_token_mj,
            input_output_energy_ratio=input_output_energy_ratio,
            peak_power_mw=peak_power_metrics.get('peak_power_mw'),
            peak_power_cpu_mw=peak_power_metrics.get('peak_power_cpu_mw'),
            peak_power_gpu_mw=peak_power_metrics.get('peak_power_gpu_mw'),
            peak_power_ane_mw=peak_power_metrics.get('peak_power_ane_mw'),
            peak_power_dram_mw=peak_power_metrics.get('peak_power_dram_mw'),
            peak_power_timestamp_ms=peak_power_metrics.get('peak_power_timestamp_ms'),
            baseline_power_mw=session.baseline_metrics.get('baseline_power_mw') if session.baseline_metrics else None,
            baseline_cpu_power_mw=session.baseline_metrics.get('baseline_cpu_power_mw') if session.baseline_metrics else None,
            baseline_gpu_power_mw=session.baseline_metrics.get('baseline_gpu_power_mw') if session.baseline_metrics else None,
            baseline_ane_power_mw=session.baseline_metrics.get('baseline_ane_power_mw') if session.baseline_metrics else None,
            baseline_dram_power_mw=session.baseline_metrics.get('baseline_dram_power_mw') if session.baseline_metrics else None,
            baseline_sample_count=session.baseline_metrics.get('baseline_sample_count') if session.baseline_metrics else None,
            cost_usd=cost_usd,
            co2_grams=co2_grams,
            status="completed"
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

        # Save tokens with metrics
        # Note: Token-level saving is phase-specific and should be done during decode phase
        # For now, we collect decode sections which represent tokens
        token_sections = [s for s in session.sections if s.phase == "decode" and "token_" in s.section_name]
        for idx, token_section in enumerate(token_sections):
            # Extract token index from section name (e.g., "decode_token_0" -> 0)
            try:
                token_index = int(token_section.section_name.split("_")[-1])
            except (IndexError, ValueError):
                token_index = idx

            # Add token with basic metrics
            # Note: token_text and layer metrics will be added in future enhancements
            token_id = self.database.add_token(
                run_id=session.run_id,
                token_index=token_index,
                token_text=None,  # Will be populated when decode loop stores token text
                phase="decode",
                start_time_ms=token_section.start_time * 1000.0,
                end_time_ms=token_section.end_time * 1000.0,
                duration_ms=token_section.duration_ms,
                energy_mj=token_section.energy_mj,
                avg_power_mw=token_section.avg_power_mw,
                is_input_token=False  # Decode tokens are output tokens
            )

        # Save layer metrics (per-layer per-token profiling data)
        if self.layer_profiler:
            layer_timings = self.layer_profiler.get_timings()
            if layer_timings:
                layer_metric_dicts = []
                for timing in layer_timings:
                    layer_metric_dicts.append({
                        "layer_index": timing.layer_index,
                        "component_name": timing.component_name,
                        "duration_ms": timing.duration_ms,
                        "activation_mean": timing.activation_mean,
                        "activation_std": timing.activation_std,
                        "activation_max": timing.activation_max,
                        "activation_sparsity": timing.activation_sparsity
                    })

                # Batch insert layer metrics
                # Note: These are associated with the run, not individual tokens
                # For per-token layer metrics, would need token_id foreign key
                # For now, we save them at run level
                self.database.add_layer_metrics(
                    run_id=session.run_id,
                    token_id=None,  # Run-level metrics (will enhance for per-token in future)
                    metrics=layer_metric_dicts
                )
                logger.debug(f"Saved {len(layer_metric_dicts)} layer metrics")

        # Save component metrics (aggregated by component type)
        # Aggregate layer timings by component name
        component_aggregates = {}
        if self.layer_profiler:
            layer_timings = self.layer_profiler.get_timings()
            for timing in layer_timings:
                comp_name = timing.component_name
                if comp_name not in component_aggregates:
                    component_aggregates[comp_name] = {
                        "total_duration_ms": 0.0,
                        "count": 0,
                        "sum_activation_mean": 0.0,
                        "sum_activation_std": 0.0,
                        "sum_activation_max": 0.0,
                        "sum_activation_sparsity": 0.0
                    }

                agg = component_aggregates[comp_name]
                agg["total_duration_ms"] += timing.duration_ms
                agg["count"] += 1
                if timing.activation_mean is not None:
                    agg["sum_activation_mean"] += timing.activation_mean
                if timing.activation_std is not None:
                    agg["sum_activation_std"] += timing.activation_std
                if timing.activation_max is not None:
                    agg["sum_activation_max"] += timing.activation_max
                if timing.activation_sparsity is not None:
                    agg["sum_activation_sparsity"] += timing.activation_sparsity

        # Convert aggregates to component metrics and batch insert
        if component_aggregates:
            component_metric_dicts = []
            for comp_name, agg in component_aggregates.items():
                count = agg["count"]
                component_metric_dicts.append({
                    "component_name": comp_name,
                    "total_duration_ms": agg["total_duration_ms"],
                    "call_count": count,
                    "avg_duration_ms": agg["total_duration_ms"] / count,
                    "avg_activation_mean": agg["sum_activation_mean"] / count if count > 0 else None,
                    "avg_activation_std": agg["sum_activation_std"] / count if count > 0 else None,
                    "avg_activation_max": agg["sum_activation_max"] / count if count > 0 else None,
                    "avg_activation_sparsity": agg["sum_activation_sparsity"] / count if count > 0 else None
                })

            self.database.add_component_metrics(
                run_id=session.run_id,
                layer_metric_id=None,  # Run-level aggregated metrics
                metrics=component_metric_dicts
            )
            logger.debug(f"Saved {len(component_metric_dicts)} component metrics")

        # Save deep operation metrics if deep profiling was enabled
        if self.deep_profiler and session.profiling_depth == "deep":
            deep_metric_dicts = []

            # Collect attention operation metrics
            attention_metrics = self.deep_profiler.get_metrics()
            for metric in attention_metrics:
                deep_metric_dicts.append({
                    "operation_type": "attention",
                    "operation_name": "attention_forward",
                    "duration_ms": metric.total_time,
                    "qk_matmul_time": metric.qk_matmul_time,
                    "scale_time": metric.scale_time,
                    "mask_time": metric.mask_time,
                    "softmax_time": metric.softmax_time,
                    "value_matmul_time": metric.value_matmul_time,
                    "attention_entropy": metric.avg_attention_entropy,
                    "attention_sparsity": metric.avg_attention_sparsity,
                    "max_attention_weight": metric.avg_max_attention_weight
                })

            # Collect MLP operation metrics
            mlp_metrics = self.deep_profiler.get_mlp_metrics()
            for metric in mlp_metrics:
                deep_metric_dicts.append({
                    "operation_type": "mlp",
                    "operation_name": "mlp_forward",
                    "duration_ms": metric.total_time,
                    "gate_proj_time": metric.gate_proj_time,
                    "up_proj_time": metric.up_proj_time,
                    "gate_up_mult_time": metric.gate_up_mult_time,
                    "down_proj_time": metric.down_proj_time,
                    "activation_kill_ratio": metric.activation_kill_ratio
                })

            # Collect LayerNorm operation metrics
            layernorm_metrics = self.deep_profiler.get_layernorm_metrics()
            for metric in layernorm_metrics:
                deep_metric_dicts.append({
                    "operation_type": "layernorm",
                    "operation_name": "layernorm_forward",
                    "duration_ms": metric.total_time,
                    "mean_time": metric.mean_time,
                    "variance_time": metric.variance_time,
                    "normalization_time": metric.normalization_time,
                    "scale_shift_time": metric.scale_shift_time,
                    "variance_ratio": metric.variance_ratio
                })

            # Batch insert all deep operation metrics
            if deep_metric_dicts:
                self.database.add_deep_operation_metrics(
                    run_id=session.run_id,
                    component_metric_id=None,  # Run-level metrics
                    metrics=deep_metric_dicts
                )
                logger.debug(f"Saved {len(deep_metric_dicts)} deep operation metrics")

        logger.info(f"Saved run {session.run_id} to database: "
                   f"{len(power_samples)} power samples, "
                   f"{len(session.sections)} sections, "
                   f"{len(token_sections)} tokens, "
                   f"{len(component_aggregates)} component types")

    def _emit_inference_complete_event(self, session: ProfilingSession) -> None:
        """
        Emit inference_complete event via WebSocket callback.

        This is called at the end of a profiling run to stream final summary statistics.

        Args:
            session: Completed ProfilingSession with all collected data
        """
        if not self.inference_complete_callback:
            return

        # Calculate total duration and energy
        end_time = time.time()
        total_duration_ms = (end_time - session.start_time) * 1000.0

        # Calculate total energy from power samples
        total_energy_mj = 0.0
        power_samples = []

        if self.power_monitor:
            samples = self.power_monitor.get_samples()
            power_samples = samples

            # Calculate energy: integrate power over time
            for i in range(len(samples) - 1):
                time_interval_ms = samples[i + 1].relative_time_ms - samples[i].relative_time_ms
                avg_power_mw = (samples[i].total_power_mw + samples[i + 1].total_power_mw) / 2.0
                energy_mj = avg_power_mw * time_interval_ms / 1000.0
                total_energy_mj += energy_mj

        # Calculate token count from decode sections
        token_sections = [s for s in session.sections if s.phase == "decode" and "token_" in s.section_name]
        token_count = len(token_sections)

        # Calculate tokens per second
        tokens_per_second = (token_count / total_duration_ms * 1000.0) if total_duration_ms > 0 and token_count > 0 else 0.0

        # Build summary statistics
        inference_complete_data = {
            "run_id": session.run_id,
            "total_duration_ms": total_duration_ms,
            "total_energy_mj": total_energy_mj,
            "token_count": token_count,
            "tokens_per_second": tokens_per_second,
            "summary_statistics": {
                "num_power_samples": len(power_samples),
                "num_sections": len(session.sections),
                "num_tokens": token_count,
                "avg_power_mw": (total_energy_mj / total_duration_ms * 1000.0) if total_duration_ms > 0 else 0.0,
                "energy_per_token_mj": (total_energy_mj / token_count) if token_count > 0 else None
            },
            "timestamp": end_time
        }

        # Emit via callback
        try:
            self.inference_complete_callback(inference_complete_data)
            logger.info(f"Emitted inference_complete event for run {session.run_id}: "
                       f"{total_duration_ms:.2f}ms, {total_energy_mj:.2f}mJ, {token_count} tokens")
        except Exception as e:
            logger.error(f"Error in inference_complete callback: {e}")

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

    # =========================================================================
    # Decode Phase Profiling Methods
    # =========================================================================

    def profile_decode_token(
        self,
        session: ProfilingSession,
        model: Any,
        input_ids: Any,
        token_index: int,
        **model_kwargs
    ) -> Any:
        """
        Profile generation of a single decode token.

        Wraps one iteration of the decode loop with section timing and
        captures per-token layer metrics.

        Args:
            session: Active profiling session
            model: The language model
            input_ids: Current input token IDs
            token_index: Index of the token being generated (0-based)
            **model_kwargs: Additional model arguments (past_key_values, attention_mask, etc.)

        Returns:
            Tuple of (next_token_id, updated_model_kwargs)

        Example:
            for token_idx in range(max_tokens):
                next_token, model_kwargs = profiler.profile_decode_token(
                    session, model, input_ids, token_idx, **model_kwargs
                )
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        """
        section_name = f"decode_token_{token_index}"

        # Reset layer profiler for fresh per-token metrics
        if session.layer_profiler:
            session.layer_profiler.reset()

        with session.section(section_name, "decode"):
            # Run model forward pass for this token
            output = model(input_ids, return_dict=True, **model_kwargs)

            # Get next token logits
            next_token_logits = output.logits[:, -1, :]

            # Update model kwargs (past_key_values, etc.)
            updated_kwargs = model_kwargs.copy()
            if hasattr(output, 'past_key_values') and output.past_key_values is not None:
                updated_kwargs['past_key_values'] = output.past_key_values

            # Get layer metrics for this token
            if session.layer_profiler:
                layer_timings = session.layer_profiler.get_timings()
                logger.debug(f"Token {token_index}: Captured {len(layer_timings)} layer timings")

            return next_token_logits, updated_kwargs

    def emit_token_complete_event(
        self,
        session: ProfilingSession,
        token_index: int,
        token_text: str,
        duration_ms: float,
        energy_mj: Optional[float] = None,
        avg_power_mw: Optional[float] = None
    ):
        """
        Emit a token_complete event via WebSocket callback.

        This should be called after a decode token is generated to stream
        real-time token metrics to connected clients.

        Args:
            session: Active profiling session
            token_index: Index of the generated token (0-based)
            token_text: Decoded text of the token
            duration_ms: Time taken to generate this token
            energy_mj: Energy consumed for this token (if available)
            avg_power_mw: Average power during token generation (if available)

        Example:
            for token_idx in range(max_tokens):
                # ... generate token ...
                profiler.emit_token_complete_event(
                    session, token_idx, token_text, duration_ms, energy_mj, avg_power_mw
                )
        """
        if not session.token_complete_callback:
            return

        # Get current power snapshot
        power_snapshot = None
        if session.power_monitor:
            current_sample = session.power_monitor.get_current()
            if current_sample:
                power_snapshot = {
                    "cpu_power_mw": current_sample.cpu_power_mw,
                    "gpu_power_mw": current_sample.gpu_power_mw,
                    "ane_power_mw": current_sample.ane_power_mw,
                    "dram_power_mw": current_sample.dram_power_mw,
                    "total_power_mw": current_sample.total_power_mw
                }

        # Get layer metrics summary for this token
        layer_metrics_summary = None
        layer_metrics = []  # Full layer-by-layer data for LiveLayerHeatmap
        if session.layer_profiler:
            layer_timings = session.layer_profiler.get_timings()
            if layer_timings:
                # Calculate summary statistics
                total_layer_time = sum(t.duration_ms for t in layer_timings)
                avg_activation_mean = sum(t.activation_mean for t in layer_timings if t.activation_mean is not None) / len(layer_timings) if layer_timings else None

                layer_metrics_summary = {
                    "num_layers": len(layer_timings),
                    "total_duration_ms": total_layer_time,
                    "avg_activation_mean": avg_activation_mean,
                    "components": {}
                }

                # Group by component type for summary
                for timing in layer_timings:
                    comp_name = timing.component_name
                    if comp_name not in layer_metrics_summary["components"]:
                        layer_metrics_summary["components"][comp_name] = {
                            "count": 0,
                            "total_duration_ms": 0.0
                        }
                    layer_metrics_summary["components"][comp_name]["count"] += 1
                    layer_metrics_summary["components"][comp_name]["total_duration_ms"] += timing.duration_ms

                # Build full layer metrics array grouped by layer
                # Group timings by layer_idx
                layers_dict = {}
                for timing in layer_timings:
                    layer_idx = timing.layer_idx
                    if layer_idx not in layers_dict:
                        layers_dict[layer_idx] = {
                            "layer_index": layer_idx,
                            "components": []
                        }

                    # Add component data
                    component = {
                        "component_name": timing.component_name,
                        "duration_ms": timing.duration_ms,
                        "activation_mean": timing.activation_mean or 0.0,
                        "activation_std": timing.activation_std or 0.0,
                        "activation_max": timing.activation_max or 0.0,
                        "sparsity": timing.activation_sparsity or 0.0
                    }
                    layers_dict[layer_idx]["components"].append(component)

                # Convert to sorted list by layer_index
                layer_metrics = [layers_dict[idx] for idx in sorted(layers_dict.keys())]

        # Build token complete message
        token_data = {
            "token_position": token_index,  # Frontend expects token_position
            "token_text": token_text,
            "duration_ms": duration_ms,
            "energy_mj": energy_mj,
            "power_snapshot_mw": avg_power_mw,  # Frontend expects power_snapshot_mw
            "power_snapshot": power_snapshot,
            "layer_metrics_summary": layer_metrics_summary,
            "layer_metrics": layer_metrics,  # Add full layer-by-layer data
            "timestamp": time.time()
        }

        # Emit via callback
        try:
            session.token_complete_callback(token_data)
            logger.debug(f"Emitted token_complete event for token {token_index}: {token_text}")
        except Exception as e:
            logger.error(f"Error in token_complete callback: {e}")

    def emit_layer_metrics_event(
        self,
        session: ProfilingSession,
        token_index: int,
        layer_index: int,
        layer_name: str
    ) -> None:
        """
        Emit layer metrics event via WebSocket callback.

        Streams detailed layer metrics for a specific layer during token processing.
        This is called after each token if the client requests detailed metrics.

        Args:
            session: Active profiling session
            token_index: Index of the token being processed
            layer_index: Index of the layer (0 to N-1)
            layer_name: Name/identifier of the layer

        Example:
            for layer_idx in range(num_layers):
                # ... process layer ...
                profiler.emit_layer_metrics_event(
                    session, token_idx, layer_idx, f"layer_{layer_idx}"
                )
        """
        if not session.layer_metrics_callback:
            return

        if not session.layer_profiler:
            return

        # Get layer timings for this token
        layer_timings = session.layer_profiler.get_timings()
        if not layer_timings:
            return

        # Filter to timings for this specific layer
        layer_specific_timings = [
            t for t in layer_timings
            if layer_name in t.module_path or f"layer.{layer_index}" in t.module_path
        ]

        if not layer_specific_timings:
            return

        # Aggregate metrics for this layer
        total_duration = sum(t.duration_ms for t in layer_specific_timings)

        # Calculate activation statistics
        activation_means = [t.activation_mean for t in layer_specific_timings if t.activation_mean is not None]
        activation_stds = [t.activation_std for t in layer_specific_timings if t.activation_std is not None]
        activation_maxs = [t.activation_max for t in layer_specific_timings if t.activation_max is not None]
        activation_sparsities = [t.activation_sparsity for t in layer_specific_timings if t.activation_sparsity is not None]

        # Build layer metrics message
        layer_metrics_data = {
            "token_index": token_index,
            "layer_index": layer_index,
            "layer_name": layer_name,
            "total_duration_ms": total_duration,
            "num_components": len(layer_specific_timings),
            "activation_stats": {
                "mean_avg": sum(activation_means) / len(activation_means) if activation_means else None,
                "std_avg": sum(activation_stds) / len(activation_stds) if activation_stds else None,
                "max_avg": sum(activation_maxs) / len(activation_maxs) if activation_maxs else None,
                "sparsity_avg": sum(activation_sparsities) / len(activation_sparsities) if activation_sparsities else None
            },
            "components": [
                {
                    "component_name": t.component_name,
                    "duration_ms": t.duration_ms,
                    "activation_mean": t.activation_mean,
                    "activation_std": t.activation_std,
                    "activation_max": t.activation_max,
                    "activation_sparsity": t.activation_sparsity
                }
                for t in layer_specific_timings
            ],
            "timestamp": time.time()
        }

        # Emit via callback
        try:
            session.layer_metrics_callback(layer_metrics_data)
            logger.debug(f"Emitted layer_metrics event for token {token_index}, layer {layer_index}")
        except Exception as e:
            logger.error(f"Error in layer_metrics callback: {e}")

    def emit_component_metrics_event(
        self,
        session: ProfilingSession,
        token_index: int,
        layer_index: int,
        component_name: str,
        component_timing: ComponentTiming
    ) -> None:
        """
        Emit component metrics event via WebSocket callback.

        Streams detailed component metrics for a specific component during token processing.
        This provides the most granular level of profiling data.

        Args:
            session: Active profiling session
            token_index: Index of the token being processed
            layer_index: Index of the layer containing this component
            component_name: Name of the component (e.g., 'q_proj', 'k_proj', 'mlp.gate_proj')
            component_timing: ComponentTiming object with metrics

        Example:
            for component_timing in layer_timings:
                profiler.emit_component_metrics_event(
                    session, token_idx, layer_idx,
                    component_timing.component_name, component_timing
                )
        """
        if not session.component_metrics_callback:
            return

        # Build component metrics message
        component_metrics_data = {
            "token_index": token_index,
            "layer_index": layer_index,
            "component_name": component_name,
            "module_path": component_timing.module_path,
            "duration_ms": component_timing.duration_ms,
            "activation_stats": {
                "mean": component_timing.activation_mean,
                "std": component_timing.activation_std,
                "max": component_timing.activation_max,
                "sparsity": component_timing.activation_sparsity
            },
            "timestamp": time.time()
        }

        # Emit via callback
        try:
            session.component_metrics_callback(component_metrics_data)
            logger.debug(f"Emitted component_metrics event for token {token_index}, layer {layer_index}, component {component_name}")
        except Exception as e:
            logger.error(f"Error in component_metrics callback: {e}")

    def profile_decode_embedding(
        self,
        session: ProfilingSession,
        embedding_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile token embedding lookup for decode.

        Args:
            session: Active profiling session
            embedding_func: Function to get embeddings (e.g., model.embed_tokens)
            *args, **kwargs: Arguments to embedding function

        Returns:
            Embedding tensor

        Example:
            embeddings = profiler.profile_decode_embedding(
                session, model.embed_tokens, prev_token_ids
            )
        """
        with session.section("embedding", "decode"):
            return embedding_func(*args, **kwargs)

    def profile_decode_position(
        self,
        session: ProfilingSession,
        position_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile position embedding for decode.

        Args:
            session: Active profiling session
            position_func: Function to add position embeddings
            *args, **kwargs: Arguments to position function

        Returns:
            Result with position embeddings

        Example:
            hidden = profiler.profile_decode_position(
                session, add_positional_encoding, hidden, position
            )
        """
        with session.section("position", "decode"):
            return position_func(*args, **kwargs)

    def profile_decode_layers(
        self,
        session: ProfilingSession,
        layers_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile transformer layers during decode.

        LayerProfiler hooks automatically capture per-layer metrics.

        Args:
            session: Active profiling session
            layers_func: Function to run transformer layers
            *args, **kwargs: Arguments to layers function

        Returns:
            Output from transformer layers

        Example:
            output = profiler.profile_decode_layers(
                session, model.model.layers, hidden_states
            )
        """
        with session.section("layers", "decode"):
            return layers_func(*args, **kwargs)

    def profile_decode_lm_head(
        self,
        session: ProfilingSession,
        lm_head_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile LM head projection during decode.

        Args:
            session: Active profiling session
            lm_head_func: Language model head projection function
            *args, **kwargs: Arguments to LM head

        Returns:
            Vocabulary logits

        Example:
            logits = profiler.profile_decode_lm_head(
                session, model.lm_head, hidden_states
            )
        """
        with session.section("lm_head", "decode"):
            return lm_head_func(*args, **kwargs)

    def profile_decode_sampling(
        self,
        session: ProfilingSession,
        sampling_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile sampling operation during decode.

        Includes temperature scaling, top_k, top_p, and final token selection.

        Args:
            session: Active profiling session
            sampling_func: Function to sample next token from logits
            *args, **kwargs: Arguments to sampling function

        Returns:
            Selected token ID

        Example:
            next_token = profiler.profile_decode_sampling(
                session, sample_token, logits, temperature=0.8, top_p=0.9
            )
        """
        with session.section("sampling", "decode"):
            return sampling_func(*args, **kwargs)

    def profile_decode_kv_cache_append(
        self,
        session: ProfilingSession,
        append_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile KV-cache append during decode.

        Args:
            session: Active profiling session
            append_func: Function to append new key/value to cache
            *args, **kwargs: Arguments to append function

        Returns:
            Updated KV cache

        Example:
            cache = profiler.profile_decode_kv_cache_append(
                session, append_to_cache, cache, new_key, new_value
            )
        """
        with session.section("kv_cache_append", "decode"):
            return append_func(*args, **kwargs)

    def profile_decode_loop(
        self,
        session: ProfilingSession,
        model: Any,
        input_ids: Any,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        **model_kwargs
    ) -> Any:
        """
        Profile complete decode loop with automatic per-token profiling.

        Convenience method that profiles token generation loop with section
        breakdown per token. LayerProfiler is automatically reset between tokens.

        Args:
            session: Active profiling session
            model: The language model
            input_ids: Initial input token IDs (after prefill)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **model_kwargs: Additional model arguments (past_key_values, etc.)

        Returns:
            Generated token IDs tensor

        Example:
            with profiler.run(prompt="Hello", model_name="llama-7b") as session:
                # Prefill
                prefill_output = profiler.profile_prefill(session, model, input_ids)

                # Decode
                generated = profiler.profile_decode_loop(
                    session, model, input_ids, max_new_tokens=50,
                    past_key_values=prefill_output.past_key_values
                )
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch is required for decode profiling")

        generated_ids = input_ids.clone()

        for token_idx in range(max_new_tokens):
            # Reset layer profiler for fresh per-token metrics
            if session.layer_profiler:
                session.layer_profiler.reset()

            section_name = f"token_{token_idx}"

            with session.section(section_name, "decode"):
                # Forward pass
                output = model(generated_ids, return_dict=True, **model_kwargs)

                # Get next token logits
                next_token_logits = output.logits[:, -1, :] / temperature

                # Apply sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Update model kwargs
                if hasattr(output, 'past_key_values') and output.past_key_values is not None:
                    model_kwargs['past_key_values'] = output.past_key_values

                # Get layer metrics
                if session.layer_profiler:
                    layer_timings = session.layer_profiler.get_timings()
                    logger.debug(f"Token {token_idx}: Captured {len(layer_timings)} layer timings")

                # Check for EOS token (assuming EOS token ID is 2)
                if next_token.item() == 2:
                    logger.info(f"EOS token generated at position {token_idx}")
                    break

        return generated_ids

    # =========================================================================
    # Post-Inference Phase Profiling Methods
    # =========================================================================

    def profile_tensor_to_cpu(
        self,
        session: ProfilingSession,
        tensor: Any
    ) -> Any:
        """
        Profile tensor transfer to CPU with automatic section timing.

        This captures the time to move output tensors from device memory
        (GPU/ANE/MPS) back to CPU for further processing.

        Args:
            session: Active ProfilingSession from run() context manager
            tensor: Output tensor to transfer to CPU

        Returns:
            Tensor on CPU

        Example:
            with profiler.run(prompt="Hello", model_name="llama-7b") as session:
                # ... inference ...
                output_cpu = profiler.profile_tensor_to_cpu(session, output_tensor)
        """
        with session.section("tensor_to_cpu", "post_inference"):
            result = tensor.to('cpu')
        return result

    def profile_detokenization(
        self,
        session: ProfilingSession,
        tokenizer: Any,
        token_ids: Any
    ) -> str:
        """
        Profile detokenization step with automatic section timing.

        This captures the time to decode token IDs back to text.

        Args:
            session: Active ProfilingSession from run() context manager
            tokenizer: Tokenizer instance
            token_ids: Token IDs to decode

        Returns:
            Decoded text string

        Example:
            with profiler.run(prompt="Hello", model_name="llama-7b") as session:
                # ... inference ...
                text = profiler.profile_detokenization(session, tokenizer, generated_ids)
        """
        with session.section("detokenization", "post_inference"):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    def profile_cleanup(
        self,
        session: ProfilingSession,
        cleanup_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profile memory cleanup operations with automatic section timing.

        This captures the time for any cleanup operations like cache clearing,
        memory release, or garbage collection.

        Args:
            session: Active ProfilingSession from run() context manager
            cleanup_func: Function that performs cleanup operations
            *args: Positional arguments for cleanup_func
            **kwargs: Keyword arguments for cleanup_func

        Returns:
            Result from cleanup_func (if any)

        Example:
            with profiler.run(prompt="Hello", model_name="llama-7b") as session:
                # ... inference ...
                profiler.profile_cleanup(session, torch.cuda.empty_cache)
        """
        with session.section("cleanup", "post_inference"):
            result = cleanup_func(*args, **kwargs) if cleanup_func else None
        return result

    def get_total_inference_energy(self, session: ProfilingSession) -> Dict[str, float]:
        """
        Calculate total inference energy from all profiled sections.

        This aggregates energy consumption across all phases:
        - Pre-inference (tokenization, tensor transfer, KV-cache init)
        - Prefill (prompt processing)
        - Decode (token generation)
        - Post-inference (detokenization, cleanup)

        Args:
            session: Active ProfilingSession with collected section data

        Returns:
            Dictionary containing:
                - total_energy_mj: Total energy consumption in millijoules
                - pre_inference_energy_mj: Energy for pre-inference phase
                - prefill_energy_mj: Energy for prefill phase
                - decode_energy_mj: Energy for decode phase
                - post_inference_energy_mj: Energy for post-inference phase
                - total_duration_ms: Total duration in milliseconds
                - avg_power_mw: Average power draw in milliwatts

        Example:
            with profiler.run(prompt="Hello", model_name="llama-7b") as session:
                # ... complete inference pipeline ...
                energy_stats = profiler.get_total_inference_energy(session)
                print(f"Total energy: {energy_stats['total_energy_mj']:.2f} mJ")
        """
        # Initialize phase energy accumulators
        phase_energy = {
            "pre_inference": 0.0,
            "prefill": 0.0,
            "decode": 0.0,
            "post_inference": 0.0
        }

        # Accumulate energy per phase from all sections
        total_energy_mj = 0.0
        total_duration_ms = 0.0

        for section in session.sections:
            if section.energy_mj is not None:
                phase_energy[section.phase] += section.energy_mj
                total_energy_mj += section.energy_mj
            total_duration_ms += section.duration_ms

        # Calculate average power
        avg_power_mw = (total_energy_mj / total_duration_ms * 1000.0) if total_duration_ms > 0 else 0.0

        return {
            "total_energy_mj": total_energy_mj,
            "pre_inference_energy_mj": phase_energy["pre_inference"],
            "prefill_energy_mj": phase_energy["prefill"],
            "decode_energy_mj": phase_energy["decode"],
            "post_inference_energy_mj": phase_energy["post_inference"],
            "total_duration_ms": total_duration_ms,
            "avg_power_mw": avg_power_mw
        }

    def aggregate_profiling_data(self, session: ProfilingSession) -> Dict[str, Any]:
        """
        Aggregate all profiling data from a completed session.

        This collects and organizes all profiling metrics:
        - Power samples from PowerMonitor
        - Section timings with energy calculations
        - Token metrics with layer breakdowns
        - Component metrics (attention, MLP, layernorm)
        - Deep operation metrics (if deep profiling enabled)

        Args:
            session: Completed ProfilingSession with all collected data

        Returns:
            Dictionary containing complete run data structure:
                - run_metadata: run_id, model, prompt, timestamps, etc.
                - power_samples: List of all power measurements
                - section_timings: List of all pipeline sections with energy
                - layer_metrics: Per-layer timing and activation statistics
                - component_metrics: Per-component breakdowns
                - deep_metrics: Operation-level metrics (if available)
                - energy_summary: Total and per-phase energy breakdown

        Example:
            with profiler.run(prompt="Hello", model_name="llama-7b") as session:
                # ... complete inference ...
                data = profiler.aggregate_profiling_data(session)
                print(f"Total sections: {len(data['section_timings'])}")
                print(f"Total energy: {data['energy_summary']['total_energy_mj']:.2f} mJ")
        """
        logger.info(f"Aggregating profiling data for run {session.run_id}")

        # 1. Collect power samples from PowerMonitor
        power_samples = []
        if session.power_monitor:
            samples = session.power_monitor.get_samples()
            power_samples = [
                {
                    "timestamp_ms": s.relative_time_ms,
                    "cpu_power_mw": s.cpu_power_mw,
                    "gpu_power_mw": s.gpu_power_mw,
                    "ane_power_mw": s.ane_power_mw,
                    "dram_power_mw": s.dram_power_mw,
                    "total_power_mw": s.total_power_mw
                }
                for s in samples
            ]
            logger.debug(f"Collected {len(power_samples)} power samples")

        # 2. Collect all section timings with energy calculations
        section_timings = []
        for section in session.sections:
            section_timings.append({
                "phase": section.phase,
                "section_name": section.section_name,
                "start_time": section.start_time,
                "end_time": section.end_time,
                "duration_ms": section.duration_ms,
                "energy_mj": section.energy_mj,
                "avg_power_mw": section.avg_power_mw,
                "num_power_samples": len(section.power_samples)
            })
        logger.debug(f"Collected {len(section_timings)} section timings")

        # 3. Collect layer metrics from LayerProfiler
        layer_metrics = []
        if session.layer_profiler:
            timings = session.layer_profiler.get_timings()
            for timing in timings:
                layer_metrics.append({
                    "layer_index": timing.layer_index,
                    "component_name": timing.component_name,
                    "duration_ms": timing.duration_ms,
                    "activation_mean": timing.activation_mean,
                    "activation_std": timing.activation_std,
                    "activation_max": timing.activation_max,
                    "activation_sparsity": timing.activation_sparsity
                })
            logger.debug(f"Collected {len(layer_metrics)} layer/component metrics")

        # 4. Collect component metrics (aggregated by component type)
        component_metrics = self._aggregate_component_metrics(layer_metrics)
        logger.debug(f"Aggregated metrics for {len(component_metrics)} component types")

        # 5. Collect deep operation metrics if enabled
        deep_metrics = None
        if session.deep_profiler and session.profiling_depth == "deep":
            deep_metrics = self._aggregate_deep_metrics(session.deep_profiler)
            logger.debug(f"Collected deep operation metrics")

        # 6. Calculate energy per section (already done in section_timings)
        # Energy is calculated in the section() context manager using power * time

        # 7. Build complete run data structure
        run_data = {
            "run_metadata": {
                "run_id": session.run_id,
                "timestamp": datetime.fromtimestamp(session.start_time).isoformat(),
                "model_name": session.model_name,
                "prompt": session.prompt,
                "response": session.response,
                "experiment_name": session.experiment_name,
                "tags": session.tags,
                "profiling_depth": session.profiling_depth,
                "start_time": session.start_time,
                "end_time": time.time()
            },
            "power_samples": power_samples,
            "section_timings": section_timings,
            "layer_metrics": layer_metrics,
            "component_metrics": component_metrics,
            "deep_metrics": deep_metrics,
            "energy_summary": self.get_total_inference_energy(session)
        }

        logger.info(f"Aggregation complete: {len(power_samples)} samples, "
                   f"{len(section_timings)} sections, "
                   f"{len(layer_metrics)} layer metrics")

        return run_data

    def _aggregate_component_metrics(self, layer_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate layer metrics by component type.

        Groups all timings by component name (e.g., q_proj, k_proj, gate_proj)
        and calculates aggregate statistics (total time, average activation stats).

        Args:
            layer_metrics: List of per-layer component metrics

        Returns:
            Dictionary mapping component name to aggregated metrics:
                - total_duration_ms: Sum of all durations for this component
                - count: Number of occurrences
                - avg_duration_ms: Average duration per occurrence
                - avg_activation_mean: Average activation mean across occurrences
                - avg_activation_std: Average activation std across occurrences
                - avg_activation_max: Average activation max across occurrences
                - avg_activation_sparsity: Average sparsity across occurrences
        """
        component_aggregates = {}

        for metric in layer_metrics:
            component_name = metric["component_name"]

            if component_name not in component_aggregates:
                component_aggregates[component_name] = {
                    "total_duration_ms": 0.0,
                    "count": 0,
                    "sum_activation_mean": 0.0,
                    "sum_activation_std": 0.0,
                    "sum_activation_max": 0.0,
                    "sum_activation_sparsity": 0.0
                }

            agg = component_aggregates[component_name]
            agg["total_duration_ms"] += metric["duration_ms"]
            agg["count"] += 1

            if metric["activation_mean"] is not None:
                agg["sum_activation_mean"] += metric["activation_mean"]
            if metric["activation_std"] is not None:
                agg["sum_activation_std"] += metric["activation_std"]
            if metric["activation_max"] is not None:
                agg["sum_activation_max"] += metric["activation_max"]
            if metric["activation_sparsity"] is not None:
                agg["sum_activation_sparsity"] += metric["activation_sparsity"]

        # Calculate averages
        component_metrics = {}
        for component_name, agg in component_aggregates.items():
            count = agg["count"]
            component_metrics[component_name] = {
                "total_duration_ms": agg["total_duration_ms"],
                "count": count,
                "avg_duration_ms": agg["total_duration_ms"] / count,
                "avg_activation_mean": agg["sum_activation_mean"] / count if count > 0 else None,
                "avg_activation_std": agg["sum_activation_std"] / count if count > 0 else None,
                "avg_activation_max": agg["sum_activation_max"] / count if count > 0 else None,
                "avg_activation_sparsity": agg["sum_activation_sparsity"] / count if count > 0 else None
            }

        return component_metrics

    def _aggregate_deep_metrics(self, deep_profiler: DeepAttentionProfiler) -> Dict[str, Any]:
        """
        Aggregate deep operation metrics from DeepAttentionProfiler.

        Collects all operation-level metrics captured during deep profiling:
        - Attention operation timings (QK^T, scale, mask, softmax, value matmul)
        - Attention extra metrics (entropy, sparsity, max weights)
        - MLP operation timings (gate, up, mult, down projections)
        - MLP activation kill ratios
        - LayerNorm operation timings (mean, variance, normalization, scale/shift)
        - LayerNorm variance ratios

        Args:
            deep_profiler: DeepAttentionProfiler with collected metrics

        Returns:
            Dictionary containing aggregated deep metrics:
                - attention_ops: List of attention operation metrics
                - mlp_ops: List of MLP operation metrics
                - layernorm_ops: List of LayerNorm operation metrics
                - summary: Aggregate statistics across all operations
        """
        # Get attention metrics
        attention_metrics = deep_profiler.get_metrics()
        attention_ops = []
        for metric in attention_metrics:
            attention_ops.append({
                "qk_matmul_time": metric.qk_matmul_time,
                "scale_time": metric.scale_time,
                "mask_time": metric.mask_time,
                "softmax_time": metric.softmax_time,
                "value_matmul_time": metric.value_matmul_time,
                "total_time": metric.total_time,
                "attention_entropy_per_head": metric.attention_entropy_per_head,
                "max_attention_weight_per_head": metric.max_attention_weight_per_head,
                "attention_sparsity_per_head": metric.attention_sparsity_per_head,
                "avg_attention_entropy": metric.avg_attention_entropy,
                "avg_max_attention_weight": metric.avg_max_attention_weight,
                "avg_attention_sparsity": metric.avg_attention_sparsity
            })

        # Get MLP metrics
        mlp_metrics = deep_profiler.get_mlp_metrics()
        mlp_ops = []
        for metric in mlp_metrics:
            mlp_ops.append({
                "gate_proj_time": metric.gate_proj_time,
                "up_proj_time": metric.up_proj_time,
                "gate_up_mult_time": metric.gate_up_mult_time,
                "down_proj_time": metric.down_proj_time,
                "total_time": metric.total_time,
                "activation_kill_ratio": metric.activation_kill_ratio
            })

        # Get LayerNorm metrics
        layernorm_metrics = deep_profiler.get_layernorm_metrics()
        layernorm_ops = []
        for metric in layernorm_metrics:
            layernorm_ops.append({
                "mean_time": metric.mean_time,
                "variance_time": metric.variance_time,
                "normalization_time": metric.normalization_time,
                "scale_shift_time": metric.scale_shift_time,
                "total_time": metric.total_time,
                "variance_ratio": metric.variance_ratio
            })

        # Calculate summary statistics
        summary = {
            "num_attention_ops": len(attention_ops),
            "num_mlp_ops": len(mlp_ops),
            "num_layernorm_ops": len(layernorm_ops),
            "total_attention_time": sum(op["total_time"] for op in attention_ops if op["total_time"] is not None),
            "total_mlp_time": sum(op["total_time"] for op in mlp_ops if op["total_time"] is not None),
            "total_layernorm_time": sum(op["total_time"] for op in layernorm_ops if op["total_time"] is not None),
            "avg_attention_entropy": sum(op["avg_attention_entropy"] for op in attention_ops if op["avg_attention_entropy"] is not None) / len(attention_ops) if attention_ops else None,
            "avg_mlp_activation_kill_ratio": sum(op["activation_kill_ratio"] for op in mlp_ops if op["activation_kill_ratio"] is not None) / len(mlp_ops) if mlp_ops else None
        }

        return {
            "attention_ops": attention_ops,
            "mlp_ops": mlp_ops,
            "layernorm_ops": layernorm_ops,
            "summary": summary
        }


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

        # Set phase in power monitor so samples are tagged correctly
        if self.power_monitor:
            try:
                self.power_monitor.set_phase(phase)
                logger.debug(f"Set power monitor phase to: {phase}")
            except Exception as e:
                logger.warning(f"Failed to set power monitor phase: {e}")

        # Emit section_start event via callback
        if self.section_event_callback:
            try:
                self.section_event_callback(
                    event_type="section_start",
                    phase=phase,
                    section_name=section_name,
                    timestamp=start_time,
                    data={
                        "relative_time_ms": start_relative_ms
                    }
                )
            except Exception as e:
                logger.error(f"Error in section_start callback: {e}")

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

            # Emit section_end event via callback
            if self.section_event_callback:
                try:
                    self.section_event_callback(
                        event_type="section_end",
                        phase=phase,
                        section_name=section_name,
                        timestamp=end_time,
                        data={
                            "relative_time_ms": end_relative_ms,
                            "duration_ms": duration_ms,
                            "energy_mj": energy_mj,
                            "avg_power_mw": avg_power_mw
                        }
                    )
                except Exception as e:
                    logger.error(f"Error in section_end callback: {e}")

    return section_context()


# Attach section method to ProfilingSession
ProfilingSession.section = _session_section
