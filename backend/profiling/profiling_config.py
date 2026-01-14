"""
Profiling Configuration

Centralized configuration for Energy Profiler performance settings.
Allows users to tune profiling overhead vs. detail level.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProfilingConfig:
    """
    Configuration for Energy Profiler.

    Performance Modes:
    ------------------

    MINIMAL (lowest overhead, ~1-2% slowdown):
    - capture_activations=False
    - deep_profiling=False
    - Captures: Power, layer timing only

    STANDARD (balanced, ~3-5% slowdown):
    - capture_activations=True
    - deep_profiling=False
    - Captures: Power, layer timing, activation stats

    DEEP (detailed, ~8-15% slowdown):
    - capture_activations=True
    - deep_profiling=True
    - Captures: Everything + operation-level metrics

    Attributes:
        capture_activations: Whether to capture activation statistics (mean, std, max, sparsity)
        deep_profiling: Whether to enable deep operation-level profiling (attention, MLP ops)
        power_sample_interval_ms: Power sampling interval in milliseconds (default: 100ms)
        batch_db_commits: Whether to batch database commits (recommended: True)
        preallocate_metrics: Number of metric entries to preallocate (improves list append performance)
        max_runs_to_keep: Maximum number of profiling runs to retain (0 = unlimited)
        max_run_age_days: Maximum age of runs to keep in days (0 = unlimited)
    """

    capture_activations: bool = True
    deep_profiling: bool = False
    power_sample_interval_ms: int = 100
    batch_db_commits: bool = True
    preallocate_metrics: int = 10000
    sparsity_threshold: float = 1e-4
    max_runs_to_keep: int = 0
    max_run_age_days: int = 0

    @classmethod
    def minimal(cls) -> "ProfilingConfig":
        """
        Minimal profiling mode: lowest overhead, basic metrics only.

        Overhead: ~1-2%
        Captures: Power samples, section timing

        Returns:
            ProfilingConfig with minimal settings
        """
        return cls(
            capture_activations=False,
            deep_profiling=False,
            power_sample_interval_ms=100,
            batch_db_commits=True,
            preallocate_metrics=5000
        )

    @classmethod
    def standard(cls) -> "ProfilingConfig":
        """
        Standard profiling mode: balanced overhead and detail.

        Overhead: ~3-5%
        Captures: Power samples, section timing, layer metrics, activation stats

        Returns:
            ProfilingConfig with standard settings
        """
        return cls(
            capture_activations=True,
            deep_profiling=False,
            power_sample_interval_ms=100,
            batch_db_commits=True,
            preallocate_metrics=10000
        )

    @classmethod
    def deep(cls) -> "ProfilingConfig":
        """
        Deep profiling mode: maximum detail, higher overhead.

        Overhead: ~8-15%
        Captures: Everything + deep operation metrics (attention ops, MLP ops, etc.)

        Returns:
            ProfilingConfig with deep profiling settings
        """
        return cls(
            capture_activations=True,
            deep_profiling=True,
            power_sample_interval_ms=100,
            batch_db_commits=True,
            preallocate_metrics=20000
        )

    @classmethod
    def custom(
        cls,
        capture_activations: bool = True,
        deep_profiling: bool = False,
        power_sample_interval_ms: int = 100,
        batch_db_commits: bool = True,
        preallocate_metrics: int = 10000,
        sparsity_threshold: float = 1e-4
    ) -> "ProfilingConfig":
        """
        Create custom profiling configuration.

        Args:
            capture_activations: Capture activation statistics
            deep_profiling: Enable deep operation-level profiling
            power_sample_interval_ms: Power sampling interval (ms)
            batch_db_commits: Batch database commits for better performance
            preallocate_metrics: Number of metrics to preallocate
            sparsity_threshold: Threshold for activation sparsity calculation

        Returns:
            ProfilingConfig with custom settings
        """
        return cls(
            capture_activations=capture_activations,
            deep_profiling=deep_profiling,
            power_sample_interval_ms=power_sample_interval_ms,
            batch_db_commits=batch_db_commits,
            preallocate_metrics=preallocate_metrics,
            sparsity_threshold=sparsity_threshold
        )

    def get_profiling_depth(self) -> str:
        """
        Get profiling depth string for database storage.

        Returns:
            'module' or 'deep' based on deep_profiling setting
        """
        return 'deep' if self.deep_profiling else 'module'

    def estimate_overhead_percentage(self) -> tuple[float, float]:
        """
        Estimate profiling overhead as percentage range.

        Returns:
            Tuple of (min_overhead_pct, max_overhead_pct)
        """
        if not self.capture_activations and not self.deep_profiling:
            return (1.0, 2.0)  # Minimal mode
        elif self.capture_activations and not self.deep_profiling:
            return (3.0, 5.0)  # Standard mode
        else:
            return (8.0, 15.0)  # Deep mode

    def __str__(self) -> str:
        """String representation of configuration."""
        mode = "DEEP" if self.deep_profiling else ("MINIMAL" if not self.capture_activations else "STANDARD")
        overhead_min, overhead_max = self.estimate_overhead_percentage()
        return (
            f"ProfilingConfig(mode={mode}, overhead={overhead_min}-{overhead_max}%, "
            f"activations={self.capture_activations}, deep={self.deep_profiling})"
        )
