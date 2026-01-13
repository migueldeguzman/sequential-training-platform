"""
PowerMonitor - Manages powermetrics subprocess for real-time power sampling

This module provides a class that wraps the macOS `powermetrics` tool to sample
CPU, GPU, ANE, and DRAM power consumption during ML inference workloads.

Requires: sudoers entry for passwordless powermetrics access
"""

import subprocess
import time
import plistlib
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import io


@dataclass
class PowerSample:
    """Single power measurement sample from powermetrics"""
    timestamp: float  # Unix timestamp
    relative_time_ms: float  # Time since profiling started (milliseconds)
    cpu_power_mw: float
    gpu_power_mw: float
    ane_power_mw: float
    dram_power_mw: float
    total_power_mw: float

    @property
    def total_power_w(self) -> float:
        """Total power in watts"""
        return self.total_power_mw / 1000.0


class PowerMonitor:
    """
    Manages powermetrics subprocess for continuous power sampling.

    Usage:
        monitor = PowerMonitor(sample_interval_ms=100)
        monitor.start()
        # ... run inference workload ...
        monitor.stop()
        samples = monitor.get_samples()

    Or use as context manager:
        with PowerMonitor(sample_interval_ms=100) as monitor:
            # ... run inference workload ...
            pass
        samples = monitor.get_samples()
    """

    def __init__(self, sample_interval_ms: int = 100):
        """
        Initialize PowerMonitor.

        Args:
            sample_interval_ms: Sampling interval in milliseconds (default: 100ms)
        """
        self.sample_interval_ms = sample_interval_ms
        self._process: Optional[subprocess.Popen] = None
        self._samples: List[PowerSample] = []
        self._start_time: Optional[float] = None
        self._running = False
        self._sampling_thread: Optional[threading.Thread] = None
        self._plist_buffer = ""

    def _parse_plist_sample(self, plist_data: Dict[str, Any]) -> Optional[PowerSample]:
        """
        Parse a single power sample from powermetrics plist output.

        Args:
            plist_data: Parsed plist dictionary from powermetrics

        Returns:
            PowerSample object or None if parsing fails
        """
        try:
            timestamp = time.time()
            relative_time_ms = (timestamp - self._start_time) * 1000.0 if self._start_time else 0.0

            # Initialize power values to 0
            cpu_power_mw = 0.0
            gpu_power_mw = 0.0
            ane_power_mw = 0.0
            dram_power_mw = 0.0

            # Extract processor data (contains CPU, GPU, ANE power)
            processor = plist_data.get('processor', {})

            # CPU power - sum all clusters
            clusters = processor.get('clusters', [])
            for cluster in clusters:
                cpu_power_mw += cluster.get('cpu_power', 0.0)

            # GPU power
            gpu = processor.get('gpu', {})
            gpu_power_mw = gpu.get('gpu_power', 0.0)

            # ANE (Apple Neural Engine) power
            ane = processor.get('ane', {})
            ane_power_mw = ane.get('power', 0.0)

            # DRAM power - from thermal samplers
            thermal = plist_data.get('thermal', {})
            if 'channels' in thermal:
                # Sum DRAM power from all channels
                for channel in thermal.get('channels', []):
                    if 'DRAM' in channel.get('name', ''):
                        dram_power_mw += channel.get('power', 0.0)

            # Calculate total power
            total_power_mw = cpu_power_mw + gpu_power_mw + ane_power_mw + dram_power_mw

            return PowerSample(
                timestamp=timestamp,
                relative_time_ms=relative_time_ms,
                cpu_power_mw=cpu_power_mw,
                gpu_power_mw=gpu_power_mw,
                ane_power_mw=ane_power_mw,
                dram_power_mw=dram_power_mw,
                total_power_mw=total_power_mw
            )
        except (KeyError, TypeError, ValueError) as e:
            # Gracefully handle parsing errors
            print(f"Warning: Failed to parse power sample: {e}")
            return None

    def _sampling_loop(self) -> None:
        """
        Background thread to continuously read and parse powermetrics output.
        """
        if not self._process or not self._process.stdout:
            return

        try:
            # Read powermetrics output line by line
            for line in self._process.stdout:
                if not self._running:
                    break

                # Buffer lines until we have a complete plist
                self._plist_buffer += line

                # Check if we have a complete plist (ends with </plist>)
                if '</plist>' in line:
                    try:
                        # Parse the complete plist
                        plist_data = plistlib.loads(self._plist_buffer.encode('utf-8'))

                        # Extract power sample
                        sample = self._parse_plist_sample(plist_data)
                        if sample:
                            self._samples.append(sample)

                        # Reset buffer for next sample
                        self._plist_buffer = ""
                    except Exception as e:
                        print(f"Warning: Failed to parse plist: {e}")
                        self._plist_buffer = ""
        except Exception as e:
            print(f"Error in sampling loop: {e}")

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if powermetrics is available with proper permissions.

        Returns:
            True if powermetrics can be run without password, False otherwise
        """
        try:
            result = subprocess.run(
                ['sudo', '-n', 'powermetrics', '--help'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def start(self) -> None:
        """
        Start the powermetrics subprocess.

        Spawns powermetrics in background and begins collecting samples.

        Raises:
            RuntimeError: If PowerMonitor is already running
            PermissionError: If powermetrics cannot be run without password
        """
        if self._running:
            raise RuntimeError("PowerMonitor is already running")

        if not self.is_available():
            raise PermissionError(
                "powermetrics requires passwordless sudo access. "
                "Run setup_powermetrics.sh or see README_POWERMETRICS.md"
            )

        # Start powermetrics subprocess
        # -i: sample interval in milliseconds
        # -f plist: output format (XML plist for easy parsing)
        # --samplers: which samplers to enable (cpu_power, gpu_power, etc.)
        self._process = subprocess.Popen(
            [
                'sudo', 'powermetrics',
                '-i', str(self.sample_interval_ms),
                '-f', 'plist',
                '--samplers', 'cpu_power,gpu_power,thermal'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        self._start_time = time.time()
        self._running = True
        self._samples = []
        self._plist_buffer = ""

        # Start background sampling thread
        self._sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampling_thread.start()

    def stop(self) -> None:
        """
        Stop the powermetrics subprocess and finalize sample collection.

        Terminates the subprocess and ensures all samples are collected.

        Raises:
            RuntimeError: If PowerMonitor is not running
        """
        if not self._running:
            raise RuntimeError("PowerMonitor is not running")

        # Signal thread to stop
        self._running = False

        # Terminate powermetrics subprocess
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

        # Wait for sampling thread to finish
        if self._sampling_thread and self._sampling_thread.is_alive():
            self._sampling_thread.join(timeout=2)

        self._process = None
        self._sampling_thread = None

    def is_running(self) -> bool:
        """Check if the PowerMonitor is currently running"""
        return self._running

    def get_samples(self) -> List[PowerSample]:
        """
        Retrieve all collected power samples.

        Returns:
            List of PowerSample objects collected since start()
        """
        return self._samples.copy()

    def get_current(self) -> Optional[PowerSample]:
        """
        Get the most recent power sample.

        Returns:
            Latest PowerSample or None if no samples collected yet
        """
        return self._samples[-1] if self._samples else None

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._running:
            self.stop()
        return False
