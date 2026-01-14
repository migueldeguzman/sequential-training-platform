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
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
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
    phase: str = 'idle'  # Inference phase: idle, pre_inference, prefill, decode, post_inference

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
        self._samples_lock = threading.Lock()  # Thread-safe access to samples
        self._start_time: Optional[float] = None
        self._running = False
        self._sampling_thread: Optional[threading.Thread] = None
        self._plist_buffer = ""
        self._plist_buffer_start_time: Optional[float] = None  # Track buffer age for timeout
        self._current_phase: str = 'idle'  # Current inference phase

        # Peak power tracking
        self._peak_total_power_mw: float = 0.0
        self._peak_cpu_power_mw: float = 0.0
        self._peak_gpu_power_mw: float = 0.0
        self._peak_ane_power_mw: float = 0.0
        self._peak_dram_power_mw: float = 0.0
        self._peak_timestamp_ms: float = 0.0

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

            # Power values are directly in the processor dict (in milliwatts)
            cpu_power_mw = processor.get('cpu_power', 0.0)
            gpu_power_mw = processor.get('gpu_power', 0.0)
            ane_power_mw = processor.get('ane_power', 0.0)

            # Combined power is also available directly
            combined_power = processor.get('combined_power', 0.0)

            # If direct values not available, fall back to cluster-based calculation
            if cpu_power_mw == 0.0:
                clusters = processor.get('clusters', [])
                for cluster in clusters:
                    cpu_power_mw += cluster.get('cpu_power', 0.0)

            # GPU power fallback - check nested gpu dict
            if gpu_power_mw == 0.0:
                gpu = processor.get('gpu', {})
                gpu_power_mw = gpu.get('gpu_power', 0.0)

            # DRAM power - from thermal samplers (not always available)
            thermal = plist_data.get('thermal', {})
            if 'channels' in thermal:
                for channel in thermal.get('channels', []):
                    if 'DRAM' in channel.get('name', ''):
                        dram_power_mw += channel.get('power', 0.0)

            # Use combined_power if available and our calculation seems off
            total_power_mw = cpu_power_mw + gpu_power_mw + ane_power_mw + dram_power_mw
            if combined_power > 0 and total_power_mw == 0:
                total_power_mw = combined_power

            # Track peak power values
            if total_power_mw > self._peak_total_power_mw:
                self._peak_total_power_mw = total_power_mw
                self._peak_cpu_power_mw = cpu_power_mw
                self._peak_gpu_power_mw = gpu_power_mw
                self._peak_ane_power_mw = ane_power_mw
                self._peak_dram_power_mw = dram_power_mw
                self._peak_timestamp_ms = relative_time_ms

            return PowerSample(
                timestamp=timestamp,
                relative_time_ms=relative_time_ms,
                cpu_power_mw=cpu_power_mw,
                gpu_power_mw=gpu_power_mw,
                ane_power_mw=ane_power_mw,
                dram_power_mw=dram_power_mw,
                total_power_mw=total_power_mw,
                phase=self._current_phase
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
            in_plist = False
            plist_buffer_timeout = 10.0  # Maximum time to buffer incomplete plist (seconds)

            # Read powermetrics output line by line
            for line in self._process.stdout:
                if not self._running:
                    break

                # Check for buffer timeout - if we've been buffering too long, reset
                if in_plist and self._plist_buffer_start_time:
                    buffer_age = time.time() - self._plist_buffer_start_time
                    if buffer_age > plist_buffer_timeout:
                        print(f"Warning: Plist buffer timeout ({buffer_age:.1f}s), discarding incomplete sample")
                        self._plist_buffer = ""
                        self._plist_buffer_start_time = None
                        in_plist = False

                # Detect start of plist more robustly
                # Look for XML declaration or plist tag, handling split tags
                line_stripped = line.strip()
                if not in_plist:
                    # Start buffering if we see XML declaration or plist start tag
                    if '<?xml' in line or '<plist' in line:
                        self._plist_buffer = line
                        self._plist_buffer_start_time = time.time()
                        in_plist = True
                else:
                    # Continue buffering
                    self._plist_buffer += line

                # Check if we have a complete plist
                # More robust check: ensure buffer has both start and end tags
                if in_plist and '</plist>' in line:
                    # Verify buffer has proper XML structure before parsing
                    buffer_lower = self._plist_buffer.lower()
                    has_xml_or_plist_start = '<?xml' in buffer_lower or '<plist' in buffer_lower

                    if has_xml_or_plist_start:
                        try:
                            # Parse the complete plist
                            plist_data = plistlib.loads(self._plist_buffer.encode('utf-8'))

                            # Extract power sample
                            sample = self._parse_plist_sample(plist_data)
                            if sample:
                                with self._samples_lock:
                                    self._samples.append(sample)
                                    # Call power sample callback if set
                                    if hasattr(self, '_power_sample_callback') and self._power_sample_callback:
                                        try:
                                            self._power_sample_callback(sample)
                                        except Exception as cb_e:
                                            print(f"Warning: Power sample callback failed: {cb_e}")

                        except plistlib.InvalidFileException as e:
                            # Invalid plist format - log and skip this sample
                            print(f"Warning: Failed to parse plist (invalid format): {e}")
                        except Exception as e:
                            # Other parsing errors - log and skip
                            print(f"Warning: Failed to parse plist: {e}")
                        finally:
                            # Reset buffer for next sample
                            self._plist_buffer = ""
                            self._plist_buffer_start_time = None
                            in_plist = False
                    else:
                        # Malformed buffer without proper start tag, reset
                        print(f"Warning: Malformed plist buffer (missing start tag), discarding")
                        self._plist_buffer = ""
                        self._plist_buffer_start_time = None
                        in_plist = False

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
            RuntimeError: If PowerMonitor is already running or subprocess fails to start
            PermissionError: If powermetrics cannot be run without password
            FileNotFoundError: If powermetrics is not installed
        """
        if self._running:
            raise RuntimeError("PowerMonitor is already running")

        if not self.is_available():
            raise PermissionError(
                "powermetrics requires passwordless sudo access. "
                "Run setup_powermetrics.sh or see README_POWERMETRICS.md for setup instructions. "
                "To add passwordless sudo access, add this line to /etc/sudoers.d/powermetrics:\n"
                f"{os.environ.get('USER', 'USERNAME')} ALL=(ALL) NOPASSWD: /usr/bin/powermetrics"
            )

        try:
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
        except FileNotFoundError:
            raise FileNotFoundError(
                "powermetrics command not found. This tool is only available on macOS. "
                "If you're on macOS, ensure powermetrics is installed at /usr/bin/powermetrics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start powermetrics subprocess: {str(e)}")

        # Verify process started successfully
        time.sleep(0.1)  # Give process a moment to start
        if self._process.poll() is not None:
            stderr_output = self._process.stderr.read() if self._process.stderr else "No error output"
            raise RuntimeError(
                f"powermetrics process terminated immediately after start. "
                f"Return code: {self._process.returncode}. "
                f"Error output: {stderr_output}"
            )

        self._start_time = time.time()
        self._running = True
        self._samples = []
        self._plist_buffer = ""
        self._plist_buffer_start_time = None

        # Reset peak power tracking
        self._peak_total_power_mw = 0.0
        self._peak_cpu_power_mw = 0.0
        self._peak_gpu_power_mw = 0.0
        self._peak_ane_power_mw = 0.0
        self._peak_dram_power_mw = 0.0
        self._peak_timestamp_ms = 0.0

        # Start background sampling thread
        self._sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampling_thread.start()

        print(f"PowerMonitor started successfully with {self.sample_interval_ms}ms sampling interval")

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
        with self._samples_lock:
            return self._samples.copy()

    def get_current(self) -> Optional[PowerSample]:
        """
        Get the most recent power sample.

        Returns:
            Latest PowerSample or None if no samples collected yet
        """
        with self._samples_lock:
            return self._samples[-1] if self._samples else None

    def set_phase(self, phase: str) -> None:
        """
        Set the current inference phase for tagging power samples.

        Args:
            phase: Phase name (idle, pre_inference, prefill, decode, post_inference)
        """
        valid_phases = {'idle', 'pre_inference', 'prefill', 'decode', 'post_inference'}
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase '{phase}'. Must be one of: {valid_phases}")
        self._current_phase = phase

    def get_phase(self) -> str:
        """
        Get the current inference phase.

        Returns:
            Current phase name
        """
        return self._current_phase

    def get_peak_power(self) -> Dict[str, float]:
        """
        Get peak power values recorded during sampling.

        Returns:
            Dictionary with peak power values for each component:
                - peak_power_mw: Peak total power
                - peak_power_cpu_mw: Peak CPU power
                - peak_power_gpu_mw: Peak GPU power
                - peak_power_ane_mw: Peak ANE power
                - peak_power_dram_mw: Peak DRAM power
                - peak_power_timestamp_ms: Time when peak occurred (relative to start)
        """
        with self._samples_lock:
            return {
                'peak_power_mw': self._peak_total_power_mw,
                'peak_power_cpu_mw': self._peak_cpu_power_mw,
                'peak_power_gpu_mw': self._peak_gpu_power_mw,
                'peak_power_ane_mw': self._peak_ane_power_mw,
                'peak_power_dram_mw': self._peak_dram_power_mw,
                'peak_power_timestamp_ms': self._peak_timestamp_ms
            }

    def measure_idle_baseline(self, duration_seconds: float = 2.0) -> Dict[str, float]:
        """
        Measure idle power baseline before inference starts.

        Samples power for the specified duration while the system is idle
        to establish a baseline for calculating active power delta.

        Args:
            duration_seconds: How long to sample idle power (default: 2.0 seconds)

        Returns:
            Dictionary with baseline power measurements:
                - baseline_power_mw: Average total idle power
                - baseline_cpu_power_mw: Average idle CPU power
                - baseline_gpu_power_mw: Average idle GPU power
                - baseline_ane_power_mw: Average idle ANE power
                - baseline_dram_power_mw: Average idle DRAM power
                - baseline_sample_count: Number of samples used for baseline

        Raises:
            RuntimeError: If PowerMonitor is not running or no samples collected

        Example:
            monitor = PowerMonitor()
            monitor.start()
            baseline = monitor.measure_idle_baseline(duration_seconds=2.0)
            # Now proceed with inference...
        """
        if not self._running:
            raise RuntimeError("PowerMonitor must be running to measure idle baseline")

        # Record starting sample count
        initial_sample_count = len(self._samples)

        # Wait for the specified duration while collecting idle samples
        print(f"Measuring idle baseline for {duration_seconds} seconds...")
        time.sleep(duration_seconds)

        # Get all samples collected during idle period
        with self._samples_lock:
            idle_samples = self._samples[initial_sample_count:]

        if not idle_samples:
            raise RuntimeError(
                f"No idle samples collected during {duration_seconds} second baseline measurement. "
                f"Check that powermetrics is sampling correctly."
            )

        # Calculate average power across idle samples
        total_cpu = sum(s.cpu_power_mw for s in idle_samples)
        total_gpu = sum(s.gpu_power_mw for s in idle_samples)
        total_ane = sum(s.ane_power_mw for s in idle_samples)
        total_dram = sum(s.dram_power_mw for s in idle_samples)
        total_power = sum(s.total_power_mw for s in idle_samples)
        sample_count = len(idle_samples)

        baseline = {
            'baseline_power_mw': total_power / sample_count,
            'baseline_cpu_power_mw': total_cpu / sample_count,
            'baseline_gpu_power_mw': total_gpu / sample_count,
            'baseline_ane_power_mw': total_ane / sample_count,
            'baseline_dram_power_mw': total_dram / sample_count,
            'baseline_sample_count': sample_count
        }

        print(f"Idle baseline established: {baseline['baseline_power_mw']:.2f} mW total power "
              f"({sample_count} samples)")

        return baseline

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._running:
            self.stop()
        return False
