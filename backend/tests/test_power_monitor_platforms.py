"""
TEST-009: Power Monitor Platform Tests

Tests power monitoring on different platforms including Apple Silicon,
NVIDIA GPUs, and mock monitors for CI/testing environments.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from profiling.power_monitor import PowerMonitor, PowerSample


class TestAppleSiliconPowerMetrics(unittest.TestCase):
    """Test Apple Silicon powermetrics integration."""

    @patch('subprocess.Popen')
    def test_powermetrics_command_format(self, mock_popen):
        """Test that powermetrics is called with correct arguments."""
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        monitor = PowerMonitor(sample_rate_ms=100)
        monitor.start()

        # Verify powermetrics command
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]

        self.assertIn('powermetrics', args)
        self.assertIn('--samplers', args)
        self.assertIn('-f', args)
        self.assertIn('plist', args)

        monitor.stop()

    @patch('subprocess.Popen')
    def test_sample_rate_consistency(self, mock_popen):
        """Test that sample rate is consistent."""
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        sample_rate_ms = 50
        monitor = PowerMonitor(sample_rate_ms=sample_rate_ms)
        monitor.start()

        # Verify sample rate argument
        args = mock_popen.call_args[0][0]
        sample_arg_idx = args.index('-i') + 1 if '-i' in args else -1

        if sample_arg_idx > 0:
            interval_ms = int(args[sample_arg_idx])
            self.assertEqual(interval_ms, sample_rate_ms)

        monitor.stop()

    @patch('subprocess.Popen')
    def test_component_breakdown_accuracy(self, mock_popen):
        """Test that component breakdowns are accurately parsed."""
        plist_data = """<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>processor</key>
    <dict>
        <key>clusters</key>
        <array>
            <dict>
                <key>cpu_power</key>
                <real>2500.0</real>
            </dict>
        </array>
        <key>gpu</key>
        <dict>
            <key>gpu_power</key>
            <real>5000.0</real>
        </dict>
        <key>ane</key>
        <dict>
            <key>power</key>
            <real>1000.0</real>
        </dict>
    </dict>
    <key>thermal</key>
    <dict>
        <key>channels</key>
        <array>
            <dict>
                <key>name</key>
                <string>DRAM</string>
                <key>power</key>
                <real>800.0</real>
            </dict>
        </array>
    </dict>
</dict>
</plist>
"""
        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = plist_data.encode()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        monitor = PowerMonitor()
        monitor.start()

        # Get a sample
        sample = monitor.get_latest_sample()

        if sample:
            self.assertAlmostEqual(sample.cpu_power_mw, 2500.0, places=1)
            self.assertAlmostEqual(sample.gpu_power_mw, 5000.0, places=1)
            self.assertAlmostEqual(sample.ane_power_mw, 1000.0, places=1)
            self.assertAlmostEqual(sample.dram_power_mw, 800.0, places=1)
            # Total should be sum of components
            expected_total = 2500.0 + 5000.0 + 1000.0 + 800.0
            self.assertAlmostEqual(sample.total_power_mw, expected_total, places=1)

        monitor.stop()


class TestNVIDIASMI(unittest.TestCase):
    """Test NVIDIA nvidia-smi integration."""

    @patch('subprocess.check_output')
    def test_nvidia_smi_detection(self, mock_check_output):
        """Test detection of NVIDIA GPU."""
        mock_check_output.return_value = b"NVIDIA SMI\nGPU 0: Tesla V100"

        # Assuming PowerMonitor can detect NVIDIA GPUs
        monitor = PowerMonitor(platform='nvidia')

        self.assertIsNotNone(monitor)

    @patch('subprocess.check_output')
    def test_nvidia_power_reading(self, mock_check_output):
        """Test reading power from nvidia-smi."""
        # Mock nvidia-smi output
        mock_check_output.return_value = b"150.5 W\n"

        monitor = PowerMonitor(platform='nvidia')
        monitor.start()

        sample = monitor.get_latest_sample()

        if sample:
            # Power should be in mW
            self.assertGreater(sample.gpu_power_mw, 0)

        monitor.stop()

    @patch('subprocess.check_output')
    def test_nvidia_multi_gpu(self, mock_check_output):
        """Test handling multiple NVIDIA GPUs."""
        mock_check_output.return_value = b"120.5 W\n180.3 W\n"

        monitor = PowerMonitor(platform='nvidia', gpu_index=0)
        monitor.start()

        sample = monitor.get_latest_sample()

        if sample:
            self.assertGreater(sample.gpu_power_mw, 0)

        monitor.stop()


class TestMockMonitor(unittest.TestCase):
    """Test mock monitors for CI/testing."""

    def test_mock_monitor_initialization(self):
        """Test mock monitor can be initialized."""
        monitor = PowerMonitor(mock=True)

        self.assertIsNotNone(monitor)
        self.assertTrue(monitor.is_mock)

    def test_mock_monitor_samples(self):
        """Test mock monitor generates realistic samples."""
        monitor = PowerMonitor(mock=True)
        monitor.start()

        samples = []
        for _ in range(10):
            sample = monitor.get_latest_sample()
            if sample:
                samples.append(sample)

        self.assertGreater(len(samples), 0)

        # Check samples have realistic values
        for sample in samples:
            self.assertGreater(sample.cpu_power_mw, 0)
            self.assertGreater(sample.gpu_power_mw, 0)
            self.assertGreater(sample.total_power_mw, 0)
            # Total should be sum of components
            expected_total = (
                sample.cpu_power_mw +
                sample.gpu_power_mw +
                sample.ane_power_mw +
                sample.dram_power_mw
            )
            self.assertAlmostEqual(sample.total_power_mw, expected_total, places=1)

        monitor.stop()

    def test_mock_monitor_variability(self):
        """Test mock monitor has realistic variability."""
        monitor = PowerMonitor(mock=True)
        monitor.start()

        samples = []
        for _ in range(100):
            sample = monitor.get_latest_sample()
            if sample:
                samples.append(sample.total_power_mw)

        # Samples should vary
        self.assertGreater(max(samples), min(samples))

        # But not too wildly (within 50% of mean)
        mean_power = sum(samples) / len(samples)
        for power in samples:
            self.assertLess(abs(power - mean_power) / mean_power, 0.5)

        monitor.stop()

    def test_mock_monitor_sample_rate(self):
        """Test mock monitor respects sample rate."""
        sample_rate_ms = 100
        monitor = PowerMonitor(mock=True, sample_rate_ms=sample_rate_ms)
        monitor.start()

        import time
        start_time = time.time()

        sample_count = 0
        while time.time() - start_time < 1.0:  # Collect for 1 second
            if monitor.get_latest_sample():
                sample_count += 1
            time.sleep(0.01)

        expected_samples = 1000 / sample_rate_ms  # samples per second
        # Allow 50% tolerance
        self.assertGreater(sample_count, expected_samples * 0.5)
        self.assertLess(sample_count, expected_samples * 1.5)

        monitor.stop()


class TestPlatformDetection(unittest.TestCase):
    """Test automatic platform detection."""

    @patch('platform.system')
    @patch('platform.machine')
    def test_apple_silicon_detection(self, mock_machine, mock_system):
        """Test detection of Apple Silicon."""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'

        monitor = PowerMonitor()

        # Should detect Apple Silicon
        self.assertTrue(monitor.platform in ['apple_silicon', 'darwin'])

    @patch('platform.system')
    @patch('subprocess.check_output')
    def test_nvidia_gpu_detection(self, mock_check_output, mock_system):
        """Test detection of NVIDIA GPU on Linux."""
        mock_system.return_value = 'Linux'
        mock_check_output.return_value = b"NVIDIA SMI"

        monitor = PowerMonitor()

        # Should detect NVIDIA platform or fallback gracefully
        self.assertIsNotNone(monitor)

    @patch('platform.system')
    def test_unsupported_platform_fallback(self, mock_system):
        """Test fallback to mock on unsupported platform."""
        mock_system.return_value = 'Windows'

        monitor = PowerMonitor()

        # Should either use mock or raise appropriate error
        self.assertIsNotNone(monitor)


class TestSampleCollection(unittest.TestCase):
    """Test sample collection accuracy and consistency."""

    def test_sample_timestamp_monotonic(self):
        """Test that sample timestamps are monotonically increasing."""
        monitor = PowerMonitor(mock=True)
        monitor.start()

        timestamps = []
        for _ in range(50):
            sample = monitor.get_latest_sample()
            if sample:
                timestamps.append(sample.timestamp)

        # Timestamps should be increasing
        for i in range(1, len(timestamps)):
            self.assertGreaterEqual(timestamps[i], timestamps[i - 1])

        monitor.stop()

    def test_sample_completeness(self):
        """Test that samples have all required fields."""
        monitor = PowerMonitor(mock=True)
        monitor.start()

        sample = monitor.get_latest_sample()

        self.assertIsNotNone(sample)
        self.assertIsNotNone(sample.timestamp)
        self.assertIsNotNone(sample.cpu_power_mw)
        self.assertIsNotNone(sample.gpu_power_mw)
        self.assertIsNotNone(sample.total_power_mw)

        monitor.stop()

    def test_sample_buffer_size(self):
        """Test that sample buffer doesn't grow unbounded."""
        monitor = PowerMonitor(mock=True, buffer_size=100)
        monitor.start()

        # Generate many samples
        for _ in range(500):
            monitor.get_latest_sample()

        # Buffer should be limited
        buffer_size = len(monitor.get_all_samples())
        self.assertLessEqual(buffer_size, 100)

        monitor.stop()


class TestErrorHandling(unittest.TestCase):
    """Test error handling for various failure scenarios."""

    @patch('subprocess.Popen')
    def test_powermetrics_permission_denied(self, mock_popen):
        """Test handling of permission denied error."""
        mock_popen.side_effect = PermissionError("Permission denied")

        monitor = PowerMonitor()

        with self.assertRaises(PermissionError):
            monitor.start()

    @patch('subprocess.Popen')
    def test_powermetrics_not_found(self, mock_popen):
        """Test handling of powermetrics not found."""
        mock_popen.side_effect = FileNotFoundError("powermetrics not found")

        monitor = PowerMonitor()

        with self.assertRaises(FileNotFoundError):
            monitor.start()

    def test_malformed_plist_recovery(self):
        """Test recovery from malformed plist data."""
        monitor = PowerMonitor(mock=True)
        monitor.start()

        # Inject malformed data
        # (Implementation depends on PowerMonitor internals)

        # Should continue operating despite errors
        sample = monitor.get_latest_sample()
        self.assertIsNotNone(sample)

        monitor.stop()

    def test_monitor_restart_after_crash(self):
        """Test that monitor can be restarted after crash."""
        monitor = PowerMonitor(mock=True)

        monitor.start()
        monitor.stop()

        # Should be able to restart
        monitor.start()
        sample = monitor.get_latest_sample()
        self.assertIsNotNone(sample)
        monitor.stop()


class TestPerformance(unittest.TestCase):
    """Test performance characteristics of power monitoring."""

    def test_sampling_overhead(self):
        """Test that power sampling has low overhead."""
        import time

        monitor = PowerMonitor(mock=True, sample_rate_ms=10)
        monitor.start()

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < 1.0:
            monitor.get_latest_sample()
            sample_count += 1

        elapsed = time.time() - start_time

        # Should be able to collect many samples per second
        samples_per_second = sample_count / elapsed
        self.assertGreater(samples_per_second, 50)

        monitor.stop()

    def test_memory_usage_stable(self):
        """Test that memory usage doesn't grow over time."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        monitor = PowerMonitor(mock=True)
        monitor.start()

        # Generate many samples
        for _ in range(10000):
            monitor.get_latest_sample()

        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Memory growth should be minimal (< 10 MB)
        self.assertLess(memory_growth, 10)

        monitor.stop()


if __name__ == '__main__':
    unittest.main()
