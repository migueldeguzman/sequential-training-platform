import pytest
import time
from backend.profiling.power_monitor import PowerMonitor, PowerSample

def test_power_sample_phase_tagging():
    """Test PowerSample phase tagging mechanism."""
    power_monitor = PowerMonitor(sample_interval_ms=10)
    
    # Test default phase is 'idle'
    power_monitor.start()
    time.sleep(0.05)  # Allow a few samples to be collected
    power_monitor.stop()
    
    samples = power_monitor.get_samples()
    assert all(sample.phase == 'idle' for sample in samples), "All samples should be in 'idle' phase by default"

def test_phase_change_tracking():
    """Verify power samples are correctly tagged when phase changes."""
    power_monitor = PowerMonitor(sample_interval_ms=10)
    
    power_monitor.start()
    
    # Simulate phase changes
    power_monitor.set_current_phase('pre_inference')
    time.sleep(0.05)
    
    power_monitor.set_current_phase('prefill')
    time.sleep(0.05)
    
    power_monitor.set_current_phase('decode')
    time.sleep(0.05)
    
    power_monitor.set_current_phase('post_inference')
    time.sleep(0.05)
    
    power_monitor.stop()
    
    samples = power_monitor.get_samples()
    
    # Count samples per phase
    phase_counts = {
        'idle': sum(1 for s in samples if s.phase == 'idle'),
        'pre_inference': sum(1 for s in samples if s.phase == 'pre_inference'),
        'prefill': sum(1 for s in samples if s.phase == 'prefill'),
        'decode': sum(1 for s in samples if s.phase == 'decode'),
        'post_inference': sum(1 for s in samples if s.phase == 'post_inference')
    }
    
    assert all(count > 0 for phase, count in phase_counts.items()), \
        f"No samples found for some phases: {phase_counts}"

def test_invalid_phase_raises_exception():
    """Ensure setting an invalid phase raises an AssertionError."""
    power_monitor = PowerMonitor()
    
    with pytest.raises(AssertionError, match="Invalid phase"):
        power_monitor.set_current_phase('invalid_phase')

def test_power_sample_dataclass_construction():
    """Verify PowerSample dataclass accepts phase parameter."""
    sample = PowerSample(
        timestamp=time.time(),
        cpu_power_mw=100.0,
        gpu_power_mw=50.0,
        ane_power_mw=20.0,
        dram_power_mw=30.0,
        total_power_mw=200.0,
        phase='prefill'
    )
    
    assert sample.phase == 'prefill', "PowerSample should accept custom phase"