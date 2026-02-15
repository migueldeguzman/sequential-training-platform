import pytest
import time
from backend.profiling.power_monitor import (
    PowerSample, 
    InferencePipelinePhase, 
    PhaseTaggedPowerMonitor
)

def test_power_sample_initialization():
    """Test basic PowerSample initialization."""
    sample = PowerSample(
        cpu_power_mw=100, 
        gpu_power_mw=50, 
        ane_power_mw=25, 
        dram_power_mw=30,
        phase=InferencePipelinePhase.PREFILL
    )
    
    assert sample.cpu_power_mw == 100
    assert sample.gpu_power_mw == 50
    assert sample.ane_power_mw == 25
    assert sample.dram_power_mw == 30
    assert sample.total_power_mw == 205
    assert sample.phase == InferencePipelinePhase.PREFILL
    assert sample.phase_confidence == 1.0

def test_power_sample_phase_tagging():
    """Test phase tagging with optional metadata."""
    sample = PowerSample(
        cpu_power_mw=200,
        phase=InferencePipelinePhase.DECODE,
        phase_confidence=0.8,
        token_index=42,
        section_name="decode_token_42"
    )
    
    assert sample.phase == InferencePipelinePhase.DECODE
    assert sample.phase_confidence == 0.8
    assert sample.token_index == 42
    assert sample.section_name == "decode_token_42"

def test_phase_tagged_power_monitor():
    """Test PhaseTaggedPowerMonitor phase tracking."""
    monitor = PhaseTaggedPowerMonitor(sample_interval_ms=10)
    
    monitor.set_current_phase(
        InferencePipelinePhase.PREFILL, 
        confidence=0.9,
        section_name="input_processing"
    )
    
    # Simulate a few power samples (mocking the _sampling_loop)
    monitor._samples = [
        PowerSample(cpu_power_mw=100, gpu_power_mw=50, phase=InferencePipelinePhase.PREFILL, phase_confidence=0.9),
        PowerSample(cpu_power_mw=120, gpu_power_mw=60, phase=InferencePipelinePhase.PREFILL, phase_confidence=0.9),
        PowerSample(cpu_power_mw=110, gpu_power_mw=55, phase=InferencePipelinePhase.PREFILL, phase_confidence=0.9)
    ]
    
    # Test get_phase_samples
    phase_samples = monitor.get_phase_samples(
        phase=InferencePipelinePhase.PREFILL, 
        min_confidence=0.5
    )
    
    assert len(phase_samples) == 3
    
    # Test get_phase_energy_summary
    summary = monitor.get_phase_energy_summary(
        phase=InferencePipelinePhase.PREFILL, 
        min_confidence=0.5
    )
    
    assert 'total_energy_mj' in summary
    assert 'avg_power_mw' in summary
    assert 'sample_count' in summary
    assert summary['sample_count'] == 3

def test_monitor_phase_tracking():
    """Test complex phase tracking and energy summary."""
    monitor = PhaseTaggedPowerMonitor(sample_interval_ms=10)
    
    # Simulate different phases
    monitor._samples.append(
        PowerSample(
            cpu_power_mw=50, 
            gpu_power_mw=20, 
            phase=InferencePipelinePhase.PRE_INFERENCE, 
            phase_confidence=0.7
        )
    )
    
    monitor._samples.extend([
        PowerSample(
            cpu_power_mw=200, 
            gpu_power_mw=100, 
            phase=InferencePipelinePhase.PREFILL, 
            phase_confidence=0.9
        ),
        PowerSample(
            cpu_power_mw=210, 
            gpu_power_mw=110, 
            phase=InferencePipelinePhase.PREFILL, 
            phase_confidence=0.9
        )
    ])
    
    monitor._samples.append(
        PowerSample(
            cpu_power_mw=180, 
            gpu_power_mw=90, 
            phase=InferencePipelinePhase.DECODE, 
            phase_confidence=0.8,
            token_index=1
        )
    )
    
    # Verify phase-specific summaries
    pre_inference_summary = monitor.get_phase_energy_summary(InferencePipelinePhase.PRE_INFERENCE)
    prefill_summary = monitor.get_phase_energy_summary(InferencePipelinePhase.PREFILL)
    decode_summary = monitor.get_phase_energy_summary(InferencePipelinePhase.DECODE)
    
    assert pre_inference_summary['sample_count'] == 1
    assert prefill_summary['sample_count'] == 2
    assert decode_summary['sample_count'] == 1

def test_confidence_filtering():
    """Test energy summary with confidence filtering."""
    monitor = PhaseTaggedPowerMonitor(sample_interval_ms=10)
    
    # Samples with varying confidence levels
    monitor._samples = [
        PowerSample(cpu_power_mw=100, phase=InferencePipelinePhase.PREFILL, phase_confidence=0.3),
        PowerSample(cpu_power_mw=200, phase=InferencePipelinePhase.PREFILL, phase_confidence=0.6),
        PowerSample(cpu_power_mw=300, phase=InferencePipelinePhase.PREFILL, phase_confidence=0.9)
    ]
    
    # Test different confidence thresholds
    low_conf_summary = monitor.get_phase_energy_summary(
        InferencePipelinePhase.PREFILL, 
        min_confidence=0.2
    )
    mid_conf_summary = monitor.get_phase_energy_summary(
        InferencePipelinePhase.PREFILL, 
        min_confidence=0.5
    )
    high_conf_summary = monitor.get_phase_energy_summary(
        InferencePipelinePhase.PREFILL, 
        min_confidence=0.8
    )
    
    assert low_conf_summary['sample_count'] == 3
    assert mid_conf_summary['sample_count'] == 2
    assert high_conf_summary['sample_count'] == 1