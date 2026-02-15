from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List
import time
import threading
import plistlib
import warnings

class InferencePipelinePhase(Enum):
    """Enum representing different phases of machine learning inference pipeline."""
    IDLE = auto()        # System idle state
    PRE_INFERENCE = auto()   # Before model forward pass (tokenization, setup)
    PREFILL = auto()     # Initial context processing
    DECODE = auto()      # Token generation phase
    POST_INFERENCE = auto()  # After model inference (cleanup, result processing)

@dataclass
class PowerSample:
    """Enhanced power sample data with phase tagging and detailed component tracking."""
    timestamp: float = field(default_factory=time.time)
    cpu_power_mw: float = 0.0
    gpu_power_mw: float = 0.0
    ane_power_mw: float = 0.0  # Apple Neural Engine
    dram_power_mw: float = 0.0
    total_power_mw: float = 0.0
    
    # New phase-tagging fields
    phase: InferencePipelinePhase = InferencePipelinePhase.IDLE
    phase_confidence: float = 1.0  # How certain are we about the phase? (0.0 - 1.0)
    
    # Additional metadata for richer context
    token_index: Optional[int] = None
    section_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and compute total power if not explicitly set."""
        if self.total_power_mw == 0.0:
            self.total_power_mw = (
                self.cpu_power_mw + 
                self.gpu_power_mw + 
                self.ane_power_mw + 
                self.dram_power_mw
            )
        
        # If phase is not set, default to IDLE
        if self.phase is None:
            self.phase = InferencePipelinePhase.IDLE

class PhaseTaggedPowerMonitor:
    """Enhanced power monitoring system with phase-aware sampling."""
    
    def __init__(self, sample_interval_ms: int = 100):
        """
        Initialize power monitor with optional phase tracking.
        
        Args:
            sample_interval_ms (int): Milliseconds between power samples. Default 100ms.
        """
        self._samples_lock = threading.Lock()
        self._samples: List[PowerSample] = []
        self._current_phase: InferencePipelinePhase = InferencePipelinePhase.IDLE
        self._current_phase_confidence: float = 1.0
        self._sample_interval_ms = sample_interval_ms
        # Other initialization code from previous implementation...

    def set_current_phase(
        self, 
        phase: InferencePipelinePhase, 
        confidence: float = 1.0,
        token_index: Optional[int] = None,
        section_name: Optional[str] = None
    ):
        """
        Update the current inference pipeline phase.
        
        Args:
            phase (InferencePipelinePhase): Current inference phase
            confidence (float): Confidence in phase identification (0.0 - 1.0)
            token_index (Optional[int]): Current token index for decode phase
            section_name (Optional[str]): Name of current processing section
        """
        with self._samples_lock:
            self._current_phase = phase
            self._current_phase_confidence = min(max(confidence, 0.0), 1.0)

    def _sampling_loop(self):
        """Override sampling loop to include phase tagging."""
        while self._is_running:
            try:
                # Existing powermetrics parsing code...
                sample_dict = self._parse_plist_sample(raw_sample)
                
                with self._samples_lock:
                    power_sample = PowerSample(
                        cpu_power_mw=sample_dict.get('cpu_power', 0),
                        gpu_power_mw=sample_dict.get('gpu_power', 0),
                        ane_power_mw=sample_dict.get('ane_power', 0),
                        dram_power_mw=sample_dict.get('dram_power', 0),
                        total_power_mw=sample_dict.get('total_power', 0),
                        phase=self._current_phase,
                        phase_confidence=self._current_phase_confidence
                    )
                    self._samples.append(power_sample)
                
                time.sleep(self._sample_interval_ms / 1000.0)
            
            except Exception as e:
                warnings.warn(f"Power sampling error: {e}")
                break

    def get_phase_samples(
        self, 
        phase: Optional[InferencePipelinePhase] = None,
        min_confidence: float = 0.5
    ) -> List[PowerSample]:
        """
        Retrieve power samples, optionally filtered by phase and confidence.
        
        Args:
            phase (Optional[InferencePipelinePhase]): Filter for specific phase
            min_confidence (float): Minimum phase confidence to include
        
        Returns:
            List[PowerSample]: Filtered list of power samples
        """
        with self._samples_lock:
            if phase is None and min_confidence == 0:
                return self._samples.copy()
            
            return [
                sample for sample in self._samples
                if (phase is None or sample.phase == phase) and
                   (min_confidence == 0 or sample.phase_confidence >= min_confidence)
            ]

    def get_phase_energy_summary(
        self, 
        phase: Optional[InferencePipelinePhase] = None,
        min_confidence: float = 0.5
    ) -> dict:
        """
        Calculate energy consumption summary for a specific phase.
        
        Args:
            phase (Optional[InferencePipelinePhase]): Phase to summarize
            min_confidence (float): Minimum phase confidence to include
        
        Returns:
            dict: Energy consumption summary
        """
        phase_samples = self.get_phase_samples(phase, min_confidence)
        
        if not phase_samples:
            return {
                'total_energy_mj': 0,
                'avg_power_mw': 0,
                'sample_count': 0
            }
        
        total_power = sum(sample.total_power_mw for sample in phase_samples)
        avg_power = total_power / len(phase_samples)
        
        # Assuming 100ms interval between samples, calculate energy
        total_energy = avg_power * len(phase_samples) * 0.1  # mW * samples * 0.1s = mJ
        
        return {
            'total_energy_mj': total_energy,
            'avg_power_mw': avg_power,
            'sample_count': len(phase_samples)
        }