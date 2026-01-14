"""
Model Architecture Detector

Auto-detects transformer model architectures and returns standardized component paths
for hook registration in LayerProfiler.

Supported architectures:
- Llama (meta-llama/Llama-*, etc.)
- Mistral (mistralai/Mistral-*, etc.)
- Phi (microsoft/phi-*, etc.)
- Qwen (Qwen/Qwen-*, etc.)
- Gemma (google/gemma-*, etc.)
- StableLM (stabilityai/stablelm-*, etc.)

Usage:
    detector = ModelArchitectureDetector(model)
    components = detector.detect()
    # Returns dict with paths to attention, mlp, and layer norm components
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ComponentPaths:
    """Standardized paths to model components for profiling."""
    architecture: str
    num_layers: int

    # Layer access pattern
    layers_path: str  # e.g., "model.layers"

    # Attention components (relative to layer)
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str

    # MLP components (relative to layer)
    gate_proj: Optional[str]  # Some architectures don't have gate
    up_proj: str
    down_proj: str

    # Layer norms (relative to layer)
    input_layernorm: str
    post_attention_layernorm: str

    # Optional: RMSNorm vs LayerNorm distinction
    norm_type: str  # "rmsnorm" or "layernorm"

    def __str__(self):
        return f"ComponentPaths(architecture={self.architecture}, num_layers={self.num_layers})"


class ModelArchitectureDetector:
    """Detects model architecture and returns component paths for profiling."""

    def __init__(self, model: Any):
        """
        Initialize detector with a PyTorch model.

        Args:
            model: PyTorch model to detect (HuggingFace transformers model)
        """
        self.model = model
        self.model_config = getattr(model, 'config', None)

    def detect(self) -> ComponentPaths:
        """
        Detect model architecture and return component paths.

        Returns:
            ComponentPaths object with standardized paths

        Raises:
            ValueError: If model architecture cannot be detected
        """
        # Try to get model type from config
        model_type = None
        if self.model_config:
            model_type = getattr(self.model_config, 'model_type', None)

        # Detect architecture based on model type or structure
        if model_type == 'stablelm' or self._is_stablelm_structure():
            return self._detect_stablelm()
        elif model_type == 'llama' or self._is_llama_structure():
            return self._detect_llama()
        elif model_type == 'mistral' or self._is_mistral_structure():
            return self._detect_mistral()
        elif model_type == 'phi' or self._is_phi_structure():
            return self._detect_phi()
        elif model_type == 'qwen2' or self._is_qwen_structure():
            return self._detect_qwen()
        elif model_type == 'gemma' or model_type == 'gemma2' or self._is_gemma_structure():
            return self._detect_gemma()
        else:
            logger.warning(
                f"Unknown architecture (model_type={model_type}). "
                "Using fallback detection."
            )
            return self._fallback_detection()

    def _is_stablelm_structure(self) -> bool:
        """Check if model has StableLM-like structure."""
        try:
            # Check for characteristic StableLM structure
            # StableLM models are similar to Llama but have different attention handling
            model_class_name = self.model.__class__.__name__
            return (
                'StableLM' in model_class_name or
                'Stablelm' in model_class_name or
                (hasattr(self.model, 'model') and
                 hasattr(self.model.model, 'layers') and
                 len(self.model.model.layers) > 0 and
                 hasattr(self.model.model.layers[0], 'self_attn') and
                 # StableLM specific: check config for stablelm markers
                 self.model_config and
                 (getattr(self.model_config, 'model_type', None) == 'stablelm' or
                  'stablelm' in str(type(self.model)).lower()))
            )
        except (AttributeError, IndexError):
            return False

    def _is_llama_structure(self) -> bool:
        """Check if model has Llama-like structure."""
        try:
            # Check for characteristic Llama structure
            return (
                hasattr(self.model, 'model') and
                hasattr(self.model.model, 'layers') and
                len(self.model.model.layers) > 0 and
                hasattr(self.model.model.layers[0], 'self_attn') and
                hasattr(self.model.model.layers[0].self_attn, 'q_proj') and
                hasattr(self.model.model.layers[0], 'mlp') and
                hasattr(self.model.model.layers[0].mlp, 'gate_proj')
            )
        except (AttributeError, IndexError):
            return False

    def _is_mistral_structure(self) -> bool:
        """Check if model has Mistral-like structure (similar to Llama)."""
        # Mistral uses similar structure to Llama
        return self._is_llama_structure()

    def _is_phi_structure(self) -> bool:
        """Check if model has Phi-like structure."""
        try:
            # Phi models have slightly different naming
            return (
                hasattr(self.model, 'model') and
                hasattr(self.model.model, 'layers') and
                len(self.model.model.layers) > 0 and
                hasattr(self.model.model.layers[0], 'self_attn') and
                hasattr(self.model.model.layers[0], 'mlp')
            )
        except (AttributeError, IndexError):
            return False

    def _is_qwen_structure(self) -> bool:
        """Check if model has Qwen-like structure."""
        try:
            # Qwen2 uses similar structure to Llama
            return (
                hasattr(self.model, 'model') and
                hasattr(self.model.model, 'layers') and
                len(self.model.model.layers) > 0 and
                hasattr(self.model.model.layers[0], 'self_attn') and
                hasattr(self.model.model.layers[0], 'mlp')
            )
        except (AttributeError, IndexError):
            return False

    def _is_gemma_structure(self) -> bool:
        """Check if model has Gemma-like structure."""
        try:
            # Gemma uses similar structure to Llama but may have subtle differences
            return (
                hasattr(self.model, 'model') and
                hasattr(self.model.model, 'layers') and
                len(self.model.model.layers) > 0 and
                hasattr(self.model.model.layers[0], 'self_attn') and
                hasattr(self.model.model.layers[0], 'mlp')
            )
        except (AttributeError, IndexError):
            return False

    def _detect_llama(self) -> ComponentPaths:
        """Detect Llama architecture and return component paths."""
        num_layers = len(self.model.model.layers)

        logger.info(f"Detected Llama architecture with {num_layers} layers")

        return ComponentPaths(
            architecture="llama",
            num_layers=num_layers,
            layers_path="model.layers",
            q_proj="self_attn.q_proj",
            k_proj="self_attn.k_proj",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            gate_proj="mlp.gate_proj",
            up_proj="mlp.up_proj",
            down_proj="mlp.down_proj",
            input_layernorm="input_layernorm",
            post_attention_layernorm="post_attention_layernorm",
            norm_type="rmsnorm"
        )

    def _detect_mistral(self) -> ComponentPaths:
        """Detect Mistral architecture and return component paths."""
        # Mistral uses the same structure as Llama
        num_layers = len(self.model.model.layers)

        logger.info(f"Detected Mistral architecture with {num_layers} layers")

        return ComponentPaths(
            architecture="mistral",
            num_layers=num_layers,
            layers_path="model.layers",
            q_proj="self_attn.q_proj",
            k_proj="self_attn.k_proj",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            gate_proj="mlp.gate_proj",
            up_proj="mlp.up_proj",
            down_proj="mlp.down_proj",
            input_layernorm="input_layernorm",
            post_attention_layernorm="post_attention_layernorm",
            norm_type="rmsnorm"
        )

    def _detect_phi(self) -> ComponentPaths:
        """Detect Phi architecture and return component paths."""
        num_layers = len(self.model.model.layers)

        logger.info(f"Detected Phi architecture with {num_layers} layers")

        # Phi-3 uses similar structure to Llama but may have different MLP naming
        # Check if it has gate_proj or different naming
        has_gate_proj = hasattr(self.model.model.layers[0].mlp, 'gate_proj')

        if has_gate_proj:
            # Phi-3 style
            return ComponentPaths(
                architecture="phi",
                num_layers=num_layers,
                layers_path="model.layers",
                q_proj="self_attn.q_proj",
                k_proj="self_attn.k_proj",
                v_proj="self_attn.v_proj",
                o_proj="self_attn.o_proj",
                gate_proj="mlp.gate_proj",
                up_proj="mlp.up_proj",
                down_proj="mlp.down_proj",
                input_layernorm="input_layernorm",
                post_attention_layernorm="post_attention_layernorm",
                norm_type="layernorm"
            )
        else:
            # Older Phi style (fc1, fc2)
            return ComponentPaths(
                architecture="phi",
                num_layers=num_layers,
                layers_path="model.layers",
                q_proj="self_attn.q_proj",
                k_proj="self_attn.k_proj",
                v_proj="self_attn.v_proj",
                o_proj="self_attn.o_proj",
                gate_proj=None,
                up_proj="mlp.fc1",
                down_proj="mlp.fc2",
                input_layernorm="input_layernorm",
                post_attention_layernorm="post_attention_layernorm",
                norm_type="layernorm"
            )

    def _detect_qwen(self) -> ComponentPaths:
        """Detect Qwen architecture and return component paths."""
        num_layers = len(self.model.model.layers)

        logger.info(f"Detected Qwen architecture with {num_layers} layers")

        # Qwen2 uses similar structure to Llama
        return ComponentPaths(
            architecture="qwen",
            num_layers=num_layers,
            layers_path="model.layers",
            q_proj="self_attn.q_proj",
            k_proj="self_attn.k_proj",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            gate_proj="mlp.gate_proj",
            up_proj="mlp.up_proj",
            down_proj="mlp.down_proj",
            input_layernorm="input_layernorm",
            post_attention_layernorm="post_attention_layernorm",
            norm_type="rmsnorm"
        )

    def _detect_gemma(self) -> ComponentPaths:
        """Detect Gemma architecture and return component paths."""
        num_layers = len(self.model.model.layers)

        logger.info(f"Detected Gemma architecture with {num_layers} layers")

        # Gemma (including Gemma 2 and Gemma 3) uses similar structure to Llama
        # with RMSNorm and the same component naming
        return ComponentPaths(
            architecture="gemma",
            num_layers=num_layers,
            layers_path="model.layers",
            q_proj="self_attn.q_proj",
            k_proj="self_attn.k_proj",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            gate_proj="mlp.gate_proj",
            up_proj="mlp.up_proj",
            down_proj="mlp.down_proj",
            input_layernorm="input_layernorm",
            post_attention_layernorm="post_attention_layernorm",
            norm_type="rmsnorm"
        )

    def _detect_stablelm(self) -> ComponentPaths:
        """Detect StableLM architecture and return component paths."""
        num_layers = len(self.model.model.layers)

        logger.info(f"Detected StableLM architecture with {num_layers} layers")

        # StableLM uses similar structure to Llama but has different KV cache handling
        # Component paths are the same as Llama
        return ComponentPaths(
            architecture="stablelm",
            num_layers=num_layers,
            layers_path="model.layers",
            q_proj="self_attn.q_proj",
            k_proj="self_attn.k_proj",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            gate_proj="mlp.gate_proj",
            up_proj="mlp.up_proj",
            down_proj="mlp.down_proj",
            input_layernorm="input_layernorm",
            post_attention_layernorm="post_attention_layernorm",
            norm_type="layernorm"
        )

    def _fallback_detection(self) -> ComponentPaths:
        """
        Fallback detection for unknown architectures.

        Attempts to infer structure by inspecting the model.
        """
        logger.warning("Using fallback detection - may not work correctly")

        # Try to find layers
        layers = None
        layers_path = None

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            layers_path = "model.layers"
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
            layers_path = "transformer.h"
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
            layers_path = "layers"
        else:
            raise ValueError(
                "Could not detect model architecture. "
                "Model does not have recognized layer structure."
            )

        if not layers or len(layers) == 0:
            raise ValueError("Model has no layers")

        num_layers = len(layers)
        first_layer = layers[0]

        # Try to detect attention components
        if hasattr(first_layer, 'self_attn') or hasattr(first_layer, 'attn'):
            attn = first_layer.self_attn if hasattr(first_layer, 'self_attn') else first_layer.attn

            # Detect projection names
            q_proj = "self_attn.q_proj" if hasattr(attn, 'q_proj') else "attn.q_proj"
            k_proj = "self_attn.k_proj" if hasattr(attn, 'k_proj') else "attn.k_proj"
            v_proj = "self_attn.v_proj" if hasattr(attn, 'v_proj') else "attn.v_proj"
            o_proj = "self_attn.o_proj" if hasattr(attn, 'o_proj') else "attn.o_proj"
        else:
            logger.warning("Could not find attention components")
            q_proj = k_proj = v_proj = o_proj = "unknown"

        # Try to detect MLP components
        if hasattr(first_layer, 'mlp'):
            mlp = first_layer.mlp

            if hasattr(mlp, 'gate_proj'):
                gate_proj = "mlp.gate_proj"
                up_proj = "mlp.up_proj"
                down_proj = "mlp.down_proj"
            elif hasattr(mlp, 'fc1'):
                gate_proj = None
                up_proj = "mlp.fc1"
                down_proj = "mlp.fc2"
            else:
                logger.warning("Could not determine MLP structure")
                gate_proj = None
                up_proj = "mlp.unknown_up"
                down_proj = "mlp.unknown_down"
        else:
            logger.warning("Could not find MLP components")
            gate_proj = None
            up_proj = "unknown"
            down_proj = "unknown"

        # Try to detect layer norms
        input_layernorm = "input_layernorm" if hasattr(first_layer, 'input_layernorm') else "ln_1"
        post_attention_layernorm = "post_attention_layernorm" if hasattr(first_layer, 'post_attention_layernorm') else "ln_2"

        # Try to determine norm type
        norm_type = "layernorm"
        if hasattr(first_layer, 'input_layernorm'):
            norm_class_name = first_layer.input_layernorm.__class__.__name__.lower()
            if 'rms' in norm_class_name:
                norm_type = "rmsnorm"

        logger.info(f"Fallback detection: {num_layers} layers, norm_type={norm_type}")

        return ComponentPaths(
            architecture="unknown",
            num_layers=num_layers,
            layers_path=layers_path,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            norm_type=norm_type
        )


def detect_model_architecture(model: Any) -> ComponentPaths:
    """
    Convenience function to detect model architecture.

    Args:
        model: PyTorch model to detect

    Returns:
        ComponentPaths object with standardized paths
    """
    detector = ModelArchitectureDetector(model)
    return detector.detect()


def is_streaming_compatible(model: Any) -> bool:
    """
    Check if a model is compatible with TextIteratorStreamer.

    Some models (like StableLM) have issues with streaming generation due to
    how they handle KV cache and past_key_values.

    Args:
        model: PyTorch model to check

    Returns:
        True if model supports streaming, False otherwise
    """
    try:
        # Get model architecture
        components = detect_model_architecture(model)

        # Known incompatible architectures
        incompatible_archs = {'stablelm'}

        if components.architecture in incompatible_archs:
            logger.warning(
                f"Model architecture '{components.architecture}' is not compatible "
                "with streaming generation. Will use non-streaming mode."
            )
            return False

        return True

    except Exception as e:
        logger.warning(f"Could not determine streaming compatibility: {e}")
        # Default to streaming-compatible if we can't detect
        return True
