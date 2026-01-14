"""
Inference Engine Detection for Energy Profiler.

Auto-detects the inference engine/backend being used for model execution.
Supports: transformers, mlx, vllm, etc.

Note: On Apple Silicon, options are more limited than NVIDIA (no TensorRT-LLM, DeepSpeed, etc.)
"""

import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def detect_inference_engine(model=None) -> str:
    """
    Detect which inference engine/backend is being used.

    Args:
        model: Optional model object to inspect for engine-specific attributes

    Returns:
        String identifying the inference engine (e.g., "transformers", "mlx", "vllm", "unknown")
    """
    # Check for MLX (Apple's ML framework for Apple Silicon)
    try:
        import mlx
        import mlx.core as mx
        # If mlx is imported and model uses mlx arrays, it's likely MLX
        if model is not None:
            if hasattr(model, '__module__') and 'mlx' in str(model.__module__):
                return "mlx"
        # Check if mlx is actively being used
        if 'mlx' in sys.modules:
            return "mlx"
    except ImportError:
        pass

    # Check for vLLM
    try:
        import vllm
        if model is not None:
            if hasattr(model, '__module__') and 'vllm' in str(model.__module__):
                return "vllm"
        if 'vllm' in sys.modules:
            return "vllm"
    except ImportError:
        pass

    # Check for llama.cpp (via llama-cpp-python)
    try:
        import llama_cpp
        if model is not None:
            if isinstance(model, type) and hasattr(model, '__module__'):
                if 'llama_cpp' in str(model.__module__):
                    return "llama-cpp"
        if 'llama_cpp' in sys.modules:
            return "llama-cpp"
    except ImportError:
        pass

    # Check for GGML/GGUF (standalone)
    try:
        import ggml
        if 'ggml' in sys.modules:
            return "ggml"
    except ImportError:
        pass

    # Check for Optimum (Hugging Face optimization library)
    try:
        import optimum
        if model is not None:
            if hasattr(model, '__module__') and 'optimum' in str(model.__module__):
                return "optimum"
        if 'optimum' in sys.modules:
            return "optimum"
    except ImportError:
        pass

    # Check for TGI (Text Generation Inference)
    try:
        import text_generation
        if 'text_generation' in sys.modules:
            return "text-generation-inference"
    except ImportError:
        pass

    # Check for CTransformers
    try:
        import ctransformers
        if model is not None:
            if hasattr(model, '__module__') and 'ctransformers' in str(model.__module__):
                return "ctransformers"
        if 'ctransformers' in sys.modules:
            return "ctransformers"
    except ImportError:
        pass

    # Check for ExLlama
    try:
        import exllama
        if model is not None:
            if hasattr(model, '__module__') and 'exllama' in str(model.__module__):
                return "exllama"
        if 'exllama' in sys.modules:
            return "exllama"
    except ImportError:
        pass

    # Check for ExLlamaV2
    try:
        import exllamav2
        if model is not None:
            if hasattr(model, '__module__') and 'exllamav2' in str(model.__module__):
                return "exllamav2"
        if 'exllamav2' in sys.modules:
            return "exllamav2"
    except ImportError:
        pass

    # Check for Hugging Face Transformers (default/fallback)
    try:
        import transformers
        if model is not None:
            # Check if it's a PreTrainedModel
            if hasattr(transformers, 'PreTrainedModel'):
                if isinstance(model, transformers.PreTrainedModel):
                    return "transformers"
            # Check module path
            if hasattr(model, '__module__') and 'transformers' in str(model.__module__):
                return "transformers"
        # If transformers is imported, it's likely being used
        if 'transformers' in sys.modules:
            return "transformers"
    except ImportError:
        pass

    # Check for PyTorch native (if no higher-level framework detected)
    try:
        import torch
        if model is not None:
            if isinstance(model, torch.nn.Module):
                # It's a PyTorch model but no specific inference framework
                return "pytorch-native"
        if 'torch' in sys.modules:
            return "pytorch-native"
    except ImportError:
        pass

    # Unknown engine
    logger.warning("Could not detect inference engine. Defaulting to 'unknown'")
    return "unknown"


def get_engine_info(engine: str) -> dict:
    """
    Get additional information about a detected inference engine.

    Args:
        engine: Engine identifier string

    Returns:
        Dictionary with engine metadata (name, description, platform_notes)
    """
    engine_info = {
        "mlx": {
            "name": "MLX",
            "description": "Apple's ML framework for Apple Silicon",
            "platform_notes": "Optimized for M-series chips with unified memory",
            "apple_silicon_native": True
        },
        "transformers": {
            "name": "Hugging Face Transformers",
            "description": "General-purpose transformer library",
            "platform_notes": "Uses PyTorch MPS backend on Apple Silicon",
            "apple_silicon_native": False
        },
        "vllm": {
            "name": "vLLM",
            "description": "Fast inference with PagedAttention",
            "platform_notes": "Limited Apple Silicon support",
            "apple_silicon_native": False
        },
        "llama-cpp": {
            "name": "llama.cpp",
            "description": "C++ inference engine for LLaMA models",
            "platform_notes": "Good Apple Silicon support with Metal",
            "apple_silicon_native": True
        },
        "ggml": {
            "name": "GGML",
            "description": "Tensor library for machine learning",
            "platform_notes": "Cross-platform, including Apple Silicon",
            "apple_silicon_native": True
        },
        "optimum": {
            "name": "Hugging Face Optimum",
            "description": "Hardware-specific optimizations",
            "platform_notes": "Various backends including ONNX Runtime",
            "apple_silicon_native": False
        },
        "text-generation-inference": {
            "name": "Text Generation Inference (TGI)",
            "description": "Production-ready inference server",
            "platform_notes": "Primarily NVIDIA-focused",
            "apple_silicon_native": False
        },
        "ctransformers": {
            "name": "CTransformers",
            "description": "Python bindings for GGML models",
            "platform_notes": "Cross-platform support",
            "apple_silicon_native": True
        },
        "exllama": {
            "name": "ExLlama",
            "description": "Fast inference for quantized models",
            "platform_notes": "NVIDIA CUDA only",
            "apple_silicon_native": False
        },
        "exllamav2": {
            "name": "ExLlamaV2",
            "description": "Improved version of ExLlama",
            "platform_notes": "NVIDIA CUDA only",
            "apple_silicon_native": False
        },
        "pytorch-native": {
            "name": "PyTorch Native",
            "description": "Direct PyTorch model without framework",
            "platform_notes": "Uses MPS backend on Apple Silicon",
            "apple_silicon_native": False
        },
        "unknown": {
            "name": "Unknown",
            "description": "Could not detect inference engine",
            "platform_notes": "N/A",
            "apple_silicon_native": False
        }
    }

    return engine_info.get(engine, engine_info["unknown"])


def is_apple_silicon_native(engine: str) -> bool:
    """
    Check if the engine is optimized for Apple Silicon.

    Args:
        engine: Engine identifier string

    Returns:
        True if engine is Apple Silicon native, False otherwise
    """
    info = get_engine_info(engine)
    return info.get("apple_silicon_native", False)
