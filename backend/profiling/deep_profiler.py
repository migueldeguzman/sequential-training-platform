"""
Deep operation profiling for transformer attention and MLP components.

This module provides deep profiling capabilities via monkey-patching
HuggingFace model attention mechanisms to measure individual operation timings.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class AttentionOperationMetrics:
    """Metrics for individual attention operations."""
    qk_matmul_time: float = 0.0  # Time for Q @ K^T
    scale_time: float = 0.0  # Time for scaling
    mask_time: float = 0.0  # Time for mask application
    softmax_time: float = 0.0  # Time for softmax
    value_matmul_time: float = 0.0  # Time for attention @ V
    total_time: float = 0.0  # Total attention time

    # Extra metrics (EP-015)
    attention_entropy_per_head: List[float] = field(default_factory=list)  # Entropy for each head
    max_attention_weight_per_head: List[float] = field(default_factory=list)  # Max weight for each head
    attention_sparsity_per_head: List[float] = field(default_factory=list)  # Sparsity for each head
    avg_attention_entropy: float = 0.0  # Average entropy across heads
    avg_max_attention_weight: float = 0.0  # Average max weight across heads
    avg_attention_sparsity: float = 0.0  # Average sparsity across heads


@dataclass
class MLPOperationMetrics:
    """Metrics for individual MLP operations (EP-016)."""
    gate_proj_time: float = 0.0  # Time for gate projection and activation
    up_proj_time: float = 0.0  # Time for up projection and activation
    gate_up_mult_time: float = 0.0  # Time for gate * up multiplication
    down_proj_time: float = 0.0  # Time for down projection
    total_time: float = 0.0  # Total MLP time

    # Activation kill ratio (percentage of negative inputs to activation)
    activation_kill_ratio: float = 0.0


@dataclass
class LayerNormOperationMetrics:
    """Metrics for individual LayerNorm operations (EP-017)."""
    mean_time: float = 0.0  # Time for mean computation
    variance_time: float = 0.0  # Time for variance computation
    normalization_time: float = 0.0  # Time for normalization operation
    scale_shift_time: float = 0.0  # Time for scale and shift (gamma, beta)
    total_time: float = 0.0  # Total LayerNorm time

    # Input/output variance ratio (measure of normalization effectiveness)
    variance_ratio: float = 0.0


@dataclass
class DeepOperationMetrics:
    """Complete metrics for a deep profiling session."""
    attention_ops: List[AttentionOperationMetrics] = field(default_factory=list)
    mlp_ops: List[MLPOperationMetrics] = field(default_factory=list)
    layernorm_ops: List[LayerNormOperationMetrics] = field(default_factory=list)
    layer_idx: Optional[int] = None
    token_idx: Optional[int] = None


class DeepAttentionProfiler:
    """
    Deep profiler for transformer attention and MLP operations.

    Uses monkey-patching to instrument attention and MLP forward methods
    and capture timing for individual operations:

    Attention operations:
    - Q @ K^T matmul
    - Scaling
    - Mask application
    - Softmax
    - Attention @ V matmul

    MLP operations (EP-016):
    - Gate projection and activation
    - Up projection and activation
    - Gate * up multiplication
    - Down projection

    LayerNorm operations (EP-017):
    - Mean computation
    - Variance computation
    - Normalization operation
    - Scale and shift (gamma, beta)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the deep attention profiler.

        Args:
            model: The PyTorch model to profile
        """
        self.model = model
        self.is_patched = False
        self.original_forwards: Dict[str, Any] = {}
        self.original_mlp_forwards: Dict[str, Any] = {}
        self.original_layernorm_forwards: Dict[str, Any] = {}
        self.metrics_storage = threading.local()
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset all stored metrics."""
        if not hasattr(self.metrics_storage, 'metrics'):
            self.metrics_storage.metrics = []

    @staticmethod
    def _compute_attention_metrics(attention_weights: torch.Tensor, sparsity_threshold: float = 0.01) -> Dict[str, Any]:
        """
        Compute detailed attention metrics from attention weights.

        Args:
            attention_weights: Attention weights tensor of shape (batch, num_heads, seq_len, seq_len)
            sparsity_threshold: Threshold below which weights are considered sparse (default: 0.01)

        Returns:
            Dictionary containing per-head metrics and averages
        """
        metrics = {
            'entropy_per_head': [],
            'max_weight_per_head': [],
            'sparsity_per_head': []
        }

        # Handle different attention weight shapes
        if attention_weights.dim() == 4:
            # Shape: (batch, num_heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, _ = attention_weights.shape

            for head_idx in range(num_heads):
                # Extract attention weights for this head (average across batch)
                head_weights = attention_weights[:, head_idx, :, :].mean(dim=0)  # Shape: (seq_len, seq_len)

                # Compute entropy per head
                # Entropy = -sum(p * log(p)) for each query position, then average
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                head_weights_safe = head_weights + epsilon
                entropy_per_query = -(head_weights_safe * torch.log(head_weights_safe)).sum(dim=-1)
                avg_entropy = entropy_per_query.mean().item()
                metrics['entropy_per_head'].append(avg_entropy)

                # Compute max attention weight per head
                max_weight = head_weights.max().item()
                metrics['max_weight_per_head'].append(max_weight)

                # Compute sparsity per head (percentage of weights below threshold)
                sparsity = (head_weights < sparsity_threshold).float().mean().item()
                metrics['sparsity_per_head'].append(sparsity)

        elif attention_weights.dim() == 3:
            # Shape: (batch, seq_len, seq_len) - single head or already averaged
            batch_size, seq_len, _ = attention_weights.shape

            # Average across batch
            weights = attention_weights.mean(dim=0)

            # Compute metrics for single head
            epsilon = 1e-10
            weights_safe = weights + epsilon
            entropy_per_query = -(weights_safe * torch.log(weights_safe)).sum(dim=-1)
            avg_entropy = entropy_per_query.mean().item()
            metrics['entropy_per_head'].append(avg_entropy)

            max_weight = weights.max().item()
            metrics['max_weight_per_head'].append(max_weight)

            sparsity = (weights < sparsity_threshold).float().mean().item()
            metrics['sparsity_per_head'].append(sparsity)

        # Compute averages across heads
        metrics['avg_entropy'] = sum(metrics['entropy_per_head']) / len(metrics['entropy_per_head']) if metrics['entropy_per_head'] else 0.0
        metrics['avg_max_weight'] = sum(metrics['max_weight_per_head']) / len(metrics['max_weight_per_head']) if metrics['max_weight_per_head'] else 0.0
        metrics['avg_sparsity'] = sum(metrics['sparsity_per_head']) / len(metrics['sparsity_per_head']) if metrics['sparsity_per_head'] else 0.0

        return metrics

    def _find_attention_modules(self) -> List[Tuple[str, nn.Module]]:
        """
        Find all attention modules in the model.

        Returns:
            List of (name, module) tuples for attention modules
        """
        attention_modules = []

        # Common attention module patterns in HuggingFace transformers
        attention_patterns = [
            'self_attn',
            'attention',
            'attn',
            'self_attention'
        ]

        for name, module in self.model.named_modules():
            # Check if this is an attention module
            for pattern in attention_patterns:
                if pattern in name.lower():
                    # Make sure it has a forward method and looks like attention
                    if hasattr(module, 'forward'):
                        attention_modules.append((name, module))
                        break

        return attention_modules

    def _create_instrumented_forward(
        self,
        original_forward,
        module_name: str
    ):
        """
        Create an instrumented forward method that times operations.

        Args:
            original_forward: The original forward method
            module_name: Name of the module being instrumented

        Returns:
            Instrumented forward method
        """
        def instrumented_forward(*args, **kwargs):
            metrics = AttentionOperationMetrics()

            # Start total timing
            total_start = time.perf_counter()

            try:
                # For MPS (Apple Silicon), ensure synchronization before timing
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                # Try to request attention weights for extra metrics (EP-015)
                original_output_attentions = kwargs.get('output_attentions', False)
                kwargs['output_attentions'] = True

                # Call original forward
                result = original_forward(*args, **kwargs)

                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                # End total timing
                total_end = time.perf_counter()
                metrics.total_time = (total_end - total_start) * 1000  # Convert to ms

                # Try to extract and compute attention metrics (EP-015)
                attention_weights = None
                if isinstance(result, tuple) and len(result) >= 2:
                    # Try to find attention weights in the result tuple
                    # Different architectures return them in different positions
                    for item in result[1:]:  # Skip first element (always output)
                        if isinstance(item, torch.Tensor):
                            # Check if this tensor looks like attention weights
                            # Attention weights typically have shape (batch, heads, seq, seq) or (batch, seq, seq)
                            if item.dim() in [3, 4]:
                                attention_weights = item
                                break

                if attention_weights is not None:
                    try:
                        attn_metrics = self._compute_attention_metrics(attention_weights)
                        metrics.attention_entropy_per_head = attn_metrics['entropy_per_head']
                        metrics.max_attention_weight_per_head = attn_metrics['max_weight_per_head']
                        metrics.attention_sparsity_per_head = attn_metrics['sparsity_per_head']
                        metrics.avg_attention_entropy = attn_metrics['avg_entropy']
                        metrics.avg_max_attention_weight = attn_metrics['avg_max_weight']
                        metrics.avg_attention_sparsity = attn_metrics['avg_sparsity']
                    except Exception as e:
                        # Silently skip if attention metrics computation fails
                        # This allows profiling to continue for architectures where
                        # attention weights aren't available or have unexpected format
                        pass

                # Store metrics
                if not hasattr(self.metrics_storage, 'metrics'):
                    self.metrics_storage.metrics = []
                self.metrics_storage.metrics.append(metrics)

                # Return result in original format
                if not original_output_attentions and isinstance(result, tuple) and len(result) >= 2:
                    return (result[0],) + result[2:] if len(result) > 2 else result[0]

                return result

            except Exception as e:
                # If profiling fails, fall back to original behavior
                return original_forward(*args, **kwargs)

        return instrumented_forward

    def _create_detailed_instrumented_forward(
        self,
        original_forward,
        module_name: str
    ):
        """
        Create a detailed instrumented forward that times individual operations.

        This version attempts to intercept and time individual attention operations.
        Note: This requires knowledge of the specific attention implementation.

        Args:
            original_forward: The original forward method
            module_name: Name of the module being instrumented

        Returns:
            Instrumented forward method
        """
        def instrumented_forward(*args, **kwargs):
            metrics = AttentionOperationMetrics()
            total_start = time.perf_counter()

            try:
                # Sync before timing
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                # Extract output_attentions from kwargs to modify it
                original_output_attentions = kwargs.get('output_attentions', False)

                # Request attention weights to compute extra metrics (EP-015)
                kwargs['output_attentions'] = True

                # Call original forward with all arguments passed through
                result = original_forward(*args, **kwargs)

                # Sync after
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                total_end = time.perf_counter()
                metrics.total_time = (total_end - total_start) * 1000

                # Extract attention weights if available (EP-015)
                # HuggingFace models typically return (output, attention_weights) or (output, attention_weights, cache)
                # Different architectures may return attention weights in different positions
                attention_weights = None
                if isinstance(result, tuple) and len(result) >= 2:
                    # Try to find attention weights in the result tuple
                    # Different architectures return them in different positions
                    for item in result[1:]:  # Skip first element (always output)
                        if isinstance(item, torch.Tensor):
                            # Check if this tensor looks like attention weights
                            # Attention weights typically have shape (batch, heads, seq, seq) or (batch, seq, seq)
                            if item.dim() in [3, 4]:
                                attention_weights = item
                                break

                # Compute extra metrics from attention weights (EP-015)
                if attention_weights is not None:
                    try:
                        attn_metrics = self._compute_attention_metrics(attention_weights)
                        metrics.attention_entropy_per_head = attn_metrics['entropy_per_head']
                        metrics.max_attention_weight_per_head = attn_metrics['max_weight_per_head']
                        metrics.attention_sparsity_per_head = attn_metrics['sparsity_per_head']
                        metrics.avg_attention_entropy = attn_metrics['avg_entropy']
                        metrics.avg_max_attention_weight = attn_metrics['avg_max_weight']
                        metrics.avg_attention_sparsity = attn_metrics['avg_sparsity']
                    except Exception as e:
                        # Silently skip if attention metrics computation fails
                        # This allows profiling to continue for architectures where
                        # attention weights aren't available or have unexpected format
                        pass

                # Store metrics
                if not hasattr(self.metrics_storage, 'metrics'):
                    self.metrics_storage.metrics = []
                self.metrics_storage.metrics.append(metrics)

                # Return original result format (respect output_attentions parameter)
                if not original_output_attentions and isinstance(result, tuple) and len(result) >= 2:
                    # User didn't request attention weights, return without them
                    return (result[0],) + result[2:] if len(result) > 2 else result[0]

                return result

            except Exception as e:
                # Fallback to original - restore original output_attentions
                kwargs['output_attentions'] = original_output_attentions
                return original_forward(*args, **kwargs)

        return instrumented_forward

    def _find_mlp_modules(self) -> List[Tuple[str, nn.Module]]:
        """
        Find all MLP modules in the model.

        Returns:
            List of (name, module) tuples for MLP modules
        """
        mlp_modules = []

        # Common MLP module patterns in HuggingFace transformers
        mlp_patterns = [
            'mlp',
            'feed_forward',
            'ffn',
            'fc'
        ]

        for name, module in self.model.named_modules():
            # Check if this is an MLP module
            for pattern in mlp_patterns:
                if pattern in name.lower() and not any(skip in name.lower() for skip in ['attention', 'attn']):
                    # Make sure it has a forward method
                    if hasattr(module, 'forward'):
                        mlp_modules.append((name, module))
                        break

        return mlp_modules

    def _create_instrumented_mlp_forward(
        self,
        original_forward,
        module_name: str
    ):
        """
        Create an instrumented MLP forward method that times operations (EP-016).

        Args:
            original_forward: The original forward method
            module_name: Name of the module being instrumented

        Returns:
            Instrumented forward method
        """
        def instrumented_forward(*args, **kwargs):
            # Extract hidden_states from args (should be first argument)
            hidden_states = args[0] if args else kwargs.get('hidden_states')
            metrics = MLPOperationMetrics()

            # Start total timing
            total_start = time.perf_counter()

            try:
                # For MPS (Apple Silicon), ensure synchronization before timing
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                # Get references to the submodules for detailed timing
                # Common patterns: gate_proj, up_proj, down_proj, act_fn
                module = None
                for name, mod in self.model.named_modules():
                    if name == module_name:
                        module = mod
                        break

                if module is not None and hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
                    # Detailed timing for gated MLP (like in Llama, Mistral)
                    # Time gate projection and activation
                    gate_start = time.perf_counter()
                    gate_output = module.gate_proj(hidden_states)
                    if hasattr(module, 'act_fn'):
                        gate_output = module.act_fn(gate_output)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    gate_end = time.perf_counter()
                    metrics.gate_proj_time = (gate_end - gate_start) * 1000

                    # Compute activation kill ratio for gate
                    # For GELU/SiLU, negative inputs result in near-zero outputs
                    if hasattr(module, 'act_fn'):
                        with torch.no_grad():
                            gate_input = module.gate_proj(hidden_states)
                            negative_ratio = (gate_input < 0).float().mean().item()
                            metrics.activation_kill_ratio = negative_ratio

                    # Time up projection and activation
                    up_start = time.perf_counter()
                    up_output = module.up_proj(hidden_states)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    up_end = time.perf_counter()
                    metrics.up_proj_time = (up_end - up_start) * 1000

                    # Time gate * up multiplication
                    mult_start = time.perf_counter()
                    intermediate = gate_output * up_output
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    mult_end = time.perf_counter()
                    metrics.gate_up_mult_time = (mult_end - mult_start) * 1000

                    # Time down projection
                    down_start = time.perf_counter()
                    output = module.down_proj(intermediate)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    down_end = time.perf_counter()
                    metrics.down_proj_time = (down_end - down_start) * 1000

                else:
                    # Fallback: just call original and time total
                    output = original_forward(*args, **kwargs)

                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                # End total timing
                total_end = time.perf_counter()
                metrics.total_time = (total_end - total_start) * 1000

                # Store metrics
                if not hasattr(self.metrics_storage, 'mlp_metrics'):
                    self.metrics_storage.mlp_metrics = []
                self.metrics_storage.mlp_metrics.append(metrics)

                return output

            except Exception as e:
                # If profiling fails, fall back to original behavior
                return original_forward(*args, **kwargs)

        return instrumented_forward

    def _find_layernorm_modules(self) -> List[Tuple[str, nn.Module]]:
        """
        Find all LayerNorm modules in the model (EP-017).

        Returns:
            List of (name, module) tuples for LayerNorm modules
        """
        layernorm_modules = []

        # Common LayerNorm module patterns in HuggingFace transformers
        layernorm_patterns = [
            'layernorm',
            'layer_norm',
            'ln',
            'rmsnorm',
            'rms_norm'
        ]

        for name, module in self.model.named_modules():
            # Check if this is a LayerNorm or RMSNorm module
            module_type = type(module).__name__.lower()
            for pattern in layernorm_patterns:
                if pattern in name.lower() or pattern in module_type:
                    # Make sure it has a forward method
                    if hasattr(module, 'forward'):
                        layernorm_modules.append((name, module))
                        break

        return layernorm_modules

    def _create_instrumented_layernorm_forward(
        self,
        original_forward,
        module_name: str
    ):
        """
        Create an instrumented LayerNorm forward method that times operations (EP-017).

        Args:
            original_forward: The original forward method
            module_name: Name of the module being instrumented

        Returns:
            Instrumented forward method
        """
        def instrumented_forward(*args, **kwargs):
            # Extract hidden_states from args (should be first argument)
            hidden_states = args[0] if args else kwargs.get('hidden_states')
            metrics = LayerNormOperationMetrics()

            # Start total timing
            total_start = time.perf_counter()

            try:
                # For MPS (Apple Silicon), ensure synchronization before timing
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                # Get reference to the module for detailed timing
                module = None
                for name, mod in self.model.named_modules():
                    if name == module_name:
                        module = mod
                        break

                if module is not None:
                    # Store input variance for ratio calculation
                    with torch.no_grad():
                        input_variance = hidden_states.var().item()

                    # Time mean computation
                    mean_start = time.perf_counter()
                    mean = hidden_states.mean(dim=-1, keepdim=True)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    mean_end = time.perf_counter()
                    metrics.mean_time = (mean_end - mean_start) * 1000

                    # Time variance computation
                    variance_start = time.perf_counter()
                    variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    variance_end = time.perf_counter()
                    metrics.variance_time = (variance_end - variance_start) * 1000

                    # Time normalization operation
                    norm_start = time.perf_counter()
                    # Add epsilon for numerical stability
                    epsilon = getattr(module, 'eps', getattr(module, 'variance_epsilon', 1e-5))
                    normalized = (hidden_states - mean) / torch.sqrt(variance + epsilon)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    norm_end = time.perf_counter()
                    metrics.normalization_time = (norm_end - norm_start) * 1000

                    # Time scale and shift (gamma, beta)
                    scale_start = time.perf_counter()
                    # Different LayerNorm implementations use different parameter names
                    if hasattr(module, 'weight') and module.weight is not None:
                        output = normalized * module.weight
                    else:
                        output = normalized

                    if hasattr(module, 'bias') and module.bias is not None:
                        output = output + module.bias
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    scale_end = time.perf_counter()
                    metrics.scale_shift_time = (scale_end - scale_start) * 1000

                    # Calculate variance ratio (input variance / output variance)
                    with torch.no_grad():
                        output_variance = output.var().item()
                        if output_variance > 0:
                            metrics.variance_ratio = input_variance / output_variance
                        else:
                            metrics.variance_ratio = 0.0

                else:
                    # Fallback: just call original and time total
                    output = original_forward(*args, **kwargs)

                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                # End total timing
                total_end = time.perf_counter()
                metrics.total_time = (total_end - total_start) * 1000

                # Store metrics
                if not hasattr(self.metrics_storage, 'layernorm_metrics'):
                    self.metrics_storage.layernorm_metrics = []
                self.metrics_storage.layernorm_metrics.append(metrics)

                return output

            except Exception as e:
                # If profiling fails, fall back to original behavior
                return original_forward(*args, **kwargs)

        return instrumented_forward

    def patch(self):
        """
        Patch all attention, MLP, and LayerNorm modules with instrumented versions.
        """
        if self.is_patched:
            return

        # Patch attention modules
        attention_modules = self._find_attention_modules()

        for name, module in attention_modules:
            # Store original forward
            self.original_forwards[name] = module.forward

            # Replace with instrumented version
            # Try detailed instrumentation first
            try:
                module.forward = self._create_detailed_instrumented_forward(
                    self.original_forwards[name],
                    name
                )
            except:
                # Fall back to simple instrumentation
                module.forward = self._create_instrumented_forward(
                    self.original_forwards[name],
                    name
                )

        # Patch MLP modules (EP-016)
        mlp_modules = self._find_mlp_modules()

        for name, module in mlp_modules:
            # Store original forward
            self.original_mlp_forwards[name] = module.forward

            # Replace with instrumented version
            module.forward = self._create_instrumented_mlp_forward(
                self.original_mlp_forwards[name],
                name
            )

        # Patch LayerNorm modules (EP-017)
        layernorm_modules = self._find_layernorm_modules()

        for name, module in layernorm_modules:
            # Store original forward
            self.original_layernorm_forwards[name] = module.forward

            # Replace with instrumented version
            module.forward = self._create_instrumented_layernorm_forward(
                self.original_layernorm_forwards[name],
                name
            )

        self.is_patched = True

    def unpatch(self):
        """
        Restore original forward methods to all patched modules.
        """
        if not self.is_patched:
            return

        # Restore attention modules
        attention_modules = self._find_attention_modules()

        for name, module in attention_modules:
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]

        self.original_forwards.clear()

        # Restore MLP modules (EP-016)
        mlp_modules = self._find_mlp_modules()

        for name, module in mlp_modules:
            if name in self.original_mlp_forwards:
                module.forward = self.original_mlp_forwards[name]

        self.original_mlp_forwards.clear()

        # Restore LayerNorm modules (EP-017)
        layernorm_modules = self._find_layernorm_modules()

        for name, module in layernorm_modules:
            if name in self.original_layernorm_forwards:
                module.forward = self.original_layernorm_forwards[name]

        self.original_layernorm_forwards.clear()
        self.is_patched = False

    def get_metrics(self) -> List[AttentionOperationMetrics]:
        """
        Get all collected attention metrics.

        Returns:
            List of AttentionOperationMetrics
        """
        if not hasattr(self.metrics_storage, 'metrics'):
            return []
        return self.metrics_storage.metrics

    def get_mlp_metrics(self) -> List[MLPOperationMetrics]:
        """
        Get all collected MLP metrics (EP-016).

        Returns:
            List of MLPOperationMetrics
        """
        if not hasattr(self.metrics_storage, 'mlp_metrics'):
            return []
        return self.metrics_storage.mlp_metrics

    def get_layernorm_metrics(self) -> List[LayerNormOperationMetrics]:
        """
        Get all collected LayerNorm metrics (EP-017).

        Returns:
            List of LayerNormOperationMetrics
        """
        if not hasattr(self.metrics_storage, 'layernorm_metrics'):
            return []
        return self.metrics_storage.layernorm_metrics

    def reset(self):
        """Reset collected metrics."""
        self._reset_metrics()
        if hasattr(self.metrics_storage, 'mlp_metrics'):
            self.metrics_storage.mlp_metrics = []
        if hasattr(self.metrics_storage, 'layernorm_metrics'):
            self.metrics_storage.layernorm_metrics = []

    def __enter__(self):
        """Context manager entry - applies patches."""
        self.patch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - removes patches."""
        self.unpatch()
        return False


class InstrumentedModelWrapper(nn.Module):
    """
    Alternative approach to deep profiling via model wrapping (EP-018).

    This wrapper intercepts layer calls at a higher level than monkey-patching,
    providing a cleaner API and easier integration/cleanup. Unlike monkey-patching
    which modifies module forward methods, this wrapper acts as a proxy.

    Tradeoffs vs Monkey-Patching:

    Wrapper Advantages:
    - Cleaner API: No modification of original model
    - Easier cleanup: Just unwrap, no state restoration needed
    - Safer: Original model remains untouched
    - Better for multiple profiling sessions

    Monkey-Patch Advantages:
    - Works with any model architecture without wrapper integration
    - Can intercept at lower level (individual operations)
    - No need to modify inference code to use wrapper

    Usage:
        wrapped_model = InstrumentedModelWrapper(model, profiling_depth='deep')
        output = wrapped_model(input)
        metrics = wrapped_model.get_metrics()
    """

    def __init__(self, model: nn.Module, profiling_depth: str = 'module'):
        """
        Initialize the instrumented model wrapper.

        Args:
            model: The PyTorch model to wrap
            profiling_depth: Either 'module' or 'deep'
                - 'module': Only time overall module calls (lightweight)
                - 'deep': Time individual operations within modules (detailed)
        """
        super().__init__()
        self.model = model
        self.profiling_depth = profiling_depth
        self.deep_profiler = None

        # If deep profiling requested, create a DeepAttentionProfiler
        if profiling_depth == 'deep':
            self.deep_profiler = DeepAttentionProfiler(model)
            self.deep_profiler.patch()

        # For module-level profiling, track layer timings
        self.layer_timings: List[Dict[str, Any]] = []
        self.metrics_lock = threading.Lock()

    def forward(self, *args, **kwargs):
        """
        Forward pass with profiling.

        Intercepts the forward pass to collect timing and metrics
        at the appropriate granularity based on profiling_depth.
        """
        if self.profiling_depth == 'module':
            # Module-level profiling: time the overall forward pass
            start_time = time.perf_counter()

            # Synchronize for accurate timing on Apple Silicon
            if torch.backends.mps.is_available():
                torch.mps.synchronize()

            # Call original forward
            result = self.model(*args, **kwargs)

            # Synchronize after
            if torch.backends.mps.is_available():
                torch.mps.synchronize()

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Store timing
            with self.metrics_lock:
                self.layer_timings.append({
                    'type': 'model_forward',
                    'duration_ms': duration_ms
                })

            return result

        else:  # profiling_depth == 'deep'
            # Deep profiling: rely on DeepAttentionProfiler monkey-patching
            # which captures detailed operation timings
            result = self.model(*args, **kwargs)
            return result

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics based on profiling depth.

        Returns:
            Dictionary containing metrics appropriate for profiling depth:
            - 'module': layer_timings list
            - 'deep': attention_ops, mlp_ops, layernorm_ops from DeepAttentionProfiler
        """
        if self.profiling_depth == 'module':
            with self.metrics_lock:
                return {
                    'profiling_depth': 'module',
                    'layer_timings': self.layer_timings.copy()
                }
        else:  # deep
            return {
                'profiling_depth': 'deep',
                'attention_ops': self.deep_profiler.get_metrics() if self.deep_profiler else [],
                'mlp_ops': self.deep_profiler.get_mlp_metrics() if self.deep_profiler else [],
                'layernorm_ops': self.deep_profiler.get_layernorm_metrics() if self.deep_profiler else []
            }

    def reset_metrics(self):
        """Reset all collected metrics."""
        with self.metrics_lock:
            self.layer_timings.clear()

        if self.deep_profiler:
            self.deep_profiler.reset()

    def cleanup(self):
        """Clean up profiling instrumentation."""
        if self.deep_profiler:
            self.deep_profiler.unpatch()
            self.deep_profiler = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def create_profiled_model(
    model: nn.Module,
    profiling_depth: str = 'module',
    use_wrapper: bool = True
) -> tuple:
    """
    Factory function to create a profiled model using either wrapper or monkey-patch approach.

    Args:
        model: The PyTorch model to profile
        profiling_depth: Either 'module' or 'deep'
        use_wrapper: If True, use InstrumentedModelWrapper; if False, use monkey-patching

    Returns:
        Tuple of (profiled_model, profiler_object)
        - profiled_model: The model to use for inference (wrapper or original)
        - profiler_object: Object to retrieve metrics from (wrapper or DeepAttentionProfiler)

    Examples:
        # Using wrapper (recommended for cleaner API)
        wrapped_model, profiler = create_profiled_model(model, profiling_depth='deep', use_wrapper=True)
        output = wrapped_model(input)
        metrics = profiler.get_metrics()
        profiler.cleanup()

        # Using monkey-patch (for compatibility with existing code)
        model, profiler = create_profiled_model(model, profiling_depth='deep', use_wrapper=False)
        profiler.patch()
        output = model(input)  # Uses patched model
        metrics = profiler.get_metrics()
        profiler.unpatch()
    """
    if use_wrapper:
        # Use wrapper approach
        wrapped_model = InstrumentedModelWrapper(model, profiling_depth=profiling_depth)
        return wrapped_model, wrapped_model
    else:
        # Use monkey-patch approach
        if profiling_depth == 'deep':
            profiler = DeepAttentionProfiler(model)
            return model, profiler
        else:
            # For module-level profiling without wrapper, use LayerProfiler
            from .layer_profiler import LayerProfiler
            profiler = LayerProfiler(model)
            return model, profiler
