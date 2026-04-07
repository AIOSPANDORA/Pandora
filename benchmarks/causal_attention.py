"""
Causal Attention - Light-Cone Masked Attention for Transformers

This module implements causal attention with light-cone masking,
where information can only propagate within causally connected regions.

References:
    - Vaswani, A., et al. (2017). Attention Is All You Need
    - Cohen, T., & Welling, M. (2016). Group Equivariant Convolutional Networks
"""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass


# Constants
MASK_VALUE = -1e9  # Value used to mask out invalid attention positions


@dataclass
class AttentionOutput:
    """Output of attention computation."""
    output: List[List[float]]
    attention_weights: List[List[float]]


def softmax(x: List[float]) -> List[float]:
    """Compute softmax of a list."""
    max_x = max(x) if x else 0
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    if sum_exp == 0:
        return [1.0 / len(x)] * len(x) if x else []
    return [e / sum_exp for e in exp_x]


def dot_product(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def matrix_vector_multiply(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Multiply matrix by vector."""
    return [dot_product(row, vector) for row in matrix]


class CausalMask:
    """Standard causal (autoregressive) attention mask."""
    
    def __init__(self, sequence_length: int):
        """
        Initialize causal mask.
        
        Args:
            sequence_length: Length of the sequence
        """
        self.sequence_length = sequence_length
        self._mask: Optional[List[List[float]]] = None
        
    def get_mask(self) -> List[List[float]]:
        """
        Get the causal mask matrix.
        
        Returns:
            Mask matrix where mask[i][j] = 0 if j > i, else 1
        """
        if self._mask is None:
            self._mask = []
            for i in range(self.sequence_length):
                row = []
                for j in range(self.sequence_length):
                    # Position i can only attend to positions <= i
                    row.append(1.0 if j <= i else 0.0)
                self._mask.append(row)
        return self._mask


class LightConeMask:
    """
    Light-cone causal attention mask.
    
    This mask allows attention only within the "light cone" - positions
    that are causally connected given a maximum information propagation speed.
    """
    
    def __init__(
        self,
        sequence_length: int,
        propagation_speed: float = 1.0,
        positions: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize light-cone mask.
        
        Args:
            sequence_length: Length of the sequence
            propagation_speed: Maximum speed of information propagation
            positions: Optional (time, space) coordinates for each token
        """
        self.sequence_length = sequence_length
        self.propagation_speed = propagation_speed
        
        # Default positions: linearly spaced in time, same spatial position
        if positions is None:
            self.positions = [(float(i), 0.0) for i in range(sequence_length)]
        else:
            self.positions = positions
            
        self._mask: Optional[List[List[float]]] = None
        
    def get_mask(self) -> List[List[float]]:
        """
        Get the light-cone mask matrix.
        
        Position i can attend to position j if j is within i's past light cone:
        |t_i - t_j| >= |x_i - x_j| / c (for t_j <= t_i)
        
        Returns:
            Light-cone mask matrix
        """
        if self._mask is None:
            self._mask = []
            for i in range(self.sequence_length):
                row = []
                t_i, x_i = self.positions[i]
                
                for j in range(self.sequence_length):
                    t_j, x_j = self.positions[j]
                    
                    # Must be in the past
                    if t_j > t_i:
                        row.append(0.0)
                        continue
                        
                    # Check light-cone constraint
                    time_diff = t_i - t_j
                    space_diff = abs(x_i - x_j)
                    
                    if self.propagation_speed > 0:
                        # Within light cone if time_diff >= space_diff / c
                        min_time_needed = space_diff / self.propagation_speed
                        if time_diff >= min_time_needed:
                            row.append(1.0)
                        else:
                            row.append(0.0)
                    else:
                        # Infinite speed: only causal constraint
                        row.append(1.0)
                        
                self._mask.append(row)
                
        return self._mask


class ScaledDotProductAttention:
    """
    Scaled dot-product attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, d_k: int):
        """
        Initialize attention.
        
        Args:
            d_k: Dimension of key vectors (for scaling)
        """
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k) if d_k > 0 else 1.0
        
    def forward(
        self,
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        mask: Optional[List[List[float]]] = None
    ) -> AttentionOutput:
        """
        Compute attention.
        
        Args:
            query: Query vectors [seq_len, d_k]
            key: Key vectors [seq_len, d_k]
            value: Value vectors [seq_len, d_v]
            mask: Optional attention mask [seq_len, seq_len]
            
        Returns:
            AttentionOutput with output and attention weights
        """
        seq_len = len(query)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                score = dot_product(query[i], key[j]) * self.scale
                row.append(score)
            scores.append(row)
            
        # Apply mask (set masked positions to large negative)
        if mask is not None:
            for i in range(seq_len):
                for j in range(seq_len):
                    if mask[i][j] == 0:
                        scores[i][j] = MASK_VALUE
                        
        # Compute attention weights via softmax
        attention_weights = [softmax(row) for row in scores]
        
        # Compute output: attention_weights * V
        output = []
        for i in range(seq_len):
            out_vec = [0.0] * len(value[0]) if value else []
            for j in range(seq_len):
                for k in range(len(out_vec)):
                    out_vec[k] += attention_weights[i][j] * value[j][k]
            output.append(out_vec)
            
        return AttentionOutput(output=output, attention_weights=attention_weights)


class CausalAttentionLayer:
    """Attention layer with causal masking."""
    
    def __init__(
        self,
        d_model: int,
        use_light_cone: bool = False,
        propagation_speed: float = 1.0
    ):
        """
        Initialize causal attention layer.
        
        Args:
            d_model: Model dimension
            use_light_cone: Whether to use light-cone masking
            propagation_speed: Speed for light-cone (if used)
        """
        self.d_model = d_model
        self.use_light_cone = use_light_cone
        self.propagation_speed = propagation_speed
        self.attention = ScaledDotProductAttention(d_model)
        
    def forward(
        self,
        x: List[List[float]],
        positions: Optional[List[Tuple[float, float]]] = None
    ) -> AttentionOutput:
        """
        Apply causal attention.
        
        Args:
            x: Input sequence [seq_len, d_model]
            positions: Optional (time, space) coordinates for light-cone
            
        Returns:
            AttentionOutput
        """
        seq_len = len(x)
        
        # Create mask
        if self.use_light_cone:
            mask_gen = LightConeMask(
                seq_len,
                self.propagation_speed,
                positions
            )
        else:
            mask_gen = CausalMask(seq_len)
            
        mask = mask_gen.get_mask()
        
        # For simplicity, use x as Q, K, V (self-attention)
        return self.attention.forward(x, x, x, mask)


def compare_attention_outputs(
    standard_output: AttentionOutput,
    lightcone_output: AttentionOutput
) -> dict:
    """
    Compare outputs from standard and light-cone attention.
    
    Args:
        standard_output: Output from standard causal attention
        lightcone_output: Output from light-cone attention
        
    Returns:
        Dictionary with comparison metrics
    """
    # Compute difference in attention weights
    n = len(standard_output.attention_weights)
    total_diff = 0.0
    mask_diff_count = 0
    
    for i in range(n):
        for j in range(n):
            w1 = standard_output.attention_weights[i][j]
            w2 = lightcone_output.attention_weights[i][j]
            diff = abs(w1 - w2)
            total_diff += diff
            if diff > 0.01:
                mask_diff_count += 1
                
    avg_diff = total_diff / (n * n) if n > 0 else 0
    
    return {
        "average_weight_difference": avg_diff,
        "positions_with_difference": mask_diff_count,
        "total_positions": n * n
    }
