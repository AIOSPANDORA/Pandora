"""
Causal Mask - Light-Cone Attention Masking

This module provides causal masking for attention mechanisms
based on relativistic light-cone constraints.

References:
    - Vaswani, A., et al. (2017). Attention Is All You Need
    - Physical causality constraints on information flow
"""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpacetimePosition:
    """A position in spacetime."""
    time: float
    space: Tuple[float, ...]
    
    def spatial_distance(self, other: "SpacetimePosition") -> float:
        """Compute Euclidean distance in spatial dimensions."""
        if len(self.space) != len(other.space):
            raise ValueError("Spatial dimensions must match")
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.space, other.space)
        ))
        
    def time_difference(self, other: "SpacetimePosition") -> float:
        """Compute time difference (self.time - other.time)."""
        return self.time - other.time


def in_past_light_cone(
    observer: SpacetimePosition,
    event: SpacetimePosition,
    speed_of_light: float = 1.0
) -> bool:
    """
    Check if an event is in the observer's past light cone.
    
    An event is in the past light cone if:
    1. It happened before the observer (t_event <= t_observer)
    2. Light could have traveled from the event to the observer
       (spatial_distance <= c * time_difference)
    
    Args:
        observer: Observer's spacetime position
        event: Event's spacetime position
        speed_of_light: Maximum information propagation speed
        
    Returns:
        True if event is in observer's past light cone
    """
    dt = observer.time - event.time
    
    # Must be in the past
    if dt < 0:
        return False
        
    # Check light cone constraint
    dx = observer.spatial_distance(event)
    return dx <= speed_of_light * dt


def in_future_light_cone(
    observer: SpacetimePosition,
    event: SpacetimePosition,
    speed_of_light: float = 1.0
) -> bool:
    """
    Check if an event is in the observer's future light cone.
    
    Args:
        observer: Observer's spacetime position
        event: Event's spacetime position
        speed_of_light: Maximum information propagation speed
        
    Returns:
        True if event is in observer's future light cone
    """
    dt = event.time - observer.time
    
    # Must be in the future
    if dt < 0:
        return False
        
    dx = observer.spatial_distance(event)
    return dx <= speed_of_light * dt


class LightConeAttentionMask:
    """
    Generate attention masks based on light-cone causality.
    
    In transformer attention, position i attending to position j
    requires information flow from j to i. This is only possible
    if j is in i's past light cone.
    """
    
    def __init__(
        self,
        speed_of_light: float = 1.0,
        use_soft_mask: bool = False,
        temperature: float = 1.0
    ):
        """
        Initialize light-cone attention mask.
        
        Args:
            speed_of_light: Maximum information propagation speed
            use_soft_mask: If True, use soft (differentiable) masking
            temperature: Temperature for soft masking
        """
        self.speed_of_light = speed_of_light
        self.use_soft_mask = use_soft_mask
        self.temperature = temperature
        
    def compute_mask(
        self,
        positions: List[SpacetimePosition]
    ) -> List[List[float]]:
        """
        Compute attention mask for given positions.
        
        mask[i][j] = 1 if position i can attend to position j
                   = 0 otherwise
        
        Args:
            positions: List of spacetime positions for each token
            
        Returns:
            Attention mask matrix
        """
        n = len(positions)
        mask = []
        
        for i in range(n):
            row = []
            for j in range(n):
                if self.use_soft_mask:
                    # Soft mask based on distance from light cone
                    score = self._soft_light_cone_score(
                        positions[i], positions[j]
                    )
                else:
                    # Hard mask
                    score = 1.0 if in_past_light_cone(
                        positions[i], positions[j], self.speed_of_light
                    ) else 0.0
                row.append(score)
            mask.append(row)
            
        return mask
        
    def _soft_light_cone_score(
        self,
        observer: SpacetimePosition,
        event: SpacetimePosition
    ) -> float:
        """
        Compute soft light-cone score using sigmoid.
        
        Score approaches 1 deep inside past light cone,
        0 deep outside.
        """
        dt = observer.time - event.time
        dx = observer.spatial_distance(event)
        
        # Light cone boundary at dx = c * dt
        # Positive margin means inside light cone
        margin = self.speed_of_light * dt - dx
        
        # Sigmoid for smooth transition
        return 1.0 / (1.0 + math.exp(-margin / self.temperature))


class CausalMaskGenerator:
    """
    Generate various types of causal masks for attention.
    """
    
    @staticmethod
    def standard_causal_mask(sequence_length: int) -> List[List[float]]:
        """
        Standard autoregressive causal mask.
        
        Position i can attend to positions 0, 1, ..., i.
        
        Args:
            sequence_length: Length of sequence
            
        Returns:
            Lower triangular mask matrix
        """
        mask = []
        for i in range(sequence_length):
            row = [1.0 if j <= i else 0.0 for j in range(sequence_length)]
            mask.append(row)
        return mask
        
    @staticmethod
    def sliding_window_mask(
        sequence_length: int,
        window_size: int
    ) -> List[List[float]]:
        """
        Sliding window attention mask.
        
        Position i can attend to positions max(0, i-window_size+1) to i.
        
        Args:
            sequence_length: Length of sequence
            window_size: Size of attention window
            
        Returns:
            Banded lower triangular mask
        """
        mask = []
        for i in range(sequence_length):
            row = []
            for j in range(sequence_length):
                in_window = j <= i and j >= max(0, i - window_size + 1)
                row.append(1.0 if in_window else 0.0)
            mask.append(row)
        return mask
        
    @staticmethod
    def light_cone_mask(
        sequence_length: int,
        positions: List[SpacetimePosition] = None,
        speed_of_light: float = 1.0
    ) -> List[List[float]]:
        """
        Light-cone causal mask.
        
        Args:
            sequence_length: Length of sequence
            positions: Optional spacetime positions (defaults to linear time)
            speed_of_light: Information propagation speed
            
        Returns:
            Light-cone constrained mask
        """
        if positions is None:
            # Default: linear time, no spatial separation
            positions = [
                SpacetimePosition(time=float(i), space=(0.0,))
                for i in range(sequence_length)
            ]
            
        mask_gen = LightConeAttentionMask(speed_of_light=speed_of_light)
        return mask_gen.compute_mask(positions)


def apply_attention_mask(
    attention_scores: List[List[float]],
    mask: List[List[float]],
    mask_value: float = -1e9
) -> List[List[float]]:
    """
    Apply mask to attention scores.
    
    Masked positions (mask=0) are set to mask_value before softmax.
    
    Args:
        attention_scores: Raw attention scores [seq_len, seq_len]
        mask: Attention mask [seq_len, seq_len]
        mask_value: Value to use for masked positions
        
    Returns:
        Masked attention scores
    """
    result = []
    for i, (scores_row, mask_row) in enumerate(zip(attention_scores, mask)):
        new_row = []
        for score, m in zip(scores_row, mask_row):
            if m == 0:
                new_row.append(mask_value)
            else:
                # For soft masks, scale the score
                new_row.append(score)
        result.append(new_row)
    return result


def combine_masks(
    mask1: List[List[float]],
    mask2: List[List[float]],
    operation: str = "and"
) -> List[List[float]]:
    """
    Combine two masks.
    
    Args:
        mask1: First mask
        mask2: Second mask
        operation: "and" (min), "or" (max), or "multiply"
        
    Returns:
        Combined mask
    """
    n = len(mask1)
    result = []
    
    for i in range(n):
        row = []
        for j in range(len(mask1[i])):
            m1, m2 = mask1[i][j], mask2[i][j]
            if operation == "and":
                val = min(m1, m2)
            elif operation == "or":
                val = max(m1, m2)
            else:  # multiply
                val = m1 * m2
            row.append(val)
        result.append(row)
        
    return result
