"""
Benchmarks package for Pandora AIOS.

This package provides benchmark implementations for validating
scientific claims in the framework.
"""

from .geodesic_optimizer import GeodesicOptimizer, AdamBaseline, SGDBaseline
from .causal_attention import (
    CausalMask,
    LightConeMask,
    ScaledDotProductAttention,
    CausalAttentionLayer
)
from .benchmark_runner import (
    run_all_benchmarks,
    run_attention_benchmark,
    print_results
)

__all__ = [
    "GeodesicOptimizer",
    "AdamBaseline",
    "SGDBaseline",
    "CausalMask",
    "LightConeMask",
    "ScaledDotProductAttention",
    "CausalAttentionLayer",
    "run_all_benchmarks",
    "run_attention_benchmark",
    "print_results"
]
