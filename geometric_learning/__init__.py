"""
Geometric Learning Package

This package provides geometric deep learning primitives including:
- Manifold representations
- Riemannian metrics (including Fisher Information)
- Geodesic optimization
- Causal (light-cone) attention masking

References:
    - Amari, S. (1998). Natural Gradient Works Efficiently in Learning
    - Bronstein, M. et al. (2017). Geometric Deep Learning
    - Bonnabel, S. (2013). Stochastic Gradient Descent on Riemannian Manifolds
"""

from .manifold import (
    Manifold,
    ManifoldPoint,
    EuclideanManifold,
    SphereManifold,
    LearnedManifold
)

from .metric import (
    RiemannianMetric,
    MetricTensor,
    EuclideanMetric,
    FisherInformationMetric,
    DiagonalFisherMetric,
    LearnedMetric
)

from .geodesic import (
    GeodesicComputer,
    GeodesicPath,
    RiemannianGradientDescent,
    NaturalGradientOptimizer,
    compute_geodesic_interpolation
)

from .causal_mask import (
    SpacetimePosition,
    LightConeAttentionMask,
    CausalMaskGenerator,
    in_past_light_cone,
    in_future_light_cone,
    apply_attention_mask,
    combine_masks
)

__all__ = [
    # Manifolds
    "Manifold",
    "ManifoldPoint",
    "EuclideanManifold",
    "SphereManifold",
    "LearnedManifold",
    
    # Metrics
    "RiemannianMetric",
    "MetricTensor",
    "EuclideanMetric",
    "FisherInformationMetric",
    "DiagonalFisherMetric",
    "LearnedMetric",
    
    # Geodesics
    "GeodesicComputer",
    "GeodesicPath",
    "RiemannianGradientDescent",
    "NaturalGradientOptimizer",
    "compute_geodesic_interpolation",
    
    # Causal Masks
    "SpacetimePosition",
    "LightConeAttentionMask",
    "CausalMaskGenerator",
    "in_past_light_cone",
    "in_future_light_cone",
    "apply_attention_mask",
    "combine_masks"
]
