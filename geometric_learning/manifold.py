"""
Manifold - Learned Manifold Representation

This module implements learned manifold representations for
geometric deep learning.

References:
    - Bronstein, M. et al. (2017). Geometric Deep Learning
    - Cohen, T., & Welling, M. (2016). Group Equivariant CNNs
"""

import math
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class ManifoldPoint:
    """A point on the manifold with local coordinates."""
    coordinates: List[float]
    tangent_basis: Optional[List[List[float]]] = None
    
    @property
    def dimension(self) -> int:
        return len(self.coordinates)


class Manifold:
    """
    Abstract base class for differentiable manifolds.
    
    A manifold is a topological space that locally resembles
    Euclidean space. This class provides the interface for
    Riemannian operations.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize manifold.
        
        Args:
            dimension: Intrinsic dimension of the manifold
        """
        self.dimension = dimension
        
    def project(self, point: List[float]) -> ManifoldPoint:
        """
        Project a point from ambient space onto the manifold.
        
        Args:
            point: Point in ambient space
            
        Returns:
            ManifoldPoint on the manifold
        """
        raise NotImplementedError
        
    def exponential_map(
        self,
        base: ManifoldPoint,
        tangent: List[float]
    ) -> ManifoldPoint:
        """
        Exponential map: move from base point along tangent vector.
        
        Args:
            base: Starting point on manifold
            tangent: Tangent vector at base
            
        Returns:
            New point on manifold
        """
        raise NotImplementedError
        
    def logarithmic_map(
        self,
        base: ManifoldPoint,
        target: ManifoldPoint
    ) -> List[float]:
        """
        Logarithmic map: tangent vector from base to target.
        
        Args:
            base: Starting point
            target: Target point
            
        Returns:
            Tangent vector at base pointing toward target
        """
        raise NotImplementedError
        
    def parallel_transport(
        self,
        vector: List[float],
        path_start: ManifoldPoint,
        path_end: ManifoldPoint
    ) -> List[float]:
        """
        Parallel transport a vector along a geodesic.
        
        Args:
            vector: Tangent vector at path_start
            path_start: Starting point
            path_end: Ending point
            
        Returns:
            Transported vector at path_end
        """
        raise NotImplementedError
        
    def geodesic_distance(
        self,
        p1: ManifoldPoint,
        p2: ManifoldPoint
    ) -> float:
        """
        Compute geodesic distance between two points.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Geodesic distance
        """
        log_vec = self.logarithmic_map(p1, p2)
        return math.sqrt(sum(v * v for v in log_vec))


class EuclideanManifold(Manifold):
    """
    Euclidean space as a (trivial) manifold.
    
    This serves as a baseline and for testing.
    """
    
    def __init__(self, dimension: int):
        super().__init__(dimension)
        
    def project(self, point: List[float]) -> ManifoldPoint:
        """Identity projection for Euclidean space."""
        return ManifoldPoint(coordinates=point.copy())
        
    def exponential_map(
        self,
        base: ManifoldPoint,
        tangent: List[float]
    ) -> ManifoldPoint:
        """Exponential map is simple addition in Euclidean space."""
        new_coords = [b + t for b, t in zip(base.coordinates, tangent)]
        return ManifoldPoint(coordinates=new_coords)
        
    def logarithmic_map(
        self,
        base: ManifoldPoint,
        target: ManifoldPoint
    ) -> List[float]:
        """Logarithmic map is simple subtraction in Euclidean space."""
        return [t - b for b, t in zip(base.coordinates, target.coordinates)]
        
    def parallel_transport(
        self,
        vector: List[float],
        path_start: ManifoldPoint,
        path_end: ManifoldPoint
    ) -> List[float]:
        """Parallel transport is identity in Euclidean space."""
        return vector.copy()


class SphereManifold(Manifold):
    """
    n-sphere embedded in (n+1)-dimensional Euclidean space.
    
    Points are constrained to ||x|| = 1.
    """
    
    def __init__(self, ambient_dimension: int):
        """
        Initialize sphere manifold.
        
        Args:
            ambient_dimension: Dimension of embedding space (n+1 for S^n)
        """
        super().__init__(ambient_dimension - 1)
        self.ambient_dimension = ambient_dimension
        
    def _normalize(self, point: List[float]) -> List[float]:
        """Normalize to unit sphere."""
        norm = math.sqrt(sum(x * x for x in point))
        if norm < 1e-10:
            # Return north pole as default
            result = [0.0] * len(point)
            result[0] = 1.0
            return result
        return [x / norm for x in point]
        
    def project(self, point: List[float]) -> ManifoldPoint:
        """Project onto unit sphere."""
        normalized = self._normalize(point)
        return ManifoldPoint(coordinates=normalized)
        
    def exponential_map(
        self,
        base: ManifoldPoint,
        tangent: List[float]
    ) -> ManifoldPoint:
        """
        Exponential map on sphere.
        
        exp_p(v) = cos(||v||)p + sin(||v||)(v/||v||)
        """
        p = base.coordinates
        v_norm = math.sqrt(sum(x * x for x in tangent))
        
        if v_norm < 1e-10:
            return ManifoldPoint(coordinates=p.copy())
            
        cos_t = math.cos(v_norm)
        sin_t = math.sin(v_norm)
        
        result = []
        for i in range(len(p)):
            result.append(cos_t * p[i] + sin_t * tangent[i] / v_norm)
            
        # Ensure on sphere
        return ManifoldPoint(coordinates=self._normalize(result))
        
    def logarithmic_map(
        self,
        base: ManifoldPoint,
        target: ManifoldPoint
    ) -> List[float]:
        """
        Logarithmic map on sphere.
        
        log_p(q) = (theta / sin(theta)) * (q - cos(theta) * p)
        where cos(theta) = p · q
        """
        p = base.coordinates
        q = target.coordinates
        
        # Compute inner product
        dot = sum(a * b for a, b in zip(p, q))
        dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
        
        theta = math.acos(dot)
        
        if abs(theta) < 1e-10:
            return [0.0] * len(p)
            
        sin_theta = math.sin(theta)
        if abs(sin_theta) < 1e-10:
            # Antipodal points - return any tangent direction
            return [0.0] * len(p)
            
        factor = theta / sin_theta
        
        result = []
        for i in range(len(p)):
            result.append(factor * (q[i] - dot * p[i]))
            
        return result
        
    def parallel_transport(
        self,
        vector: List[float],
        path_start: ManifoldPoint,
        path_end: ManifoldPoint
    ) -> List[float]:
        """
        Parallel transport on sphere along geodesic.
        """
        p = path_start.coordinates
        q = path_end.coordinates
        
        # Get geodesic tangent
        log_pq = self.logarithmic_map(path_start, path_end)
        log_norm = math.sqrt(sum(x * x for x in log_pq))
        
        if log_norm < 1e-10:
            return vector.copy()
            
        # Normalize tangent direction
        u = [x / log_norm for x in log_pq]
        
        # Component along geodesic direction
        v_parallel = sum(a * b for a, b in zip(vector, u))
        
        # Perpendicular component (stays the same)
        v_perp = [v - v_parallel * ui for v, ui in zip(vector, u)]
        
        # Rotated parallel component
        theta = log_norm
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        result = []
        for i in range(len(vector)):
            # New tangent direction at q
            u_q_i = -sin_t * p[i] + cos_t * u[i]
            result.append(v_perp[i] + v_parallel * u_q_i)
            
        return result


class LearnedManifold(Manifold):
    """
    A manifold with learned metric and geometry.
    
    This uses a neural network-style parameterization to learn
    the local geometry from data.
    """
    
    def __init__(
        self,
        dimension: int,
        embedding_dimension: int = None
    ):
        """
        Initialize learned manifold.
        
        Args:
            dimension: Intrinsic dimension
            embedding_dimension: Dimension of embedding (default: 2 * dimension)
        """
        super().__init__(dimension)
        self.embedding_dimension = embedding_dimension or 2 * dimension
        
        # Learnable parameters (initialized to identity-like)
        # In a real implementation, these would be neural network weights
        self._encoder_weights: List[List[float]] = [
            [1.0 if i == j else 0.0 for j in range(dimension)]
            for i in range(self.embedding_dimension)
        ]
        self._decoder_weights: List[List[float]] = [
            [1.0 if i == j else 0.0 for j in range(self.embedding_dimension)]
            for i in range(dimension)
        ]
        
    def project(self, point: List[float]) -> ManifoldPoint:
        """Project through encoder then decoder."""
        # Encode
        encoded = self._apply_linear(point[:self.dimension], self._encoder_weights)
        # Decode back
        decoded = self._apply_linear(encoded, self._decoder_weights)
        return ManifoldPoint(coordinates=decoded)
        
    def _apply_linear(
        self,
        x: List[float],
        weights: List[List[float]]
    ) -> List[float]:
        """Apply linear transformation."""
        result = []
        for row in weights:
            val = sum(w * xi for w, xi in zip(row, x))
            result.append(val)
        return result
        
    def exponential_map(
        self,
        base: ManifoldPoint,
        tangent: List[float]
    ) -> ManifoldPoint:
        """First-order exponential map approximation."""
        new_coords = [b + t for b, t in zip(base.coordinates, tangent)]
        return self.project(new_coords)
        
    def logarithmic_map(
        self,
        base: ManifoldPoint,
        target: ManifoldPoint
    ) -> List[float]:
        """First-order logarithmic map approximation."""
        return [t - b for b, t in zip(base.coordinates, target.coordinates)]
        
    def parallel_transport(
        self,
        vector: List[float],
        path_start: ManifoldPoint,
        path_end: ManifoldPoint
    ) -> List[float]:
        """First-order parallel transport (identity for flat approximation)."""
        return vector.copy()
        
    def set_encoder_weights(self, weights: List[List[float]]):
        """Set encoder weights (for learning)."""
        self._encoder_weights = weights
        
    def set_decoder_weights(self, weights: List[List[float]]):
        """Set decoder weights (for learning)."""
        self._decoder_weights = weights
