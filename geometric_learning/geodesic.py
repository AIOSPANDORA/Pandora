"""
Geodesic - Geodesic Optimization Implementation

This module implements geodesic-based optimization on Riemannian manifolds.

References:
    - Bonnabel, S. (2013). Stochastic Gradient Descent on Riemannian Manifolds
    - Absil, P.A., et al. (2008). Optimization Algorithms on Matrix Manifolds
"""

import math
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

from .manifold import Manifold, ManifoldPoint, EuclideanManifold
from .metric import RiemannianMetric, EuclideanMetric


@dataclass
class GeodesicPath:
    """A geodesic path on a manifold."""
    start: ManifoldPoint
    end: ManifoldPoint
    initial_velocity: List[float]
    length: float


class GeodesicComputer:
    """
    Compute geodesics on Riemannian manifolds.
    
    A geodesic is a curve that locally minimizes distance,
    generalizing straight lines to curved spaces.
    """
    
    def __init__(self, manifold: Manifold, metric: RiemannianMetric = None):
        """
        Initialize geodesic computer.
        
        Args:
            manifold: The manifold to compute geodesics on
            metric: Optional Riemannian metric (uses Euclidean if None)
        """
        self.manifold = manifold
        self.metric = metric or EuclideanMetric(manifold.dimension)
        
    def geodesic(
        self,
        start: ManifoldPoint,
        end: ManifoldPoint,
        num_steps: int = 10
    ) -> List[ManifoldPoint]:
        """
        Compute geodesic path between two points.
        
        Uses the exponential map to trace the geodesic.
        
        Args:
            start: Starting point
            end: Ending point
            num_steps: Number of points along geodesic
            
        Returns:
            List of points along the geodesic
        """
        # Get initial velocity via logarithmic map
        velocity = self.manifold.logarithmic_map(start, end)
        
        # Trace geodesic using exponential map
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            scaled_velocity = [v * t for v in velocity]
            point = self.manifold.exponential_map(start, scaled_velocity)
            path.append(point)
            
        return path
        
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
        velocity = self.manifold.logarithmic_map(p1, p2)
        return self.metric.norm(p1.coordinates, velocity)
        
    def parallel_transport_along_geodesic(
        self,
        vector: List[float],
        start: ManifoldPoint,
        end: ManifoldPoint
    ) -> List[float]:
        """
        Parallel transport a vector along a geodesic.
        
        Args:
            vector: Tangent vector at start
            start: Starting point
            end: Ending point
            
        Returns:
            Transported vector at end
        """
        return self.manifold.parallel_transport(vector, start, end)


class RiemannianGradientDescent:
    """
    Gradient descent on Riemannian manifolds.
    
    Updates follow geodesics in the direction of steepest descent
    with respect to the Riemannian metric.
    """
    
    def __init__(
        self,
        manifold: Manifold,
        metric: RiemannianMetric,
        learning_rate: float = 0.01,
        momentum: float = 0.0
    ):
        """
        Initialize Riemannian gradient descent.
        
        Args:
            manifold: Manifold to optimize on
            metric: Riemannian metric defining geometry
            learning_rate: Step size
            momentum: Momentum coefficient
        """
        self.manifold = manifold
        self.metric = metric
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self._velocity: Optional[List[float]] = None
        self._current_point: Optional[ManifoldPoint] = None
        
        self.iterations = 0
        
    def step(
        self,
        point: ManifoldPoint,
        euclidean_gradient: List[float]
    ) -> ManifoldPoint:
        """
        Take one optimization step.
        
        Args:
            point: Current point on manifold
            euclidean_gradient: Euclidean gradient of objective
            
        Returns:
            New point on manifold
        """
        # Convert to Riemannian gradient
        riemannian_grad = self.metric.gradient_to_natural(
            point.coordinates,
            euclidean_gradient
        )
        
        # Apply momentum
        if self._velocity is None or self._current_point is None:
            self._velocity = [0.0] * len(riemannian_grad)
        else:
            # Parallel transport previous velocity to current point
            self._velocity = self.manifold.parallel_transport(
                self._velocity,
                self._current_point,
                point
            )
            
        # Update velocity with momentum
        for i in range(len(self._velocity)):
            self._velocity[i] = (
                self.momentum * self._velocity[i] - 
                self.learning_rate * riemannian_grad[i]
            )
            
        # Move along geodesic
        new_point = self.manifold.exponential_map(point, self._velocity)
        self._current_point = point
        self.iterations += 1
        
        return new_point
        
    def optimize(
        self,
        initial_point: ManifoldPoint,
        gradient_fn: Callable[[ManifoldPoint], List[float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> ManifoldPoint:
        """
        Run optimization until convergence.
        
        Args:
            initial_point: Starting point
            gradient_fn: Function returning Euclidean gradient
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimized point
        """
        point = initial_point
        
        for _ in range(max_iterations):
            gradient = gradient_fn(point)
            grad_norm = self.metric.norm(point.coordinates, gradient)
            
            if grad_norm < tolerance:
                break
                
            point = self.step(point, gradient)
            
        return point
        
    def reset(self):
        """Reset optimizer state."""
        self._velocity = None
        self._current_point = None
        self.iterations = 0


class NaturalGradientOptimizer:
    """
    Natural gradient optimizer using Fisher Information.
    
    This is equivalent to following geodesics on the statistical manifold
    of probability distributions.
    """
    
    DEFAULT_FISHER_EMA_BETA = 0.99
    
    def __init__(
        self,
        dimension: int,
        learning_rate: float = 0.01,
        damping: float = 0.001,
        momentum: float = 0.9,
        fisher_ema_beta: float = None
    ):
        """
        Initialize natural gradient optimizer.
        
        Args:
            dimension: Parameter dimension
            learning_rate: Learning rate
            damping: Damping for Fisher matrix inversion
            momentum: Momentum coefficient
            fisher_ema_beta: Beta for exponential moving average of Fisher (default: 0.99)
        """
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.damping = damping
        self.momentum = momentum
        
        self._velocity: Optional[List[float]] = None
        self._fisher_diag: Optional[List[float]] = None
        
        # For running average of Fisher
        self._fisher_ema_beta = fisher_ema_beta if fisher_ema_beta is not None else self.DEFAULT_FISHER_EMA_BETA
        
        self.iterations = 0
        
    def update_fisher(self, gradient: List[float]):
        """
        Update Fisher diagonal estimate with new gradient.
        
        Uses exponential moving average of gradient outer products.
        """
        if self._fisher_diag is None:
            self._fisher_diag = [g * g + self.damping for g in gradient]
        else:
            for i in range(self.dimension):
                self._fisher_diag[i] = (
                    self._fisher_ema_beta * self._fisher_diag[i] +
                    (1 - self._fisher_ema_beta) * gradient[i] ** 2
                )
                
    def step(
        self,
        parameters: List[float],
        gradient: List[float]
    ) -> List[float]:
        """
        Take one natural gradient step.
        
        Args:
            parameters: Current parameters
            gradient: Euclidean gradient
            
        Returns:
            Updated parameters
        """
        # Update Fisher estimate
        self.update_fisher(gradient)
        
        # Compute natural gradient
        natural_grad = []
        for i in range(self.dimension):
            ng = gradient[i] / (self._fisher_diag[i] + self.damping)
            natural_grad.append(ng)
            
        # Apply momentum
        if self._velocity is None:
            self._velocity = [0.0] * self.dimension
            
        for i in range(self.dimension):
            self._velocity[i] = (
                self.momentum * self._velocity[i] + 
                natural_grad[i]
            )
            
        # Update parameters
        new_params = []
        for i in range(self.dimension):
            new_params.append(parameters[i] - self.learning_rate * self._velocity[i])
            
        self.iterations += 1
        return new_params
        
    def reset(self):
        """Reset optimizer state."""
        self._velocity = None
        self._fisher_diag = None
        self.iterations = 0


def compute_geodesic_interpolation(
    manifold: Manifold,
    points: List[ManifoldPoint],
    num_interp: int = 5
) -> List[ManifoldPoint]:
    """
    Compute geodesic interpolation through a sequence of points.
    
    Args:
        manifold: The manifold
        points: Sequence of points to interpolate
        num_interp: Number of interpolation points between each pair
        
    Returns:
        Interpolated path
    """
    if len(points) < 2:
        return points
        
    computer = GeodesicComputer(manifold)
    result = []
    
    for i in range(len(points) - 1):
        segment = computer.geodesic(points[i], points[i + 1], num_interp)
        # Avoid duplicating endpoints
        if i == 0:
            result.extend(segment)
        else:
            result.extend(segment[1:])
            
    return result
