"""
Metric - Fisher Information Metric and Learned Metrics

This module implements Riemannian metrics on parameter manifolds,
including the Fisher Information Metric.

References:
    - Fisher, R.A. (1925). Theory of Statistical Estimation
    - Rao, C.R. (1945). Information and Accuracy Attainable in Estimation
    - Amari, S. (1998). Natural Gradient Works Efficiently in Learning
"""

import math
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class MetricTensor:
    """Riemannian metric tensor at a point."""
    matrix: List[List[float]]
    dimension: int
    
    @property
    def is_positive_definite(self) -> bool:
        """Check if metric is positive definite (required for Riemannian)."""
        # Simple check: all diagonal elements positive
        # Full check would require eigenvalue computation
        for i in range(self.dimension):
            if self.matrix[i][i] <= 0:
                return False
        return True


class RiemannianMetric:
    """
    Base class for Riemannian metrics.
    
    A Riemannian metric defines an inner product on tangent spaces,
    enabling distance computation and geodesic calculations.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize metric.
        
        Args:
            dimension: Dimension of the manifold
        """
        self.dimension = dimension
        
    def metric_tensor(self, point: List[float]) -> MetricTensor:
        """
        Compute metric tensor at a point.
        
        Args:
            point: Coordinates on manifold
            
        Returns:
            MetricTensor at the point
        """
        raise NotImplementedError
        
    def inner_product(
        self,
        point: List[float],
        v1: List[float],
        v2: List[float]
    ) -> float:
        """
        Compute inner product of two tangent vectors.
        
        <v1, v2>_p = v1^T G(p) v2
        
        Args:
            point: Base point
            v1: First tangent vector
            v2: Second tangent vector
            
        Returns:
            Inner product value
        """
        g = self.metric_tensor(point)
        
        result = 0.0
        for i in range(self.dimension):
            for j in range(self.dimension):
                result += v1[i] * g.matrix[i][j] * v2[j]
                
        return result
        
    def norm(self, point: List[float], vector: List[float]) -> float:
        """
        Compute norm of a tangent vector.
        
        Args:
            point: Base point
            vector: Tangent vector
            
        Returns:
            Norm ||vector||_p
        """
        return math.sqrt(max(0, self.inner_product(point, vector, vector)))
        
    def gradient_to_natural(
        self,
        point: List[float],
        gradient: List[float]
    ) -> List[float]:
        """
        Convert Euclidean gradient to natural (Riemannian) gradient.
        
        natural_grad = G^{-1} @ gradient
        
        Args:
            point: Current point
            gradient: Euclidean gradient
            
        Returns:
            Natural gradient
        """
        g = self.metric_tensor(point)
        g_inv = self._invert_matrix(g.matrix)
        
        # Multiply G^{-1} @ gradient
        result = []
        for i in range(self.dimension):
            val = sum(g_inv[i][j] * gradient[j] for j in range(self.dimension))
            result.append(val)
            
        return result
        
    def _invert_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        """
        Invert a symmetric positive definite matrix.
        Uses simple Gaussian elimination for small matrices.
        """
        n = len(matrix)
        
        # Create augmented matrix [A|I]
        aug = []
        for i in range(n):
            row = matrix[i].copy()
            for j in range(n):
                row.append(1.0 if i == j else 0.0)
            aug.append(row)
            
        # Forward elimination
        for i in range(n):
            # Find pivot
            max_val = abs(aug[i][i])
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > max_val:
                    max_val = abs(aug[k][i])
                    max_row = k
                    
            # Swap rows
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Check for singularity
            if abs(aug[i][i]) < 1e-10:
                # Add regularization
                aug[i][i] = 1e-10
                
            # Eliminate column
            for k in range(n):
                if k != i:
                    factor = aug[k][i] / aug[i][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
                        
        # Normalize rows
        for i in range(n):
            divisor = aug[i][i]
            for j in range(2 * n):
                aug[i][j] /= divisor
                
        # Extract inverse
        result = []
        for i in range(n):
            result.append(aug[i][n:])
            
        return result


class EuclideanMetric(RiemannianMetric):
    """
    Standard Euclidean metric (identity matrix).
    """
    
    def metric_tensor(self, point: List[float]) -> MetricTensor:
        """Return identity matrix."""
        matrix = [
            [1.0 if i == j else 0.0 for j in range(self.dimension)]
            for i in range(self.dimension)
        ]
        return MetricTensor(matrix=matrix, dimension=self.dimension)
        
    def gradient_to_natural(
        self,
        point: List[float],
        gradient: List[float]
    ) -> List[float]:
        """For Euclidean metric, natural = Euclidean."""
        return gradient.copy()


class FisherInformationMetric(RiemannianMetric):
    """
    Fisher Information Metric for parameter estimation.
    
    The Fisher metric at θ is:
    g_{ij}(θ) = E_{p(x|θ)}[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
    
    This metric makes gradient descent invariant to parameterization.
    """
    
    def __init__(
        self,
        dimension: int,
        log_likelihood_gradient: Callable[[List[float], List[float]], List[float]] = None,
        regularization: float = 0.001
    ):
        """
        Initialize Fisher metric.
        
        Args:
            dimension: Parameter dimension
            log_likelihood_gradient: Function (params, data) -> gradient of log likelihood
            regularization: Damping factor for numerical stability
        """
        super().__init__(dimension)
        self.log_likelihood_gradient = log_likelihood_gradient
        self.regularization = regularization
        
        # Cached gradient samples for Monte Carlo estimation
        self._gradient_samples: List[List[float]] = []
        
    def add_gradient_sample(self, gradient: List[float]):
        """
        Add a gradient sample for Fisher estimation.
        
        Args:
            gradient: Gradient of log-likelihood at a data point
        """
        self._gradient_samples.append(gradient.copy())
        
    def clear_samples(self):
        """Clear accumulated gradient samples."""
        self._gradient_samples = []
        
    def metric_tensor(self, point: List[float]) -> MetricTensor:
        """
        Compute Fisher Information Matrix via Monte Carlo.
        
        F_{ij} ≈ (1/N) Σ_k g_i^(k) g_j^(k)
        
        where g^(k) is the k-th gradient sample.
        """
        n = len(self._gradient_samples)
        
        if n == 0:
            # Return regularized identity if no samples
            matrix = [
                [self.regularization if i == j else 0.0 
                 for j in range(self.dimension)]
                for i in range(self.dimension)
            ]
            return MetricTensor(matrix=matrix, dimension=self.dimension)
            
        # Compute empirical Fisher
        matrix = []
        for i in range(self.dimension):
            row = []
            for j in range(self.dimension):
                val = sum(
                    sample[i] * sample[j] 
                    for sample in self._gradient_samples
                ) / n
                # Add regularization on diagonal
                if i == j:
                    val += self.regularization
                row.append(val)
            matrix.append(row)
            
        return MetricTensor(matrix=matrix, dimension=self.dimension)


class DiagonalFisherMetric(RiemannianMetric):
    """
    Diagonal approximation to Fisher Information Matrix.
    
    More efficient than full Fisher but still captures per-parameter
    curvature information.
    """
    
    def __init__(
        self,
        dimension: int,
        regularization: float = 0.001
    ):
        super().__init__(dimension)
        self.regularization = regularization
        self._gradient_samples: List[List[float]] = []
        
    def add_gradient_sample(self, gradient: List[float]):
        """Add gradient sample."""
        self._gradient_samples.append(gradient.copy())
        
    def clear_samples(self):
        """Clear samples."""
        self._gradient_samples = []
        
    def metric_tensor(self, point: List[float]) -> MetricTensor:
        """Compute diagonal Fisher."""
        n = len(self._gradient_samples)
        
        diagonal = [self.regularization] * self.dimension
        
        if n > 0:
            for i in range(self.dimension):
                diagonal[i] += sum(
                    sample[i] ** 2 for sample in self._gradient_samples
                ) / n
                
        matrix = [
            [diagonal[i] if i == j else 0.0 for j in range(self.dimension)]
            for i in range(self.dimension)
        ]
        
        return MetricTensor(matrix=matrix, dimension=self.dimension)
        
    def gradient_to_natural(
        self,
        point: List[float],
        gradient: List[float]
    ) -> List[float]:
        """Efficient natural gradient for diagonal metric."""
        g = self.metric_tensor(point)
        # For diagonal, just divide each component
        return [gradient[i] / g.matrix[i][i] for i in range(self.dimension)]


class LearnedMetric(RiemannianMetric):
    """
    A metric that can be learned from data.
    
    The metric is parameterized as G = L L^T where L is a lower
    triangular matrix (Cholesky factor).
    """
    
    def __init__(
        self,
        dimension: int,
        initial_scale: float = 1.0
    ):
        super().__init__(dimension)
        
        # Initialize Cholesky factor to scaled identity
        self._cholesky = [
            [initial_scale if i == j else 0.0 for j in range(dimension)]
            for i in range(dimension)
        ]
        
    def metric_tensor(self, point: List[float]) -> MetricTensor:
        """Compute G = L L^T."""
        matrix = []
        for i in range(self.dimension):
            row = []
            for j in range(self.dimension):
                # (L L^T)_{ij} = Σ_k L_{ik} L_{jk}
                val = sum(
                    self._cholesky[i][k] * self._cholesky[j][k]
                    for k in range(min(i, j) + 1)
                )
                row.append(val)
            matrix.append(row)
            
        return MetricTensor(matrix=matrix, dimension=self.dimension)
        
    def set_cholesky_factor(self, cholesky: List[List[float]]):
        """
        Set the Cholesky factor directly.
        
        Args:
            cholesky: Lower triangular matrix L such that G = L L^T
        """
        self._cholesky = cholesky
        
    def get_cholesky_factor(self) -> List[List[float]]:
        """Get current Cholesky factor."""
        return [row.copy() for row in self._cholesky]
