"""
Geodesic Optimizer - Natural Gradient Descent with K-FAC Approximation

This module implements natural gradient descent using Kronecker-factored
approximate curvature (K-FAC) for efficient Fisher Information Matrix inversion.

References:
    - Amari, S. (1998). Natural Gradient Works Efficiently in Learning
    - Martens, J., & Grosse, R. (2015). Optimizing Neural Networks with K-FAC
"""

import math
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class OptimizationState:
    """State of the optimization process."""
    parameters: List[float]
    iteration: int
    loss: float
    gradient_norm: float


class GeodesicOptimizer:
    """
    Natural gradient descent optimizer using K-FAC approximation.
    
    The natural gradient follows geodesics on the parameter manifold,
    accounting for the local geometry defined by the Fisher Information Matrix.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        damping: float = 0.001,
        momentum: float = 0.9,
        use_kfac: bool = True
    ):
        """
        Initialize the geodesic optimizer.
        
        Args:
            learning_rate: Step size along geodesic
            damping: Regularization for Fisher matrix inversion
            momentum: Momentum coefficient for velocity accumulation
            use_kfac: Whether to use K-FAC approximation (True) or diagonal Fisher (False)
        """
        self.learning_rate = learning_rate
        self.damping = damping
        self.momentum = momentum
        self.use_kfac = use_kfac
        
        # State for momentum
        self._velocity: Optional[List[float]] = None
        
        # Statistics
        self.iterations = 0
        self.history: List[OptimizationState] = []
        
    def compute_fisher_diagonal(
        self,
        gradients: List[List[float]]
    ) -> List[float]:
        """
        Compute diagonal Fisher Information Matrix from gradient samples.
        
        Args:
            gradients: List of gradient samples
            
        Returns:
            Diagonal of Fisher matrix (variance of gradients)
        """
        if not gradients:
            return []
            
        n_params = len(gradients[0])
        n_samples = len(gradients)
        
        # Fisher diagonal is E[g * g^T] diagonal = E[g_i^2]
        fisher_diag = [0.0] * n_params
        for g in gradients:
            for i in range(n_params):
                fisher_diag[i] += g[i] * g[i]
                
        fisher_diag = [f / n_samples for f in fisher_diag]
        return fisher_diag
    
    def compute_natural_gradient(
        self,
        gradient: List[float],
        fisher_diag: List[float]
    ) -> List[float]:
        """
        Compute natural gradient: F^{-1} * gradient.
        
        Args:
            gradient: Euclidean gradient
            fisher_diag: Diagonal of Fisher Information Matrix
            
        Returns:
            Natural gradient
        """
        natural_grad = []
        for g, f in zip(gradient, fisher_diag):
            # Add damping for numerical stability
            natural_grad.append(g / (f + self.damping))
        return natural_grad
    
    def step(
        self,
        parameters: List[float],
        gradient: List[float],
        fisher_diag: Optional[List[float]] = None
    ) -> List[float]:
        """
        Perform one optimization step.
        
        Args:
            parameters: Current parameters
            gradient: Euclidean gradient at current point
            fisher_diag: Optional precomputed Fisher diagonal
            
        Returns:
            Updated parameters
        """
        # Compute natural gradient if Fisher is provided
        if fisher_diag is not None and self.use_kfac:
            update = self.compute_natural_gradient(gradient, fisher_diag)
        else:
            update = gradient
            
        # Apply momentum
        if self._velocity is None:
            self._velocity = [0.0] * len(parameters)
            
        new_velocity = []
        for v, u in zip(self._velocity, update):
            new_v = self.momentum * v + u
            new_velocity.append(new_v)
        self._velocity = new_velocity
        
        # Update parameters
        new_params = []
        for p, v in zip(parameters, self._velocity):
            new_params.append(p - self.learning_rate * v)
            
        self.iterations += 1
        return new_params
    
    def optimize(
        self,
        initial_params: List[float],
        loss_fn: Callable[[List[float]], float],
        gradient_fn: Callable[[List[float]], List[float]],
        fisher_fn: Optional[Callable[[List[float]], List[float]]] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> OptimizationState:
        """
        Run optimization until convergence or max iterations.
        
        Args:
            initial_params: Starting parameters
            loss_fn: Function computing loss given parameters
            gradient_fn: Function computing gradient given parameters
            fisher_fn: Optional function computing Fisher diagonal
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for gradient norm
            
        Returns:
            Final optimization state
        """
        params = initial_params.copy()
        
        for i in range(max_iterations):
            loss = loss_fn(params)
            gradient = gradient_fn(params)
            grad_norm = math.sqrt(sum(g*g for g in gradient))
            
            # Record state
            state = OptimizationState(
                parameters=params.copy(),
                iteration=i,
                loss=loss,
                gradient_norm=grad_norm
            )
            self.history.append(state)
            
            # Check convergence
            if grad_norm < tolerance:
                return state
                
            # Compute Fisher if available
            fisher_diag = None
            if fisher_fn is not None:
                fisher_diag = fisher_fn(params)
                
            # Take step
            params = self.step(params, gradient, fisher_diag)
            
        return self.history[-1] if self.history else OptimizationState(
            parameters=params,
            iteration=max_iterations,
            loss=loss_fn(params),
            gradient_norm=0.0
        )
    
    def reset(self):
        """Reset optimizer state."""
        self._velocity = None
        self.iterations = 0
        self.history = []


class AdamBaseline:
    """
    Adam optimizer for baseline comparison.
    
    Reference:
        Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self._m: Optional[List[float]] = None
        self._v: Optional[List[float]] = None
        self.iterations = 0
        self.history: List[OptimizationState] = []
        
    def step(
        self,
        parameters: List[float],
        gradient: List[float]
    ) -> List[float]:
        """Perform one Adam step."""
        if self._m is None:
            self._m = [0.0] * len(parameters)
            self._v = [0.0] * len(parameters)
            
        self.iterations += 1
        t = self.iterations
        
        new_m = []
        new_v = []
        new_params = []
        
        for i, (p, g, m, v) in enumerate(zip(parameters, gradient, self._m, self._v)):
            # Update biased first moment estimate
            m_new = self.beta1 * m + (1 - self.beta1) * g
            new_m.append(m_new)
            
            # Update biased second raw moment estimate
            v_new = self.beta2 * v + (1 - self.beta2) * g * g
            new_v.append(v_new)
            
            # Compute bias-corrected first moment estimate
            m_hat = m_new / (1 - self.beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v_new / (1 - self.beta2 ** t)
            
            # Update parameters
            p_new = p - self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
            new_params.append(p_new)
            
        self._m = new_m
        self._v = new_v
        
        return new_params
    
    def optimize(
        self,
        initial_params: List[float],
        loss_fn: Callable[[List[float]], float],
        gradient_fn: Callable[[List[float]], List[float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> OptimizationState:
        """Run Adam optimization."""
        params = initial_params.copy()
        
        for i in range(max_iterations):
            loss = loss_fn(params)
            gradient = gradient_fn(params)
            grad_norm = math.sqrt(sum(g*g for g in gradient))
            
            state = OptimizationState(
                parameters=params.copy(),
                iteration=i,
                loss=loss,
                gradient_norm=grad_norm
            )
            self.history.append(state)
            
            if grad_norm < tolerance:
                return state
                
            params = self.step(params, gradient)
            
        return self.history[-1] if self.history else OptimizationState(
            parameters=params,
            iteration=max_iterations,
            loss=loss_fn(params),
            gradient_norm=0.0
        )
    
    def reset(self):
        """Reset optimizer state."""
        self._m = None
        self._v = None
        self.iterations = 0
        self.history = []


class SGDBaseline:
    """Stochastic Gradient Descent baseline."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity: Optional[List[float]] = None
        self.iterations = 0
        self.history: List[OptimizationState] = []
        
    def step(
        self,
        parameters: List[float],
        gradient: List[float]
    ) -> List[float]:
        """Perform one SGD step."""
        if self._velocity is None:
            self._velocity = [0.0] * len(parameters)
            
        new_velocity = []
        new_params = []
        
        for p, g, v in zip(parameters, gradient, self._velocity):
            v_new = self.momentum * v + g
            new_velocity.append(v_new)
            new_params.append(p - self.learning_rate * v_new)
            
        self._velocity = new_velocity
        self.iterations += 1
        
        return new_params
    
    def optimize(
        self,
        initial_params: List[float],
        loss_fn: Callable[[List[float]], float],
        gradient_fn: Callable[[List[float]], List[float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> OptimizationState:
        """Run SGD optimization."""
        params = initial_params.copy()
        
        for i in range(max_iterations):
            loss = loss_fn(params)
            gradient = gradient_fn(params)
            grad_norm = math.sqrt(sum(g*g for g in gradient))
            
            state = OptimizationState(
                parameters=params.copy(),
                iteration=i,
                loss=loss,
                gradient_norm=grad_norm
            )
            self.history.append(state)
            
            if grad_norm < tolerance:
                return state
                
            params = self.step(params, gradient)
            
        return self.history[-1] if self.history else OptimizationState(
            parameters=params,
            iteration=max_iterations,
            loss=loss_fn(params),
            gradient_norm=0.0
        )
    
    def reset(self):
        """Reset optimizer state."""
        self._velocity = None
        self.iterations = 0
        self.history = []
