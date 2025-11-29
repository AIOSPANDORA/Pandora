"""
Benchmark Runner - Compare Geodesic Optimizer Against Baselines

This module runs benchmarks comparing natural gradient descent (geodesic optimizer)
against standard optimizers (Adam, SGD) on various test functions.
"""

import math
import time
from typing import Dict, List, Callable, Tuple, Any
from dataclasses import dataclass

from .geodesic_optimizer import GeodesicOptimizer, AdamBaseline, SGDBaseline
from .causal_attention import (
    CausalAttentionLayer,
    LightConeMask,
    CausalMask,
    compare_attention_outputs
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    optimizer_name: str
    test_function: str
    iterations: int
    final_loss: float
    time_seconds: float
    converged: bool
    
    
@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult]
    summary: Dict[str, Any]


# Test Functions

def rosenbrock(params: List[float]) -> float:
    """
    Rosenbrock function - a classic optimization test.
    Minimum at (1, 1, ..., 1) with value 0.
    """
    n = len(params)
    total = 0.0
    for i in range(n - 1):
        # Clamp intermediate values to prevent overflow
        term1 = params[i + 1] - params[i] ** 2
        term1 = max(-1e10, min(1e10, term1))
        term2 = 1 - params[i]
        term2 = max(-1e5, min(1e5, term2))
        total += 100 * term1 ** 2 + term2 ** 2
        total = max(-1e15, min(1e15, total))
    return total


def rosenbrock_gradient(params: List[float]) -> List[float]:
    """Gradient of Rosenbrock function."""
    n = len(params)
    grad = [0.0] * n
    
    for i in range(n - 1):
        # Clamp to prevent extreme values
        term = params[i + 1] - params[i] ** 2
        term = max(-1e5, min(1e5, term))
        
        grad[i] += -400 * params[i] * term - 2 * (1 - params[i])
        grad[i] = max(-1e10, min(1e10, grad[i]))
        
        grad[i + 1] += 200 * term
        grad[i + 1] = max(-1e10, min(1e10, grad[i + 1]))
        
    return grad


def rosenbrock_fisher(params: List[float]) -> List[float]:
    """Approximate Fisher diagonal for Rosenbrock (using gradient squared)."""
    grad = rosenbrock_gradient(params)
    return [g * g + 0.01 for g in grad]


def quadratic(params: List[float]) -> float:
    """
    Simple quadratic function: sum(i * x_i^2).
    Minimum at origin with value 0.
    """
    return sum((i + 1) * x ** 2 for i, x in enumerate(params))


def quadratic_gradient(params: List[float]) -> List[float]:
    """Gradient of quadratic function."""
    return [2 * (i + 1) * x for i, x in enumerate(params)]


def quadratic_fisher(params: List[float]) -> List[float]:
    """Fisher diagonal for quadratic (constant)."""
    return [2 * (i + 1) for i in range(len(params))]


def sphere(params: List[float]) -> float:
    """
    Sphere function: sum(x_i^2).
    Minimum at origin with value 0.
    """
    return sum(x ** 2 for x in params)


def sphere_gradient(params: List[float]) -> List[float]:
    """Gradient of sphere function."""
    return [2 * x for x in params]


def sphere_fisher(params: List[float]) -> List[float]:
    """Fisher diagonal for sphere (constant)."""
    return [1.0] * len(params)


# Benchmark Runners

def run_optimizer_benchmark(
    optimizer_name: str,
    test_name: str,
    loss_fn: Callable[[List[float]], float],
    gradient_fn: Callable[[List[float]], List[float]],
    fisher_fn: Callable[[List[float]], List[float]] = None,
    initial_params: List[float] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> BenchmarkResult:
    """
    Run a single optimizer benchmark.
    
    Args:
        optimizer_name: Name of optimizer ("geodesic", "adam", "sgd")
        test_name: Name of test function
        loss_fn: Loss function
        gradient_fn: Gradient function
        fisher_fn: Optional Fisher diagonal function
        initial_params: Starting parameters
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        BenchmarkResult
    """
    if initial_params is None:
        initial_params = [-2.0, 2.0]
        
    # Create optimizer
    if optimizer_name == "geodesic":
        optimizer = GeodesicOptimizer(learning_rate=0.01, damping=0.001)
    elif optimizer_name == "adam":
        optimizer = AdamBaseline(learning_rate=0.01)
    else:  # sgd
        optimizer = SGDBaseline(learning_rate=0.01, momentum=0.9)
        
    # Run optimization
    start_time = time.time()
    
    if optimizer_name == "geodesic" and fisher_fn is not None:
        final_state = optimizer.optimize(
            initial_params, loss_fn, gradient_fn, fisher_fn,
            max_iterations, tolerance
        )
    else:
        final_state = optimizer.optimize(
            initial_params, loss_fn, gradient_fn,
            max_iterations, tolerance
        )
        
    elapsed = time.time() - start_time
    
    converged = final_state.gradient_norm < tolerance
    
    return BenchmarkResult(
        optimizer_name=optimizer_name,
        test_function=test_name,
        iterations=final_state.iteration + 1,
        final_loss=final_state.loss,
        time_seconds=elapsed,
        converged=converged
    )


def run_all_benchmarks(
    dimensions: int = 2,
    max_iterations: int = 1000
) -> BenchmarkSuite:
    """
    Run all benchmarks comparing optimizers.
    
    Args:
        dimensions: Number of dimensions for test functions
        max_iterations: Maximum iterations per run
        
    Returns:
        BenchmarkSuite with all results
    """
    results = []
    
    # Test functions and their derivatives
    test_functions = [
        ("sphere", sphere, sphere_gradient, sphere_fisher),
        ("quadratic", quadratic, quadratic_gradient, quadratic_fisher),
        ("rosenbrock", rosenbrock, rosenbrock_gradient, rosenbrock_fisher),
    ]
    
    optimizers = ["geodesic", "adam", "sgd"]
    
    # Initial parameters - use small values to avoid overflow
    initial_params = [1.0] * dimensions
    
    for test_name, loss_fn, grad_fn, fisher_fn in test_functions:
        for opt_name in optimizers:
            result = run_optimizer_benchmark(
                opt_name, test_name,
                loss_fn, grad_fn, fisher_fn,
                initial_params.copy(),
                max_iterations
            )
            results.append(result)
            
    # Compute summary
    summary = compute_summary(results)
    
    return BenchmarkSuite(results=results, summary=summary)


def compute_summary(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute summary statistics from benchmark results."""
    # Group by optimizer
    by_optimizer: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.optimizer_name not in by_optimizer:
            by_optimizer[r.optimizer_name] = []
        by_optimizer[r.optimizer_name].append(r)
        
    summary = {}
    for opt_name, opt_results in by_optimizer.items():
        avg_iterations = sum(r.iterations for r in opt_results) / len(opt_results)
        avg_time = sum(r.time_seconds for r in opt_results) / len(opt_results)
        convergence_rate = sum(1 for r in opt_results if r.converged) / len(opt_results)
        
        summary[opt_name] = {
            "average_iterations": avg_iterations,
            "average_time_seconds": avg_time,
            "convergence_rate": convergence_rate,
            "num_tests": len(opt_results)
        }
        
    return summary


def run_attention_benchmark(
    sequence_length: int = 10,
    d_model: int = 8,
    propagation_speed: float = 1.0
) -> Dict[str, Any]:
    """
    Benchmark standard vs light-cone attention.
    
    Args:
        sequence_length: Length of test sequence
        d_model: Model dimension
        propagation_speed: Speed for light-cone mask
        
    Returns:
        Benchmark results dictionary
    """
    # Create test input (random-ish pattern)
    x = []
    for i in range(sequence_length):
        row = [math.sin(i * 0.5 + j * 0.3) for j in range(d_model)]
        x.append(row)
        
    # Create positions with some spatial spread
    positions = [(float(i), float(i % 3)) for i in range(sequence_length)]
    
    # Standard causal attention
    standard_layer = CausalAttentionLayer(d_model, use_light_cone=False)
    
    start = time.time()
    standard_output = standard_layer.forward(x)
    standard_time = time.time() - start
    
    # Light-cone attention
    lightcone_layer = CausalAttentionLayer(
        d_model, 
        use_light_cone=True,
        propagation_speed=propagation_speed
    )
    
    start = time.time()
    lightcone_output = lightcone_layer.forward(x, positions)
    lightcone_time = time.time() - start
    
    # Compare
    comparison = compare_attention_outputs(standard_output, lightcone_output)
    
    return {
        "sequence_length": sequence_length,
        "d_model": d_model,
        "standard_time_seconds": standard_time,
        "lightcone_time_seconds": lightcone_time,
        "attention_comparison": comparison
    }


def print_results(suite: BenchmarkSuite):
    """Print benchmark results in a formatted way."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    for result in suite.results:
        status = "✓" if result.converged else "✗"
        print(f"\n{status} {result.optimizer_name} on {result.test_function}:")
        print(f"   Iterations: {result.iterations}")
        print(f"   Final Loss: {result.final_loss:.6e}")
        print(f"   Time: {result.time_seconds:.4f}s")
        
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    
    for opt_name, stats in suite.summary.items():
        print(f"\n{opt_name}:")
        print(f"   Average Iterations: {stats['average_iterations']:.1f}")
        print(f"   Average Time: {stats['average_time_seconds']:.4f}s")
        print(f"   Convergence Rate: {stats['convergence_rate']:.1%}")


if __name__ == "__main__":
    # Run benchmarks
    print("Running optimization benchmarks...")
    suite = run_all_benchmarks(dimensions=2, max_iterations=500)
    print_results(suite)
    
    print("\n" + "=" * 60)
    print("ATTENTION BENCHMARK")
    print("=" * 60)
    
    attention_results = run_attention_benchmark(
        sequence_length=10,
        d_model=8,
        propagation_speed=1.0
    )
    
    print(f"\nSequence Length: {attention_results['sequence_length']}")
    print(f"Model Dimension: {attention_results['d_model']}")
    print(f"Standard Attention Time: {attention_results['standard_time_seconds']:.4f}s")
    print(f"Light-Cone Attention Time: {attention_results['lightcone_time_seconds']:.4f}s")
    print(f"Weight Difference: {attention_results['attention_comparison']['average_weight_difference']:.4f}")
