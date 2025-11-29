# Benchmarks

This directory contains benchmark implementations for validating claims made in the Pandora AIOS Scientific Framework.

## Purpose

Following the principle of scientific rigor, these benchmarks provide:

1. **Empirical validation** of theoretical claims
2. **Reproducible comparisons** against baseline methods
3. **Falsification tests** for core conjectures

## Contents

### `geodesic_optimizer.py`

Implements natural gradient descent with K-FAC (Kronecker-Factored Approximate Curvature) approximation.

**Key Classes:**
- `GeodesicOptimizer`: Natural gradient descent following geodesics on the parameter manifold
- `AdamBaseline`: Standard Adam optimizer for comparison
- `SGDBaseline`: Standard SGD with momentum for comparison

**Based on:**
- Amari, S. (1998). Natural Gradient Works Efficiently in Learning
- Martens, J., & Grosse, R. (2015). Optimizing Neural Networks with K-FAC

### `causal_attention.py`

Implements attention mechanisms with causal masking, including light-cone constraints.

**Key Classes:**
- `CausalMask`: Standard autoregressive attention mask
- `LightConeMask`: Light-cone masked attention (information propagation constraints)
- `ScaledDotProductAttention`: Core attention computation
- `CausalAttentionLayer`: Complete attention layer with masking

**Based on:**
- Vaswani, A., et al. (2017). Attention Is All You Need
- Relativistic causality constraints on information flow

### `benchmark_runner.py`

Main benchmark execution and comparison framework.

**Features:**
- Multiple test functions (Rosenbrock, Quadratic, Sphere)
- Automated comparison across optimizers
- Summary statistics and convergence tracking
- Attention mechanism benchmarking

## Running Benchmarks

### Quick Start

```bash
cd /path/to/Pandora
python -m benchmarks.benchmark_runner
```

### Running Specific Tests

```python
from benchmarks.benchmark_runner import run_all_benchmarks, print_results

# Run optimization benchmarks
suite = run_all_benchmarks(dimensions=5, max_iterations=1000)
print_results(suite)
```

### Running Attention Benchmarks

```python
from benchmarks.benchmark_runner import run_attention_benchmark

results = run_attention_benchmark(
    sequence_length=20,
    d_model=16,
    propagation_speed=2.0
)
print(results)
```

## Interpreting Results

### Optimization Benchmarks

Results compare three optimizers on standard test functions:

| Metric | Description |
|--------|-------------|
| Iterations | Steps to convergence (or max) |
| Final Loss | Loss value at termination |
| Time | Wall-clock time in seconds |
| Converged | Whether gradient norm < tolerance |

**Claim Validation:**
- If `geodesic` consistently requires fewer iterations than `adam`/`sgd`, the geodesic efficiency claim is supported
- If `geodesic` takes similar or more iterations, the claim requires qualification

### Attention Benchmarks

Results compare standard causal attention vs. light-cone attention:

| Metric | Description |
|--------|-------------|
| Time | Computation time for each method |
| Weight Difference | How much attention patterns differ |
| Positions with Difference | Count of differing attention positions |

## Test Functions

### Sphere Function
```
f(x) = Σ x_i²
```
Simple convex function. Minimum at origin.

### Quadratic Function
```
f(x) = Σ (i+1) * x_i²
```
Ill-conditioned quadratic (different curvature per dimension).

### Rosenbrock Function
```
f(x) = Σ [100(x_{i+1} - x_i²)² + (1-x_i)²]
```
Classic non-convex test function. Minimum at (1,1,...,1).

## Adding New Benchmarks

1. Define your test function and its gradient
2. Optionally define Fisher diagonal approximation
3. Add to `run_all_benchmarks()` or create custom benchmark
4. Document expected behavior and falsification criteria

## Dependencies

- Python standard library only (no numpy required)
- Uses `math`, `time`, `dataclasses`, `typing` modules

## References

1. Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*, 10(2), 251-276.
2. Martens, J., & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature. *ICML 2015*.
3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
4. Bonnabel, S. (2013). Stochastic Gradient Descent on Riemannian Manifolds. *IEEE TAC*.

---

*"In God we trust. All others must bring data."* — W. Edwards Deming
