"""
Unit tests for benchmark components and geometric learning modules.

Tests cover:
- Geodesic optimizer and baselines
- Causal attention mechanisms
- Manifold operations
- Metric computations
- Ethical constraints
"""

import unittest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.geodesic_optimizer import (
    GeodesicOptimizer,
    AdamBaseline,
    SGDBaseline,
    OptimizationState
)
from benchmarks.causal_attention import (
    CausalMask,
    LightConeMask,
    ScaledDotProductAttention,
    CausalAttentionLayer,
    softmax
)
from benchmarks.benchmark_runner import (
    rosenbrock,
    rosenbrock_gradient,
    quadratic,
    quadratic_gradient,
    sphere,
    sphere_gradient,
    run_optimizer_benchmark,
    run_all_benchmarks
)

from geometric_learning.manifold import (
    EuclideanManifold,
    SphereManifold,
    LearnedManifold,
    ManifoldPoint
)
from geometric_learning.metric import (
    EuclideanMetric,
    FisherInformationMetric,
    DiagonalFisherMetric,
    MetricTensor
)
from geometric_learning.geodesic import (
    GeodesicComputer,
    RiemannianGradientDescent,
    NaturalGradientOptimizer
)
from geometric_learning.causal_mask import (
    SpacetimePosition,
    LightConeAttentionMask,
    CausalMaskGenerator,
    in_past_light_cone,
    in_future_light_cone
)

from ethics.formal_constraints import (
    ReciprocityChecker,
    HarmAssessor,
    FairnessEvaluator,
    ValueAlignmentScorer,
    HumanOversightHook,
    EthicalConstraintEnforcer,
    EthicalContext,
    EthicalStatus
)


class TestGeodesicOptimizer(unittest.TestCase):
    """Tests for geodesic optimizer."""
    
    def setUp(self):
        self.optimizer = GeodesicOptimizer(learning_rate=0.1, damping=0.01)
        
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.iterations, 0)
        self.assertEqual(len(self.optimizer.history), 0)
        
    def test_compute_fisher_diagonal(self):
        """Test Fisher diagonal computation."""
        gradients = [[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]]
        fisher = self.optimizer.compute_fisher_diagonal(gradients)
        
        # Fisher[0] = (1^2 + 3^2 + 2^2) / 3 = 14/3
        self.assertAlmostEqual(fisher[0], 14/3, places=5)
        # Fisher[1] = (4 + 16 + 9) / 3 = 29/3
        self.assertAlmostEqual(fisher[1], 29/3, places=5)
        
    def test_compute_natural_gradient(self):
        """Test natural gradient computation."""
        gradient = [1.0, 1.0]
        fisher = [2.0, 4.0]
        
        natural_grad = self.optimizer.compute_natural_gradient(gradient, fisher)
        
        # natural_grad[i] = grad[i] / (fisher[i] + damping)
        self.assertAlmostEqual(natural_grad[0], 1.0 / 2.01, places=3)
        self.assertAlmostEqual(natural_grad[1], 1.0 / 4.01, places=3)
        
    def test_step(self):
        """Test single optimization step."""
        params = [2.0, 2.0]
        gradient = [4.0, 4.0]
        
        new_params = self.optimizer.step(params, gradient)
        
        # Should move in negative gradient direction
        self.assertLess(new_params[0], params[0])
        self.assertLess(new_params[1], params[1])
        self.assertEqual(self.optimizer.iterations, 1)
        
    def test_optimize_sphere(self):
        """Test optimization on sphere function."""
        initial = [2.0, 2.0]
        result = self.optimizer.optimize(
            initial, sphere, sphere_gradient,
            max_iterations=100, tolerance=1e-4
        )
        
        # Should converge near origin
        self.assertLess(result.loss, 0.1)


class TestBaselineOptimizers(unittest.TestCase):
    """Tests for Adam and SGD baselines."""
    
    def test_adam_step(self):
        """Test Adam optimizer step."""
        adam = AdamBaseline(learning_rate=0.1)
        params = [2.0, 2.0]
        gradient = [4.0, 4.0]
        
        new_params = adam.step(params, gradient)
        
        self.assertLess(new_params[0], params[0])
        self.assertEqual(adam.iterations, 1)
        
    def test_sgd_step(self):
        """Test SGD optimizer step."""
        sgd = SGDBaseline(learning_rate=0.1, momentum=0.9)
        params = [2.0, 2.0]
        gradient = [4.0, 4.0]
        
        new_params = sgd.step(params, gradient)
        
        self.assertLess(new_params[0], params[0])
        self.assertEqual(sgd.iterations, 1)
        
    def test_adam_optimize(self):
        """Test Adam on sphere function."""
        adam = AdamBaseline(learning_rate=0.1)
        result = adam.optimize([2.0, 2.0], sphere, sphere_gradient, max_iterations=50)
        
        self.assertLess(result.loss, 0.5)


class TestCausalAttention(unittest.TestCase):
    """Tests for causal attention mechanisms."""
    
    def test_causal_mask(self):
        """Test standard causal mask."""
        mask = CausalMask(4)
        m = mask.get_mask()
        
        # Lower triangular
        self.assertEqual(m[0], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(m[1], [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(m[3], [1.0, 1.0, 1.0, 1.0])
        
    def test_light_cone_mask(self):
        """Test light-cone mask."""
        positions = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        mask = LightConeMask(4, propagation_speed=1.0, positions=positions)
        m = mask.get_mask()
        
        # With no spatial separation, should be same as causal
        self.assertEqual(m[0], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(m[3], [1.0, 1.0, 1.0, 1.0])
        
    def test_light_cone_with_spatial(self):
        """Test light-cone mask with spatial separation."""
        # Position 0: t=0, x=0
        # Position 1: t=1, x=2 (outside light cone of 0 if c=1)
        positions = [(0.0, 0.0), (1.0, 2.0)]
        mask = LightConeMask(2, propagation_speed=1.0, positions=positions)
        m = mask.get_mask()
        
        # Position 0 can only attend to itself
        self.assertEqual(m[0][0], 1.0)
        # Position 1 cannot attend to 0 (outside light cone)
        self.assertEqual(m[1][0], 0.0)
        
    def test_attention_computation(self):
        """Test scaled dot-product attention."""
        attention = ScaledDotProductAttention(d_k=2)
        query = [[1.0, 0.0], [0.0, 1.0]]
        key = [[1.0, 0.0], [0.0, 1.0]]
        value = [[1.0, 2.0], [3.0, 4.0]]
        
        result = attention.forward(query, key, value)
        
        self.assertEqual(len(result.output), 2)
        self.assertEqual(len(result.attention_weights), 2)
        
    def test_softmax(self):
        """Test softmax function."""
        result = softmax([1.0, 2.0, 3.0])
        
        self.assertAlmostEqual(sum(result), 1.0, places=5)
        self.assertGreater(result[2], result[1])
        self.assertGreater(result[1], result[0])


class TestManifolds(unittest.TestCase):
    """Tests for manifold implementations."""
    
    def test_euclidean_manifold(self):
        """Test Euclidean manifold operations."""
        manifold = EuclideanManifold(3)
        
        p1 = manifold.project([1.0, 2.0, 3.0])
        self.assertEqual(p1.coordinates, [1.0, 2.0, 3.0])
        
        # Exponential map is addition
        tangent = [1.0, 0.0, 0.0]
        p2 = manifold.exponential_map(p1, tangent)
        self.assertEqual(p2.coordinates, [2.0, 2.0, 3.0])
        
        # Log map is subtraction
        log = manifold.logarithmic_map(p1, p2)
        self.assertEqual(log, [1.0, 0.0, 0.0])
        
    def test_sphere_manifold_projection(self):
        """Test sphere manifold projection."""
        manifold = SphereManifold(3)
        
        p = manifold.project([3.0, 4.0, 0.0])
        
        # Should be unit norm
        norm = math.sqrt(sum(x**2 for x in p.coordinates))
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    def test_sphere_exponential_map(self):
        """Test sphere exponential map."""
        manifold = SphereManifold(3)
        
        # Start at north pole
        base = ManifoldPoint(coordinates=[1.0, 0.0, 0.0])
        # Move along tangent
        tangent = [0.0, math.pi/2, 0.0]
        
        result = manifold.exponential_map(base, tangent)
        
        # Should be on sphere
        norm = math.sqrt(sum(x**2 for x in result.coordinates))
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    def test_learned_manifold(self):
        """Test learned manifold."""
        manifold = LearnedManifold(dimension=2, embedding_dimension=4)
        
        p = manifold.project([1.0, 2.0])
        self.assertEqual(len(p.coordinates), 2)


class TestMetrics(unittest.TestCase):
    """Tests for Riemannian metrics."""
    
    def test_euclidean_metric(self):
        """Test Euclidean metric."""
        metric = EuclideanMetric(2)
        g = metric.metric_tensor([1.0, 1.0])
        
        # Should be identity
        self.assertEqual(g.matrix[0][0], 1.0)
        self.assertEqual(g.matrix[1][1], 1.0)
        self.assertEqual(g.matrix[0][1], 0.0)
        
    def test_euclidean_inner_product(self):
        """Test Euclidean inner product."""
        metric = EuclideanMetric(2)
        
        v1 = [3.0, 4.0]
        v2 = [1.0, 0.0]
        
        ip = metric.inner_product([0.0, 0.0], v1, v2)
        self.assertAlmostEqual(ip, 3.0, places=5)
        
    def test_euclidean_norm(self):
        """Test Euclidean norm."""
        metric = EuclideanMetric(2)
        
        norm = metric.norm([0.0, 0.0], [3.0, 4.0])
        self.assertAlmostEqual(norm, 5.0, places=5)
        
    def test_fisher_metric(self):
        """Test Fisher Information Metric."""
        metric = FisherInformationMetric(2, regularization=0.01)
        
        # Add gradient samples
        metric.add_gradient_sample([1.0, 2.0])
        metric.add_gradient_sample([2.0, 1.0])
        
        g = metric.metric_tensor([0.0, 0.0])
        
        # Should have positive diagonal
        self.assertGreater(g.matrix[0][0], 0)
        self.assertGreater(g.matrix[1][1], 0)
        
    def test_diagonal_fisher(self):
        """Test diagonal Fisher metric."""
        metric = DiagonalFisherMetric(2, regularization=0.01)
        
        metric.add_gradient_sample([1.0, 2.0])
        
        natural = metric.gradient_to_natural([0.0, 0.0], [1.0, 1.0])
        
        # Natural gradient should be scaled by inverse Fisher
        self.assertEqual(len(natural), 2)


class TestGeodesics(unittest.TestCase):
    """Tests for geodesic computations."""
    
    def test_euclidean_geodesic(self):
        """Test geodesic on Euclidean manifold."""
        manifold = EuclideanManifold(2)
        computer = GeodesicComputer(manifold)
        
        start = ManifoldPoint(coordinates=[0.0, 0.0])
        end = ManifoldPoint(coordinates=[1.0, 1.0])
        
        path = computer.geodesic(start, end, num_steps=5)
        
        self.assertEqual(len(path), 6)  # num_steps + 1
        self.assertEqual(path[0].coordinates, [0.0, 0.0])
        self.assertAlmostEqual(path[-1].coordinates[0], 1.0, places=5)
        
    def test_geodesic_distance(self):
        """Test geodesic distance computation."""
        manifold = EuclideanManifold(2)
        computer = GeodesicComputer(manifold)
        
        p1 = ManifoldPoint(coordinates=[0.0, 0.0])
        p2 = ManifoldPoint(coordinates=[3.0, 4.0])
        
        dist = computer.geodesic_distance(p1, p2)
        self.assertAlmostEqual(dist, 5.0, places=5)
        
    def test_natural_gradient_optimizer(self):
        """Test natural gradient optimizer."""
        optimizer = NaturalGradientOptimizer(
            dimension=2,
            learning_rate=0.1,
            damping=0.01
        )
        
        params = [2.0, 2.0]
        gradient = [4.0, 4.0]
        
        new_params = optimizer.step(params, gradient)
        
        self.assertLess(new_params[0], params[0])


class TestCausalMasks(unittest.TestCase):
    """Tests for causal mask generation."""
    
    def test_in_past_light_cone(self):
        """Test past light cone check."""
        observer = SpacetimePosition(time=2.0, space=(0.0,))
        
        # Event in past, no spatial separation
        event1 = SpacetimePosition(time=1.0, space=(0.0,))
        self.assertTrue(in_past_light_cone(observer, event1))
        
        # Event in future
        event2 = SpacetimePosition(time=3.0, space=(0.0,))
        self.assertFalse(in_past_light_cone(observer, event2))
        
        # Event in past but too far
        event3 = SpacetimePosition(time=1.0, space=(2.0,))
        self.assertFalse(in_past_light_cone(observer, event3, speed_of_light=1.0))
        
    def test_in_future_light_cone(self):
        """Test future light cone check."""
        observer = SpacetimePosition(time=0.0, space=(0.0,))
        
        # Event in future, no spatial separation
        event1 = SpacetimePosition(time=1.0, space=(0.0,))
        self.assertTrue(in_future_light_cone(observer, event1))
        
        # Event in past
        event2 = SpacetimePosition(time=-1.0, space=(0.0,))
        self.assertFalse(in_future_light_cone(observer, event2))
        
    def test_standard_causal_mask(self):
        """Test standard causal mask generation."""
        mask = CausalMaskGenerator.standard_causal_mask(3)
        
        self.assertEqual(mask[0], [1.0, 0.0, 0.0])
        self.assertEqual(mask[1], [1.0, 1.0, 0.0])
        self.assertEqual(mask[2], [1.0, 1.0, 1.0])
        
    def test_sliding_window_mask(self):
        """Test sliding window mask."""
        mask = CausalMaskGenerator.sliding_window_mask(5, window_size=2)
        
        # Position 0: can only see 0
        self.assertEqual(mask[0], [1.0, 0.0, 0.0, 0.0, 0.0])
        # Position 2: can see 1, 2
        self.assertEqual(mask[2], [0.0, 1.0, 1.0, 0.0, 0.0])


class TestEthicalConstraints(unittest.TestCase):
    """Tests for ethical constraint enforcement."""
    
    def test_reciprocity_checker(self):
        """Test reciprocity checking."""
        checker = ReciprocityChecker()
        
        # Record some history
        checker.record_action("alice", "help", "bob", 0.8)
        
        # Check reciprocity
        score, _ = checker.check_reciprocity("bob", "alice", 0.5)
        
        # Should be positive (bob is reciprocating)
        self.assertGreater(score, 0.3)
        
    def test_harm_assessor(self):
        """Test harm assessment."""
        assessor = HarmAssessor()
        
        # Safe action
        score1, concerns1 = assessor.assess_harm(
            "backup data and verify integrity",
            {},
            reversible=True
        )
        self.assertLess(score1, 0.3)
        
        # Harmful action
        score2, concerns2 = assessor.assess_harm(
            "delete all files and terminate processes",
            {"data_loss": True},
            reversible=False
        )
        self.assertGreater(score2, 0.5)
        self.assertGreater(len(concerns2), 0)
        
    def test_fairness_evaluator(self):
        """Test fairness evaluation."""
        evaluator = FairnessEvaluator()
        
        # Fair outcomes
        score1, _ = evaluator.evaluate_fairness({
            "alice": 0.5,
            "bob": 0.6,
            "charlie": 0.55
        })
        self.assertGreater(score1, 0.7)
        
        # Unfair outcomes
        score2, concerns = evaluator.evaluate_fairness({
            "alice": 0.9,
            "bob": -0.5
        })
        self.assertLess(score2, 0.7)
        self.assertGreater(len(concerns), 0)
        
    def test_value_alignment_scorer(self):
        """Test value alignment scoring."""
        scorer = ValueAlignmentScorer()
        
        # Aligned action
        score1, _ = scorer.score_alignment(
            "protect user privacy and provide transparent honest reporting"
        )
        self.assertGreater(score1, 0.2)
        
        # Neutral action
        score2, _ = scorer.score_alignment(
            "process data batch"
        )
        self.assertLessEqual(score2, score1)
        
    def test_human_oversight_hook(self):
        """Test oversight hook."""
        hook = HumanOversightHook(
            oversight_threshold=0.7,
            auto_approve_threshold=0.9
        )
        
        # High score - no oversight needed
        needs1, _ = hook.check_requires_oversight(0.95)
        self.assertFalse(needs1)
        
        # Low score - oversight needed
        needs2, _ = hook.check_requires_oversight(0.5)
        self.assertTrue(needs2)
        
        # High risk indicators
        needs3, _ = hook.check_requires_oversight(0.8, ["irreversible"])
        self.assertTrue(needs3)
        
    def test_ethical_constraint_enforcer(self):
        """Test full ethical enforcement."""
        enforcer = EthicalConstraintEnforcer()
        
        context = EthicalContext(
            actor="system",
            affected_parties=["user"],
            action_type="data_operation"
        )
        
        # Good action
        assessment1 = enforcer.assess_action(
            "backup user data with transparent reporting",
            context,
            expected_outcomes={"user": 0.8},
            affected_resources={},
            reversible=True
        )
        # Should be approved or at least warning (not rejected)
        self.assertIn(assessment1.status, [EthicalStatus.APPROVED, EthicalStatus.WARNING])
        
        # Bad action
        assessment2 = enforcer.assess_action(
            "delete all user files and bypass security",
            context,
            expected_outcomes={"user": -0.9},
            affected_resources={"data_loss": True, "privacy_violation": True},
            reversible=False
        )
        self.assertIn(assessment2.status, 
                      [EthicalStatus.REJECTED, EthicalStatus.REQUIRES_REVIEW, EthicalStatus.WARNING])


class TestBenchmarkRunner(unittest.TestCase):
    """Tests for benchmark runner."""
    
    def test_rosenbrock_function(self):
        """Test Rosenbrock function."""
        # Minimum at (1, 1)
        min_val = rosenbrock([1.0, 1.0])
        self.assertAlmostEqual(min_val, 0.0, places=5)
        
        # Non-minimum
        other_val = rosenbrock([0.0, 0.0])
        self.assertGreater(other_val, 0)
        
    def test_rosenbrock_gradient(self):
        """Test Rosenbrock gradient."""
        grad = rosenbrock_gradient([1.0, 1.0])
        
        # At minimum, gradient should be zero
        self.assertAlmostEqual(grad[0], 0.0, places=5)
        self.assertAlmostEqual(grad[1], 0.0, places=5)
        
    def test_quadratic_function(self):
        """Test quadratic function."""
        val = quadratic([1.0, 1.0, 1.0])
        # sum((i+1) * 1^2) = 1 + 2 + 3 = 6
        self.assertAlmostEqual(val, 6.0, places=5)
        
    def test_sphere_function(self):
        """Test sphere function."""
        val = sphere([3.0, 4.0])
        self.assertAlmostEqual(val, 25.0, places=5)
        
    def test_run_optimizer_benchmark(self):
        """Test running single benchmark."""
        result = run_optimizer_benchmark(
            "adam", "sphere",
            sphere, sphere_gradient,
            initial_params=[2.0, 2.0],
            max_iterations=50
        )
        
        self.assertEqual(result.optimizer_name, "adam")
        self.assertEqual(result.test_function, "sphere")
        self.assertGreater(result.iterations, 0)
        
    def test_run_all_benchmarks(self):
        """Test running all benchmarks."""
        # Use smaller initial values to avoid overflow
        from benchmarks.benchmark_runner import run_all_benchmarks as run_benchmarks
        suite = run_benchmarks(dimensions=2, max_iterations=20)
        
        # Should have results for 3 optimizers x 3 functions = 9
        self.assertEqual(len(suite.results), 9)
        
        # Should have summary for each optimizer
        self.assertIn("geodesic", suite.summary)
        self.assertIn("adam", suite.summary)
        self.assertIn("sgd", suite.summary)


if __name__ == "__main__":
    unittest.main()
