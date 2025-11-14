"""
ML Quantum Addon

Provides machine learning capabilities with Hamiltonian awareness for quantum profiles.
Enables Hamiltonian parameter learning, expectation value tracking, and ML-guided optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from .hamiltonian import Hamiltonian


class MLQuantumAddon:
    """
    Machine Learning addon for quantum profiles with Hamiltonian awareness.
    
    Provides:
    - Hamiltonian logging and tracking
    - Expectation value computation and history
    - Hamiltonian parameter learning and optimization
    - ML-guided Hamiltonian construction
    - Integration with quantum profiles for adaptive learning
    
    Attributes:
        profile: Reference to quantum profile (optional)
        hamiltonian_history: History of Hamiltonian configurations
        expectation_history: History of expectation values
        learning_rate: Learning rate for parameter optimization
        parameter_history: History of learned parameters
    """
    
    def __init__(self, profile=None, learning_rate: float = 0.01):
        """
        Initialize the ML Quantum Addon.
        
        Args:
            profile: Quantum profile to attach to (optional)
            learning_rate: Learning rate for parameter optimization
        """
        self.profile = profile
        self.learning_rate = learning_rate
        
        # History tracking
        self.hamiltonian_history: List[Dict[str, Any]] = []
        self.expectation_history: List[Dict[str, Any]] = []
        self.parameter_history: List[Dict[str, Any]] = []
        
        # Current parameters being learned
        self.learned_parameters: Dict[str, float] = {}
        
        # Statistics
        self.total_measurements = 0
        self.training_iterations = 0
    
    def attach_profile(self, profile) -> None:
        """
        Attach to a quantum profile.
        
        Args:
            profile: Quantum profile instance to attach to
        """
        self.profile = profile
    
    def log_hamiltonian(self, hamiltonian: Hamiltonian, metadata: Optional[Dict] = None) -> None:
        """
        Log a Hamiltonian configuration for tracking and analysis.
        
        Args:
            hamiltonian: Hamiltonian to log
            metadata: Optional metadata dict (e.g., {'epoch': 10, 'loss': 0.5})
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'n_qubits': hamiltonian.n_qubits,
            'n_terms': hamiltonian.get_num_terms(),
            'terms': hamiltonian.get_terms(),
            'metadata': metadata or {}
        }
        self.hamiltonian_history.append(entry)
    
    def log_expectation_value(
        self,
        hamiltonian: Hamiltonian,
        state: np.ndarray,
        label: str = "measurement",
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Compute and log an expectation value.
        
        Args:
            hamiltonian: Hamiltonian to compute expectation with
            state: Quantum state
            label: Label for this measurement
            metadata: Optional metadata
        
        Returns:
            float: Computed expectation value
        """
        expectation = hamiltonian.compute_expectation(state)
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'expectation_value': float(expectation),
            'state_norm': float(np.linalg.norm(state)),
            'metadata': metadata or {}
        }
        self.expectation_history.append(entry)
        self.total_measurements += 1
        
        return expectation
    
    def compute_gradient(
        self,
        hamiltonian: Hamiltonian,
        state: np.ndarray,
        param_index: int,
        epsilon: float = 1e-5
    ) -> float:
        """
        Compute gradient of expectation value with respect to a Hamiltonian parameter.
        
        Uses finite difference: ∂E/∂θ ≈ (E(θ+ε) - E(θ-ε)) / (2ε)
        
        Args:
            hamiltonian: Hamiltonian (modified in place temporarily)
            state: Quantum state
            param_index: Index of parameter (term) to vary
            epsilon: Finite difference step size
        
        Returns:
            float: Gradient estimate
        """
        terms = hamiltonian.get_terms()
        if param_index >= len(terms):
            raise ValueError(f"Invalid parameter index: {param_index}")
        
        original_coeff, pauli_string = terms[param_index]
        
        # E(θ + ε)
        hamiltonian.terms[param_index] = (original_coeff + epsilon, pauli_string)
        hamiltonian._matrix_dirty = True
        energy_plus = hamiltonian.compute_energy(state)
        
        # E(θ - ε)
        hamiltonian.terms[param_index] = (original_coeff - epsilon, pauli_string)
        hamiltonian._matrix_dirty = True
        energy_minus = hamiltonian.compute_energy(state)
        
        # Restore original
        hamiltonian.terms[param_index] = (original_coeff, pauli_string)
        hamiltonian._matrix_dirty = True
        
        # Compute gradient
        gradient = (energy_plus - energy_minus) / (2 * epsilon)
        return gradient
    
    def optimize_hamiltonian_parameters(
        self,
        hamiltonian: Hamiltonian,
        state: np.ndarray,
        target_energy: float,
        num_iterations: int = 100,
        convergence_threshold: float = 1e-4
    ) -> Tuple[float, int]:
        """
        Optimize Hamiltonian parameters to approach target energy.
        
        Uses gradient descent on term coefficients to minimize
        |E(state) - target_energy|^2.
        
        Args:
            hamiltonian: Hamiltonian to optimize
            state: Reference quantum state
            target_energy: Target energy value
            num_iterations: Maximum number of optimization iterations
            convergence_threshold: Stop when |E - target| < threshold
        
        Returns:
            Tuple[float, int]: (final_error, iterations_performed)
        """
        for iteration in range(num_iterations):
            # Compute current energy and error
            current_energy = hamiltonian.compute_energy(state)
            error = abs(current_energy - target_energy)
            
            # Check convergence
            if error < convergence_threshold:
                self.training_iterations += iteration + 1
                return error, iteration + 1
            
            # Gradient descent on each parameter
            for i in range(len(hamiltonian.terms)):
                gradient = self.compute_gradient(hamiltonian, state, i)
                
                # Update parameter
                coeff, pauli_string = hamiltonian.terms[i]
                # Gradient of error^2: 2*(E - target) * ∂E/∂θ
                gradient_error = 2 * (current_energy - target_energy) * gradient
                new_coeff = coeff - self.learning_rate * gradient_error
                
                hamiltonian.terms[i] = (new_coeff, pauli_string)
                hamiltonian._matrix_dirty = True
            
            # Log progress every 10 iterations
            if iteration % 10 == 0:
                self.log_hamiltonian(
                    hamiltonian,
                    metadata={'iteration': iteration, 'error': error}
                )
        
        self.training_iterations += num_iterations
        final_energy = hamiltonian.compute_energy(state)
        final_error = abs(final_energy - target_energy)
        
        return final_error, num_iterations
    
    def learn_from_measurements(
        self,
        measurements: List[Tuple[np.ndarray, float]],
        initial_hamiltonian: Optional[Hamiltonian] = None
    ) -> Hamiltonian:
        """
        Learn a Hamiltonian from measurement data.
        
        Given pairs of (state, measured_energy), constructs or refines
        a Hamiltonian that best fits the data.
        
        Args:
            measurements: List of (state, energy) measurement pairs
            initial_hamiltonian: Starting Hamiltonian (if None, creates new one)
        
        Returns:
            Hamiltonian: Learned/refined Hamiltonian
        """
        if not measurements:
            raise ValueError("Need at least one measurement")
        
        # Get number of qubits from first state
        n_qubits = int(np.log2(measurements[0][0].shape[0]))
        
        # Initialize or use provided Hamiltonian
        if initial_hamiltonian is None:
            hamiltonian = Hamiltonian(n_qubits)
            # Start with basic single-qubit Z terms
            for i in range(n_qubits):
                pauli_string = 'I' * i + 'Z' + 'I' * (n_qubits - i - 1)
                hamiltonian.add_term(1.0, pauli_string)
        else:
            hamiltonian = initial_hamiltonian
        
        # Optimize to fit each measurement
        for state, target_energy in measurements:
            error, iterations = self.optimize_hamiltonian_parameters(
                hamiltonian, state, target_energy, num_iterations=50
            )
            self.log_expectation_value(
                hamiltonian, state, "learned",
                metadata={'target': target_energy, 'error': error}
            )
        
        return hamiltonian
    
    def suggest_hamiltonian_term(
        self,
        n_qubits: int,
        existing_terms: List[str]
    ) -> Tuple[float, str]:
        """
        Suggest a new Hamiltonian term based on ML heuristics.
        
        Args:
            n_qubits: Number of qubits
            existing_terms: List of existing Pauli strings
        
        Returns:
            Tuple[float, str]: (suggested_coefficient, pauli_string)
        """
        # Simple heuristic: suggest complementary terms
        # Count operator types in existing terms
        operator_counts = {'X': 0, 'Y': 0, 'Z': 0, 'I': 0}
        for term in existing_terms:
            for op in term:
                operator_counts[op] += 1
        
        # Find least-used operator
        non_identity = {k: v for k, v in operator_counts.items() if k != 'I'}
        least_used = min(non_identity, key=non_identity.get) if non_identity else 'Z'
        
        # Suggest term with least-used operator
        # Place on random qubit that doesn't have many operators
        import random
        qubit_idx = random.randint(0, n_qubits - 1)
        pauli_string = 'I' * qubit_idx + least_used + 'I' * (n_qubits - qubit_idx - 1)
        
        # Suggest moderate coefficient
        coefficient = 0.5
        
        return coefficient, pauli_string
    
    def get_expectation_statistics(self) -> Dict[str, float]:
        """
        Get statistics about logged expectation values.
        
        Returns:
            Dict[str, float]: Statistics including mean, std, min, max
        """
        if not self.expectation_history:
            return {}
        
        values = [entry['expectation_value'] for entry in self.expectation_history]
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of ML addon activity.
        
        Returns:
            Dict[str, Any]: Summary statistics and information
        """
        return {
            'profile_attached': self.profile is not None,
            'profile_name': self.profile.name if self.profile else None,
            'learning_rate': self.learning_rate,
            'total_measurements': self.total_measurements,
            'training_iterations': self.training_iterations,
            'hamiltonian_logs': len(self.hamiltonian_history),
            'expectation_logs': len(self.expectation_history),
            'expectation_stats': self.get_expectation_statistics()
        }
    
    def __str__(self) -> str:
        """String representation."""
        attached = f" attached to {self.profile.name}" if self.profile else " (unattached)"
        return f"MLQuantumAddon{attached}: {self.total_measurements} measurements, {self.training_iterations} training iterations"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<MLQuantumAddon(learning_rate={self.learning_rate}, measurements={self.total_measurements})>"
