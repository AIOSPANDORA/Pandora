"""
ML Quantum Addon

Provides machine learning capabilities with Hamiltonian awareness for quantum profiles.
Enables Hamiltonian parameter learning, expectation value tracking, and ML-guided optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from .hamiltonian import Hamiltonian

# MLQuantumAddon - Machine Learning Quantum Process Logging and Training
# This addon provides on-the-fly machine learning capabilities for quantum processors,
# logging all quantum operations and supporting incremental training of ML models.
#
# Features:
# - Logs quantum gates, measurements, expansion, and diagnostics
# - Supports on-the-fly incremental training (default: enabled)
# - Compatible with scikit-learn and custom pluggable models
# - Self-instrumenting for adaptive learning

import time
import logging
from typing import Any, Dict, List, Optional, Callable
import numpy as np


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
        
        Uses finite difference: dE/dtheta approx (E(theta+eps) - E(theta-eps)) / (2*eps)
        where d denotes partial derivative and approx means approximately equal.
        
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
    Machine Learning addon for quantum processors that logs operations
    and supports on-the-fly incremental training.
    
    Attributes:
        model: Optional ML model supporting partial_fit for incremental learning
        train_on_fly: Enable/disable on-the-fly training (default: True)
        event_log: List of all logged quantum events
        feature_buffer: Buffer for features awaiting training
        label_buffer: Buffer for labels awaiting training
    """
    
    def __init__(self, model: Optional[Any] = None, train_on_fly: bool = True,
                 buffer_size: int = 10, log_level: int = logging.INFO):
        """
        Initialize the MLQuantumAddon.
        
        Args:
            model: Optional ML model with partial_fit method for incremental learning.
                   Compatible with scikit-learn models (SGDClassifier, etc.)
            train_on_fly: Enable on-the-fly incremental training (default: True)
            buffer_size: Number of events to buffer before training (default: 10)
            log_level: Logging level (default: INFO)
        """
        self.model = model
        self.train_on_fly = train_on_fly
        self.buffer_size = buffer_size
        self.event_log: List[Dict[str, Any]] = []
        self.feature_buffer: List[np.ndarray] = []
        self.label_buffer: List[int] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        self.logger.info(f"MLQuantumAddon initialized (train_on_fly={train_on_fly})")
    
    def _log_event(self, event_type: str, operation: str, 
                   register: Any, context: Dict[str, Any]) -> None:
        """
        Internal method to log an event.
        
        Args:
            event_type: Type of event (e.g., 'before_gate', 'after_gate')
            operation: The quantum operation being performed
            register: Register or qubit being operated on
            context: Additional context information
        """
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'operation': operation,
            'register': register,
            'context': context.copy()
        }
        self.event_log.append(event)
        self.logger.debug(f"Logged event: {event_type} - {operation} on {register}")
    
    def _extract_features(self, event: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features from an event for ML training.
        
        Args:
            event: Event dictionary
            
        Returns:
            numpy array of features
        """
        # Basic feature extraction - can be extended based on needs
        features = []
        
        # Event type encoding (simple hash-based encoding)
        features.append(hash(event['event_type']) % 100)
        
        # Operation encoding
        features.append(hash(event['operation']) % 100)
        
        # Register encoding (handle various types)
        if isinstance(event['register'], (int, float)):
            features.append(float(event['register']))
        else:
            features.append(hash(str(event['register'])) % 100)
        
        # Context features
        context = event.get('context', {})
        features.append(context.get('qubits', 0))
        
        # Timestamp features (time of day, hour)
        timestamp = event['timestamp']
        features.append(timestamp % 86400)  # Time of day in seconds
        
        return np.array(features, dtype=np.float32)
    
    def _generate_label(self, event: Dict[str, Any]) -> int:
        """
        Generate a label for supervised learning (can be customized).
        
        Args:
            event: Event dictionary
            
        Returns:
            Integer label
        """
        # Simple label generation based on event type
        # In practice, this would be more sophisticated
        label_map = {
            'before_gate': 0,
            'after_gate': 1,
            'before_measurement': 2,
            'after_measurement': 3,
            'expansion': 4,
            'diagnostic': 5
        }
        return label_map.get(event['event_type'], 0)
    
    def _train_model(self) -> None:
        """
        Perform incremental training on buffered data if model is available.
        """
        if self.model is None:
            return
        
        if not self.feature_buffer or not self.label_buffer:
            return
        
        if not hasattr(self.model, 'partial_fit'):
            self.logger.warning("Model does not support partial_fit, skipping training")
            return
        
        try:
            X = np.array(self.feature_buffer)
            y = np.array(self.label_buffer)
            
            # Train with classes parameter if this is the first training
            if not hasattr(self.model, 'classes_'):
                classes = np.unique(y)
                self.model.partial_fit(X, y, classes=classes)
            else:
                self.model.partial_fit(X, y)
            
            self.logger.info(f"Trained model on {len(X)} samples")
            
            # Clear buffers after training
            self.feature_buffer.clear()
            self.label_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
    
    def _maybe_train(self, event: Dict[str, Any]) -> None:
        """
        Buffer event and train if buffer is full and train_on_fly is enabled.
        
        Args:
            event: Event to process for training
        """
        if not self.train_on_fly or self.model is None:
            return
        
        # Extract features and labels
        features = self._extract_features(event)
        label = self._generate_label(event)
        
        # Add to buffer
        self.feature_buffer.append(features)
        self.label_buffer.append(label)
        
        # Train if buffer is full
        if len(self.feature_buffer) >= self.buffer_size:
            self._train_model()
    
    # Quantum operation hooks
    
    def before_gate(self, gate: str, register: Any, context: Dict[str, Any]) -> None:
        """
        Hook called before a quantum gate is applied.
        
        Args:
            gate: Gate name (e.g., 'H', 'CNOT', 'X')
            register: Register or qubit being operated on
            context: Additional context (e.g., number of qubits, state)
        """
        self._log_event('before_gate', gate, register, context)
        if self.event_log:
            self._maybe_train(self.event_log[-1])
    
    def after_gate(self, gate: str, register: Any, context: Dict[str, Any]) -> None:
        """
        Hook called after a quantum gate is applied.
        
        Args:
            gate: Gate name (e.g., 'H', 'CNOT', 'X')
            register: Register or qubit being operated on
            context: Additional context (e.g., number of qubits, state)
        """
        self._log_event('after_gate', gate, register, context)
        if self.event_log:
            self._maybe_train(self.event_log[-1])
    
    def before_measurement(self, register: Any, context: Dict[str, Any]) -> None:
        """
        Hook called before a quantum measurement.
        
        Args:
            register: Register being measured
            context: Additional context
        """
        self._log_event('before_measurement', 'measure', register, context)
        if self.event_log:
            self._maybe_train(self.event_log[-1])
    
    def after_measurement(self, register: Any, result: Any, context: Dict[str, Any]) -> None:
        """
        Hook called after a quantum measurement.
        
        Args:
            register: Register that was measured
            result: Measurement result
            context: Additional context including result
        """
        context = context.copy()
        context['result'] = result
        self._log_event('after_measurement', 'measure', register, context)
        if self.event_log:
            self._maybe_train(self.event_log[-1])
    
    def log_expansion(self, description: str, context: Dict[str, Any]) -> None:
        """
        Log quantum system expansion (e.g., adding qubits).
        
        Args:
            description: Description of the expansion
            context: Additional context
        """
        self._log_event('expansion', description, None, context)
        if self.event_log:
            self._maybe_train(self.event_log[-1])
    
    def log_diagnostic(self, diagnostic_type: str, data: Any, context: Dict[str, Any]) -> None:
        """
        Log diagnostic information.
        
        Args:
            diagnostic_type: Type of diagnostic (e.g., 'state_vector', 'fidelity')
            data: Diagnostic data
            context: Additional context
        """
        context = context.copy()
        context['diagnostic_data'] = data
        self._log_event('diagnostic', diagnostic_type, None, context)
        if self.event_log:
            self._maybe_train(self.event_log[-1])
    
    # Data access methods
    
    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get logged events, optionally filtered by type.
        
        Args:
            event_type: Optional event type filter
            
        Returns:
            List of event dictionaries
        """
        if event_type is None:
            return self.event_log.copy()
        return [e for e in self.event_log if e['event_type'] == event_type]
    
    def get_event_count(self) -> int:
        """
        Get total number of logged events.
        
        Returns:
            Number of logged events
        """
        return len(self.event_log)
    
    def clear_events(self) -> None:
        """Clear all logged events and buffers."""
        self.event_log.clear()
        self.feature_buffer.clear()
        self.label_buffer.clear()
        self.logger.info("Cleared all events and buffers")
    
    def get_model(self) -> Optional[Any]:
        """
        Get the current ML model.
        
        Returns:
            The ML model or None
        """
        return self.model
    
    def set_model(self, model: Any) -> None:
        """
        Set or update the ML model.
        
        Args:
            model: New ML model with partial_fit method
        """
        self.model = model
        self.logger.info(f"Model updated to {type(model).__name__}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged events and training.
        
        Returns:
            Dictionary with statistics
        """
        event_types = {}
        for event in self.event_log:
            et = event['event_type']
            event_types[et] = event_types.get(et, 0) + 1
        
        return {
            'total_events': len(self.event_log),
            'event_types': event_types,
            'buffer_size': len(self.feature_buffer),
            'model_available': self.model is not None,
            'train_on_fly': self.train_on_fly
        }
