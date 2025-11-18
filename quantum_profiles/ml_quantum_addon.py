"""
MLQuantumAddon - Machine Learning Quantum Process Logging and Training

This addon provides on-the-fly machine learning capabilities for quantum processors,
logging all quantum operations and supporting incremental training of ML models.

Features:
- Logs quantum gates, measurements, expansion, and diagnostics
- Supports on-the-fly incremental training (default: enabled)
- Compatible with scikit-learn and custom pluggable models
- Self-instrumenting for adaptive learning

Usage:
    Basic usage with default settings:
    >>> from quantum_profiles.ml_quantum_addon import MLQuantumAddon
    >>> addon = MLQuantumAddon()
    >>> addon.before_gate("H", 0, {"qubits": 4})
    >>> addon.after_gate("H", 0, {"qubits": 4})
    
    With custom ML model (scikit-learn):
    >>> from sklearn.linear_model import SGDClassifier
    >>> model = SGDClassifier()
    >>> addon = MLQuantumAddon(model=model, train_on_fly=True)
    
    With custom pluggable model:
    >>> class CustomModel:
    ...     def partial_fit(self, X, y, classes=None):
    ...         # Your incremental learning logic
    ...         pass
    >>> addon = MLQuantumAddon(model=CustomModel(), train_on_fly=True)
    
    Accessing logged events:
    >>> events = addon.get_events()
    >>> print(f"Logged {len(events)} events")
"""

import time
import logging
from typing import Any, Dict, List, Optional, Callable
import numpy as np


class MLQuantumAddon:
    """
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
        """
        Clear all logged events and buffers.
        """
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
