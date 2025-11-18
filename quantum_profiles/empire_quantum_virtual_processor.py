"""
Empire Quantum Virtual Processor

A high-performance quantum processor profile focused on optimized quantum computation
with integrated ML-based process logging and adaptive learning.

Philosophy:
- Emphasizes performance and optimization
- Aggressive computation strategies
- Self-instrumenting with ML-based monitoring
- Learns from all quantum operations for maximum efficiency
"""

from quantum_virtual_processor import QuantumVirtualProcessor
from quantum_profiles.ml_quantum_addon import MLQuantumAddon


class EmpireQuantumVirtualProcessor(QuantumVirtualProcessor):
    """
    Empire quantum processor with ML-based self-instrumentation.
    
    This profile emphasizes high-performance quantum computation while
    automatically logging and learning from all operations.
    
    Usage:
        >>> processor = EmpireQuantumVirtualProcessor(qubits=6)
        >>> processor.apply_gate("H", 0)
        >>> result = processor.measure()
        >>> stats = processor.get_addon_statistics()
    """
    
    # Define addons list with ML quantum addon enabled by default
    ADDONS = [
        MLQuantumAddon(train_on_fly=True)
    ]
    
    def __init__(self, qubits=6, **kwargs):
        """
        Initialize Empire quantum processor with ML instrumentation.
        
        Args:
            qubits: Number of qubits (default: 6)
            **kwargs: Additional arguments
        """
        super().__init__(qubits)
        
        # Initialize addons
        self.addons = [addon for addon in self.ADDONS]
        
        # Store processor context for addon hooks
        self._context = {
            'qubits': qubits,
            'profile': 'Empire'
        }
    
    def _before_gate_hooks(self, gate: str, register: int) -> None:
        """
        Call all addon before_gate hooks.
        
        Args:
            gate: Gate name
            register: Register/qubit index
        """
        for addon in self.addons:
            if hasattr(addon, 'before_gate'):
                addon.before_gate(gate, register, self._context)
    
    def _after_gate_hooks(self, gate: str, register: int) -> None:
        """
        Call all addon after_gate hooks.
        
        Args:
            gate: Gate name
            register: Register/qubit index
        """
        for addon in self.addons:
            if hasattr(addon, 'after_gate'):
                addon.after_gate(gate, register, self._context)
    
    def _before_measurement_hooks(self, register: int = None) -> None:
        """
        Call all addon before_measurement hooks.
        
        Args:
            register: Optional register/qubit index
        """
        for addon in self.addons:
            if hasattr(addon, 'before_measurement'):
                addon.before_measurement(register, self._context)
    
    def _after_measurement_hooks(self, register: int, result: int) -> None:
        """
        Call all addon after_measurement hooks.
        
        Args:
            register: Register/qubit index
            result: Measurement result
        """
        for addon in self.addons:
            if hasattr(addon, 'after_measurement'):
                addon.after_measurement(register, result, self._context)
    
    def apply_gate(self, gate: str, register: int, **kwargs) -> None:
        """
        Apply a quantum gate with ML monitoring.
        
        Args:
            gate: Gate name (e.g., 'H', 'X', 'CNOT')
            register: Register/qubit to apply gate to
            **kwargs: Additional gate parameters
        """
        # Update context with any additional parameters
        context = self._context.copy()
        context.update(kwargs)
        self._context = context
        
        # Before hooks
        self._before_gate_hooks(gate, register)
        
        # Apply the gate (call parent implementation)
        super().apply_gate(gate, register)
        
        # After hooks
        self._after_gate_hooks(gate, register)
    
    def measure(self, register: int = None) -> int:
        """
        Measure quantum register with ML monitoring.
        
        Args:
            register: Optional register to measure (None = measure all)
            
        Returns:
            Measurement result
        """
        # Before hooks
        self._before_measurement_hooks(register)
        
        # Perform measurement (call parent implementation)
        result = super().measure()
        
        # After hooks
        self._after_measurement_hooks(register if register is not None else 0, result)
        
        return result
    
    def expand_qubits(self, additional_qubits: int) -> None:
        """
        Expand the quantum system with additional qubits.
        
        Args:
            additional_qubits: Number of qubits to add
        """
        old_qubits = self.qubits
        self.qubits += additional_qubits
        self._context['qubits'] = self.qubits
        
        # Log expansion to addons
        for addon in self.addons:
            if hasattr(addon, 'log_expansion'):
                addon.log_expansion(
                    f"Expanded from {old_qubits} to {self.qubits} qubits",
                    self._context
                )
    
    def get_diagnostic_info(self) -> dict:
        """
        Get diagnostic information about processor state.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostic = {
            'qubits': self.qubits,
            'profile': 'Empire',
            'addons': len(self.addons)
        }
        
        # Log diagnostic to addons
        for addon in self.addons:
            if hasattr(addon, 'log_diagnostic'):
                addon.log_diagnostic('processor_state', diagnostic, self._context)
        
        return diagnostic
    
    def get_addon_statistics(self) -> dict:
        """
        Get statistics from all addons.
        
        Returns:
            Dictionary mapping addon index to statistics
        """
        stats = {}
        for i, addon in enumerate(self.addons):
            if hasattr(addon, 'get_statistics'):
                stats[f'addon_{i}'] = addon.get_statistics()
        return stats
    
    def get_ml_addon(self) -> MLQuantumAddon:
        """
        Get the ML quantum addon instance.
        
        Returns:
            MLQuantumAddon instance or None if not found
        """
        for addon in self.addons:
            if isinstance(addon, MLQuantumAddon):
                return addon
        return None
