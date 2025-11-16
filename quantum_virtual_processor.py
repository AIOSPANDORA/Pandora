"""
QuantumVirtualProcessor
Simulates quantum registers/gates, integrates with classical and AI routines.
Philosophy: Accepts uncertainty, works in harmony with classical logic.

Now enhanced with Hamiltonian support through quantum profiles:
- Alternative: Explores alternative computational pathways via superposition
- Castle: Defensive stable quantum operations with energy barriers
- Hive: Collective behavior through distributed entanglement
- Empire: Hierarchical control with cascading interactions
- Omega: Optimized balanced configuration across all operators
"""

from quantum_profiles import QuantumProfileManager


class QuantumVirtualProcessor:
    """
    Enhanced quantum virtual processor with Hamiltonian-aware profile support.
    
    Provides quantum simulation capabilities with multiple processor profiles,
    each with characteristic Hamiltonians for different computational paradigms.
    """
    
    def __init__(self, qubits=6, profile='alternative'):
        """
        Initialize the quantum virtual processor.
        
        Args:
            qubits: Number of qubits (default: 6)
            profile: Initial quantum profile ('alternative', 'castle', 'hive', 'empire', 'omega')
        """
        self.qubits = qubits
        
        # Initialize profile manager with Hamiltonian support
        self.profile_manager = QuantumProfileManager(
            default_profile=profile,
            n_qubits=qubits
        )
    
    def apply_gate(self, gate, reg):
        """
        Apply a quantum gate operation.
        
        Args:
            gate: Gate identifier (e.g., 'H', 'X', 'CNOT')
            reg: Register/qubit index(es) to apply gate to
        """
        # Basic gate operations (placeholder for full implementation)
        # In full implementation, this would modify the profile's quantum state
        pass
    
    def measure(self):
        """
        Perform quantum measurement, projecting to classical state.
        
        Returns:
            Measurement outcome based on current state probabilities
        """
        import numpy as np
        state = self.profile_manager.get_state()
        probabilities = np.abs(state) ** 2
        
        # Sample from probability distribution
        outcome = np.random.choice(len(probabilities), p=probabilities)
        return outcome
    
    # === Hamiltonian-aware operations ===
    
    def add_hamiltonian_term(self, coefficient, pauli_string):
        """
        Add a term to the current profile's Hamiltonian.
        
        Args:
            coefficient: Weight for this term
            pauli_string: Pauli operator string (e.g., "XYZ")
        """
        self.profile_manager.add_term(coefficient, pauli_string)
    
    def compute_energy(self, state=None):
        """
        Compute energy expectation value under current Hamiltonian.
        
        Args:
            state: Quantum state (None = current state)
        
        Returns:
            float: Energy expectation value
        """
        return self.profile_manager.compute_energy(state)
    
    def get_hamiltonian(self):
        """
        Get the current profile's Hamiltonian.
        
        Returns:
            Hamiltonian object
        """
        return self.profile_manager.get_hamiltonian()
    
    def time_evolve(self, time):
        """
        Evolve the quantum state under the Hamiltonian for given time.
        
        Args:
            time: Evolution time
        
        Returns:
            Evolved state vector
        """
        return self.profile_manager.time_evolution(time)
    
    def switch_profile(self, profile_name):
        """
        Switch to a different quantum processor profile.
        
        Args:
            profile_name: Name of profile ('alternative', 'castle', 'hive', 'empire', 'omega')
        """
        self.profile_manager.switch_profile(profile_name)
    
    def get_ml_addon(self):
        """
        Get the ML quantum addon for Hamiltonian learning.
        
        Returns:
            MLQuantumAddon instance
        """
        return self.profile_manager.get_ml_addon()
    
    def get_summary(self):
        """
        Get summary of processor state and configuration.
        
        Returns:
            Dictionary with summary information
        """
        return self.profile_manager.get_summary()