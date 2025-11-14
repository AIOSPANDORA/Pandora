"""
Hive Quantum Profile

A quantum processor profile inspired by collective behavior and interconnectedness.
Features a Hamiltonian with all-to-all connectivity and collective interactions.
"""

import numpy as np
from .base_profile import BaseQuantumProfile


class HiveProfile(BaseQuantumProfile):
    """
    Hive quantum processor profile.
    
    Characteristic Hamiltonian emphasizes collective behavior through
    multi-qubit interactions and distributed entanglement.
    
    Default structure:
    - All-to-all Z-Z interactions for collective coupling
    - Distributed X fields for collective superposition
    - Y terms for global phase coherence
    """
    
    def __init__(self, n_qubits: int = 6):
        """
        Initialize the Hive profile.
        
        Args:
            n_qubits (int): Number of qubits (default: 6)
        """
        super().__init__("Hive", n_qubits)
        self.initialize_characteristic_hamiltonian()
    
    def initialize_characteristic_hamiltonian(self) -> None:
        """
        Initialize the Hive profile's characteristic Hamiltonian.
        
        Creates a Hamiltonian with:
        - All-to-all Z-Z interactions (collective coupling)
        - Uniform X fields (collective superposition)
        - Global Y terms for phase coherence
        """
        # All-to-all Z-Z coupling (collective interactions)
        coupling_strength = 0.5 / self.n_qubits  # Normalize by system size
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                pauli_string = ['I'] * self.n_qubits
                pauli_string[i] = 'Z'
                pauli_string[j] = 'Z'
                self.add_term(coupling_strength, ''.join(pauli_string))
        
        # Uniform X field for collective superposition
        for i in range(self.n_qubits):
            pauli_string = 'I' * i + 'X' + 'I' * (self.n_qubits - i - 1)
            self.add_term(0.8, pauli_string)
        
        # Global Y terms for phase coherence (on subset of qubits)
        for i in range(0, self.n_qubits, 3):
            pauli_string = 'I' * i + 'Y' + 'I' * (self.n_qubits - i - 1)
            self.add_term(0.3, pauli_string)
