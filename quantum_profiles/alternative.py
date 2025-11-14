"""
Alternative Quantum Profile

A quantum processor profile emphasizing alternative quantum pathways.
Features a Hamiltonian with mixed Pauli operators promoting superposition and entanglement.
"""

import numpy as np
from .base_profile import BaseQuantumProfile


class AlternativeProfile(BaseQuantumProfile):
    """
    Alternative quantum processor profile.
    
    Characteristic Hamiltonian emphasizes X and Y operators to explore
    alternative computational pathways through superposition states.
    
    Default structure:
    - Transverse field (X terms) for superposition
    - Y terms for phase exploration
    - Weak Z coupling for stability
    """
    
    def __init__(self, n_qubits: int = 6):
        """
        Initialize the Alternative profile.
        
        Args:
            n_qubits (int): Number of qubits (default: 6)
        """
        super().__init__("Alternative", n_qubits)
        self.initialize_characteristic_hamiltonian()
    
    def initialize_characteristic_hamiltonian(self) -> None:
        """
        Initialize the Alternative profile's characteristic Hamiltonian.
        
        Creates a Hamiltonian with:
        - Strong transverse X fields on all qubits
        - Y terms for quantum phase exploration
        - Weak nearest-neighbor Z-Z coupling
        """
        # Transverse X field on each qubit
        for i in range(self.n_qubits):
            pauli_string = 'I' * i + 'X' + 'I' * (self.n_qubits - i - 1)
            self.add_term(1.0, pauli_string)
        
        # Y terms for phase exploration on alternating qubits
        for i in range(0, self.n_qubits, 2):
            pauli_string = 'I' * i + 'Y' + 'I' * (self.n_qubits - i - 1)
            self.add_term(0.5, pauli_string)
        
        # Weak nearest-neighbor Z-Z coupling for stability
        for i in range(self.n_qubits - 1):
            pauli_string = 'I' * i + 'ZZ' + 'I' * (self.n_qubits - i - 2)
            self.add_term(0.2, pauli_string)
