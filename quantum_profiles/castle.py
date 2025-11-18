"""
Castle Quantum Profile

A quantum processor profile designed for defensive and stable quantum operations.
Features a Hamiltonian with strong Z-field terms creating protective barriers.
"""

import numpy as np
from .base_profile import BaseQuantumProfile


class CastleProfile(BaseQuantumProfile):
    """
    Castle quantum processor profile.
    
    Characteristic Hamiltonian emphasizes stability and defensive structures
    through dominant Z-field terms creating energy barriers.
    
    Default structure:
    - Strong longitudinal Z fields for stable computational basis
    - Nearest-neighbor Z-Z interactions forming barriers
    - Weak X perturbations for controlled transitions
    """
    
    def __init__(self, n_qubits: int = 6):
        """
        Initialize the Castle profile.
        
        Args:
            n_qubits (int): Number of qubits (default: 6)
        """
        super().__init__("Castle", n_qubits)
        self.initialize_characteristic_hamiltonian()
    
    def initialize_characteristic_hamiltonian(self) -> None:
        """
        Initialize the Castle profile's characteristic Hamiltonian.
        
        Creates a Hamiltonian with:
        - Strong Z fields creating stable basis states
        - Z-Z nearest-neighbor interactions forming protective barriers
        - Weak X terms for controlled tunneling
        """
        # Strong Z field on each qubit (defensive stance)
        for i in range(self.n_qubits):
            pauli_string = 'I' * i + 'Z' + 'I' * (self.n_qubits - i - 1)
            self.add_term(2.0, pauli_string)
        
        # Strong nearest-neighbor Z-Z coupling (barriers)
        for i in range(self.n_qubits - 1):
            pauli_string = 'I' * i + 'ZZ' + 'I' * (self.n_qubits - i - 2)
            self.add_term(1.5, pauli_string)
        
        # Weak X perturbations for controlled transitions
        for i in range(1, self.n_qubits, 2):
            pauli_string = 'I' * i + 'X' + 'I' * (self.n_qubits - i - 1)
            self.add_term(0.3, pauli_string)
