"""
Empire Quantum Profile

A quantum processor profile emphasizing hierarchical control and ordered structures.
Features a Hamiltonian with layered interactions and cascading effects.
"""

import numpy as np
from .base_profile import BaseQuantumProfile


class EmpireProfile(BaseQuantumProfile):
    """
    Empire quantum processor profile.
    
    Characteristic Hamiltonian emphasizes hierarchical structures with
    cascading interactions from central control qubits to peripheral ones.
    
    Default structure:
    - Central control qubits with strong Z fields
    - Hierarchical X-X coupling from center outward
    - Peripheral Z stabilization
    """
    
    def __init__(self, n_qubits: int = 6):
        """
        Initialize the Empire profile.
        
        Args:
            n_qubits (int): Number of qubits (default: 6)
        """
        super().__init__("Empire", n_qubits)
        self.initialize_characteristic_hamiltonian()
    
    def initialize_characteristic_hamiltonian(self) -> None:
        """
        Initialize the Empire profile's characteristic Hamiltonian.
        
        Creates a Hamiltonian with:
        - Strong Z field on central qubits (control)
        - Hierarchical X-X coupling radiating from center
        - Peripheral Z terms for boundary stabilization
        """
        # Central control qubit(s) with strong Z field
        center = self.n_qubits // 2
        center_strength = 3.0
        pauli_string = 'I' * center + 'Z' + 'I' * (self.n_qubits - center - 1)
        self.add_term(center_strength, pauli_string)
        
        # Hierarchical coupling: center to neighbors with decreasing strength
        for distance in range(1, (self.n_qubits + 1) // 2):
            strength = 1.5 / distance
            
            # Left neighbor
            left_idx = center - distance
            if left_idx >= 0:
                pauli_string = ['I'] * self.n_qubits
                pauli_string[center] = 'X'
                pauli_string[left_idx] = 'X'
                self.add_term(strength, ''.join(pauli_string))
            
            # Right neighbor
            right_idx = center + distance
            if right_idx < self.n_qubits:
                pauli_string = ['I'] * self.n_qubits
                pauli_string[center] = 'X'
                pauli_string[right_idx] = 'X'
                self.add_term(strength, ''.join(pauli_string))
        
        # Peripheral Z stabilization
        for i in [0, self.n_qubits - 1]:
            pauli_string = 'I' * i + 'Z' + 'I' * (self.n_qubits - i - 1)
            self.add_term(1.0, pauli_string)
