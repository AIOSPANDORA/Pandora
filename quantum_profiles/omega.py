"""
Omega Quantum Profile

A quantum processor profile representing the ultimate or final state.
Features a Hamiltonian with balanced, optimized interactions across all operators.
"""

import numpy as np
from .base_profile import BaseQuantumProfile


class OmegaProfile(BaseQuantumProfile):
    """
    Omega quantum processor profile.
    
    Characteristic Hamiltonian represents an optimized, balanced configuration
    incorporating all Pauli operators in a harmonious arrangement.
    
    Default structure:
    - Balanced X, Y, Z single-qubit terms
    - Mixed two-qubit interactions (XX, YY, ZZ)
    - Three-body interactions for higher-order correlations
    """
    
    def __init__(self, n_qubits: int = 6):
        """
        Initialize the Omega profile.
        
        Args:
            n_qubits (int): Number of qubits (default: 6)
        """
        super().__init__("Omega", n_qubits)
        self.initialize_characteristic_hamiltonian()
    
    def initialize_characteristic_hamiltonian(self) -> None:
        """
        Initialize the Omega profile's characteristic Hamiltonian.
        
        Creates a Hamiltonian with:
        - Balanced single-qubit X, Y, Z terms
        - Mixed two-qubit XX, YY, ZZ interactions
        - Selected three-body terms for higher-order effects
        """
        # Balanced single-qubit terms (X, Y, Z) with equal weight
        base_strength = 1.0
        for i in range(self.n_qubits):
            for pauli in ['X', 'Y', 'Z']:
                pauli_string = 'I' * i + pauli + 'I' * (self.n_qubits - i - 1)
                self.add_term(base_strength / 3.0, pauli_string)
        
        # Mixed two-qubit interactions: XX, YY, ZZ nearest-neighbor
        coupling_strength = 0.7
        for i in range(self.n_qubits - 1):
            for pauli_pair in ['XX', 'YY', 'ZZ']:
                pauli_string = 'I' * i + pauli_pair + 'I' * (self.n_qubits - i - 2)
                self.add_term(coupling_strength / 3.0, pauli_string)
        
        # Three-body interactions (selected positions for complexity without explosion)
        if self.n_qubits >= 3:
            three_body_strength = 0.3
            # Add three-body terms at beginning, middle, and end
            positions = [0, max(0, self.n_qubits // 2 - 1), max(0, self.n_qubits - 3)]
            
            for pos in positions:
                if pos + 2 < self.n_qubits:
                    # XYZ pattern
                    pauli_string = 'I' * pos + 'XYZ' + 'I' * (self.n_qubits - pos - 3)
                    self.add_term(three_body_strength, pauli_string)
