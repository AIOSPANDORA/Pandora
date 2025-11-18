"""
Base Quantum Profile

Provides common interface and functionality for all quantum processor profiles.
All profiles support Hamiltonian operations and integrate with quantum state management.
"""

import numpy as np
from typing import Optional, Tuple
from .hamiltonian import Hamiltonian


class BaseQuantumProfile:
    """
    Base class for quantum processor profiles with Hamiltonian support.
    
    All profile subclasses inherit Hamiltonian management and common quantum operations.
    Each profile can define its own characteristic Hamiltonian structure and behavior.
    
    Attributes:
        name (str): Profile name
        n_qubits (int): Number of qubits
        hamiltonian (Hamiltonian): Hamiltonian for this profile
        state (np.ndarray): Current quantum state vector
    """
    
    def __init__(self, name: str, n_qubits: int = 6):
        """
        Initialize a quantum profile.
        
        Args:
            name (str): Name of the profile
            n_qubits (int): Number of qubits (default: 6)
        """
        self.name = name
        self.n_qubits = n_qubits
        self.hamiltonian = Hamiltonian(n_qubits)
        
        # Initialize to |0...0⟩ state
        self.state = np.zeros(2 ** n_qubits, dtype=complex)
        self.state[0] = 1.0  # Ground state
    
    def add_term(self, coefficient: float, pauli_string: str) -> None:
        """
        Add a term to the profile's Hamiltonian.
        
        Args:
            coefficient (float): Weight/coefficient for this term
            pauli_string (str): String of Pauli operators
        """
        self.hamiltonian.add_term(coefficient, pauli_string)
    
    def compute_energy(self, state: Optional[np.ndarray] = None) -> float:
        """
        Compute energy expectation value for the given state (or current state).
        
        Args:
            state (np.ndarray, optional): State to compute energy for.
                                         If None, uses current state.
        
        Returns:
            float: Energy expectation value <ψ|H|ψ>
        """
        if state is None:
            state = self.state
        return self.hamiltonian.compute_energy(state)
    
    def get_hamiltonian(self) -> Hamiltonian:
        """
        Get the profile's Hamiltonian object.
        
        Returns:
            Hamiltonian: The Hamiltonian for this profile
        """
        return self.hamiltonian
    
    def time_evolution(self, time: float, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evolve a state (or current state) under this profile's Hamiltonian.
        
        Args:
            time (float): Evolution time
            state (np.ndarray, optional): State to evolve. If None, uses and updates current state.
        
        Returns:
            np.ndarray: Evolved state
        """
        if state is None:
            state = self.state
            # Update current state with evolution
            self.state = self.hamiltonian.time_evolution(state, time)
            return self.state
        else:
            # Just return evolved state without updating internal state
            return self.hamiltonian.time_evolution(state, time)
    
    def get_state(self) -> np.ndarray:
        """
        Get the current quantum state.
        
        Returns:
            np.ndarray: Current state vector
        """
        return self.state.copy()
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Set the quantum state.
        
        Args:
            state (np.ndarray): New state vector (must be normalized)
        """
        if state.shape[0] != 2 ** self.n_qubits:
            raise ValueError(
                f"State dimension ({state.shape[0]}) must match "
                f"Hilbert space dimension ({2 ** self.n_qubits})"
            )
        
        # Normalize if not already normalized
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0):
            state = state / norm
        
        self.state = state
    
    def get_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Compute and return the ground state of the Hamiltonian.
        
        Returns:
            Tuple[float, np.ndarray]: (ground_energy, ground_state)
        """
        return self.hamiltonian.get_ground_state()
    
    def initialize_characteristic_hamiltonian(self) -> None:
        """
        Initialize profile-specific characteristic Hamiltonian.
        
        Should be overridden by subclasses to define their unique Hamiltonians.
        """
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}Profile({self.n_qubits} qubits, {self.hamiltonian.get_num_terms()} H-terms)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<{self.name}Profile(n_qubits={self.n_qubits}, hamiltonian={self.hamiltonian})>"
