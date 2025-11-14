"""
Hamiltonian Module

Provides a robust Hamiltonian class for quantum simulations, supporting:
- Sum of weighted Pauli operators
- Matrix assembly
- Energy/expectation value calculations
- Time evolution

The Hamiltonian class serves as the foundation for quantum processor profiles
and enables deep Hamiltonian machine learning throughout Pandora's quantum interface.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class Hamiltonian:
    """
    Robust Hamiltonian class for quantum systems.
    
    Supports construction from weighted Pauli operators, matrix assembly,
    energy calculations, expectation values, and time evolution.
    
    Attributes:
        n_qubits (int): Number of qubits in the system
        terms (List[Tuple[float, str]]): List of (coefficient, pauli_string) terms
        matrix (np.ndarray): Cached matrix representation (computed on demand)
    """
    
    # Pauli matrices for single-qubit operations
    PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    PAULI_MAP = {
        'I': PAULI_I,
        'X': PAULI_X,
        'Y': PAULI_Y,
        'Z': PAULI_Z,
    }
    
    def __init__(self, n_qubits: int):
        """
        Initialize a Hamiltonian for n_qubits.
        
        Args:
            n_qubits (int): Number of qubits in the system
        """
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        self.n_qubits = n_qubits
        self.terms: List[Tuple[float, str]] = []
        self._matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True
    
    def add_term(self, coefficient: float, pauli_string: str) -> None:
        """
        Add a weighted Pauli operator term to the Hamiltonian.
        
        Args:
            coefficient (float): Weight/coefficient for this term
            pauli_string (str): String of Pauli operators (e.g., "XYZ", "IIZ")
                               Length must match n_qubits
        
        Example:
            h = Hamiltonian(3)
            h.add_term(1.5, "IIZ")  # 1.5 * Z on qubit 2
            h.add_term(2.0, "XXI")  # 2.0 * X⊗X on qubits 0,1
        """
        if len(pauli_string) != self.n_qubits:
            raise ValueError(
                f"Pauli string length ({len(pauli_string)}) must match "
                f"number of qubits ({self.n_qubits})"
            )
        
        # Validate pauli string contains only valid operators
        for char in pauli_string:
            if char not in self.PAULI_MAP:
                raise ValueError(f"Invalid Pauli operator: {char}")
        
        self.terms.append((coefficient, pauli_string.upper()))
        self._matrix_dirty = True
    
    def _compute_term_matrix(self, pauli_string: str) -> np.ndarray:
        """
        Compute the matrix representation of a Pauli string.
        
        Args:
            pauli_string (str): String of Pauli operators
        
        Returns:
            np.ndarray: Matrix representation of the tensor product
        """
        # Start with first qubit's operator
        result = self.PAULI_MAP[pauli_string[0]]
        
        # Tensor product with remaining qubits
        for pauli_char in pauli_string[1:]:
            result = np.kron(result, self.PAULI_MAP[pauli_char])
        
        return result
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the full matrix representation of the Hamiltonian.
        
        Returns:
            np.ndarray: 2^n × 2^n matrix representing H = Σ c_i P_i
        """
        if not self._matrix_dirty and self._matrix is not None:
            return self._matrix
        
        dim = 2 ** self.n_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for coefficient, pauli_string in self.terms:
            term_matrix = self._compute_term_matrix(pauli_string)
            matrix += coefficient * term_matrix
        
        self._matrix = matrix
        self._matrix_dirty = False
        return matrix
    
    def compute_energy(self, state: np.ndarray) -> float:
        """
        Compute the energy expectation value <ψ|H|ψ> for a given state.
        
        Args:
            state (np.ndarray): Quantum state vector (must be normalized)
        
        Returns:
            float: Energy expectation value
        """
        if state.shape[0] != 2 ** self.n_qubits:
            raise ValueError(
                f"State dimension ({state.shape[0]}) must match "
                f"Hilbert space dimension ({2 ** self.n_qubits})"
            )
        
        # Compute <ψ|H|ψ>
        H_matrix = self.get_matrix()
        H_psi = H_matrix @ state
        energy = np.vdot(state, H_psi)
        
        # Return real part (imaginary should be negligible for Hermitian H)
        return np.real(energy)
    
    def compute_expectation(self, state: np.ndarray) -> float:
        """
        Alias for compute_energy - computes expectation value <ψ|H|ψ>.
        
        Args:
            state (np.ndarray): Quantum state vector (must be normalized)
        
        Returns:
            float: Expectation value
        """
        return self.compute_energy(state)
    
    def time_evolution(self, state: np.ndarray, time: float) -> np.ndarray:
        """
        Evolve a quantum state under this Hamiltonian for a given time.
        
        Applies the unitary evolution operator U(t) = e^(-iHt) to the state.
        
        Args:
            state (np.ndarray): Initial quantum state vector
            time (float): Evolution time
        
        Returns:
            np.ndarray: Evolved state |ψ(t)⟩ = e^(-iHt)|ψ(0)⟩
        """
        if state.shape[0] != 2 ** self.n_qubits:
            raise ValueError(
                f"State dimension ({state.shape[0]}) must match "
                f"Hilbert space dimension ({2 ** self.n_qubits})"
            )
        
        # Compute evolution operator U = exp(-iHt)
        H_matrix = self.get_matrix()
        U = self._matrix_exponential(-1j * time * H_matrix)
        
        # Apply to state
        evolved_state = U @ state
        
        return evolved_state
    
    def _matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix exponential via diagonalization.
        
        For Hermitian matrices, uses eigendecomposition:
        exp(M) = V @ diag(exp(eigenvalues)) @ V†
        
        Args:
            matrix (np.ndarray): Matrix to exponentiate
        
        Returns:
            np.ndarray: Matrix exponential
        """
        # Use eigendecomposition for Hermitian matrices
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        exp_eigenvalues = np.exp(eigenvalues)
        
        # Reconstruct: V @ diag(exp(λ)) @ V†
        result = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.conj().T
        
        return result
    
    def get_ground_state_energy(self) -> float:
        """
        Compute the ground state (minimum eigenvalue) energy.
        
        Returns:
            float: Ground state energy (lowest eigenvalue of H)
        """
        H_matrix = self.get_matrix()
        eigenvalues = np.linalg.eigvalsh(H_matrix)
        return np.min(eigenvalues)
    
    def get_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Compute the ground state energy and corresponding state vector.
        
        Returns:
            Tuple[float, np.ndarray]: (ground_energy, ground_state_vector)
        """
        H_matrix = self.get_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        
        # Find minimum eigenvalue index
        min_idx = np.argmin(eigenvalues)
        ground_energy = eigenvalues[min_idx]
        ground_state = eigenvectors[:, min_idx]
        
        return ground_energy, ground_state
    
    def get_num_terms(self) -> int:
        """
        Get the number of terms in the Hamiltonian.
        
        Returns:
            int: Number of terms
        """
        return len(self.terms)
    
    def get_terms(self) -> List[Tuple[float, str]]:
        """
        Get a copy of all Hamiltonian terms.
        
        Returns:
            List[Tuple[float, str]]: List of (coefficient, pauli_string) terms
        """
        return self.terms.copy()
    
    def __str__(self) -> str:
        """String representation of the Hamiltonian."""
        if not self.terms:
            return f"Hamiltonian({self.n_qubits} qubits, 0 terms)"
        
        terms_str = []
        for coeff, pauli in self.terms:
            sign = "+" if coeff >= 0 else ""
            terms_str.append(f"{sign}{coeff:.4f}*{pauli}")
        
        return f"Hamiltonian({self.n_qubits} qubits): {' '.join(terms_str)}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<Hamiltonian(n_qubits={self.n_qubits}, n_terms={len(self.terms)})>"
