"""
Quantum Profile Manager

Manages quantum processor profiles and provides unified Hamiltonian-aware interface.
Allows switching between profiles and accessing Hamiltonian operations regardless
of the active profile.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from .alternative import AlternativeProfile
from .castle import CastleProfile
from .hive import HiveProfile
from .empire import EmpireProfile
from .omega import OmegaProfile
from .ml_quantum_addon import MLQuantumAddon
from .hamiltonian import Hamiltonian


class QuantumProfileManager:
    """
    Manages quantum processor profiles with unified Hamiltonian interface.
    
    Provides:
    - Profile switching and management
    - Unified Hamiltonian operations across all profiles
    - ML quantum addon integration
    - State synchronization across profiles
    
    Attributes:
        profiles: Dictionary of available profiles
        active_profile: Currently active profile
        ml_addon: ML quantum addon for learning and optimization
    """
    
    AVAILABLE_PROFILES = {
        'alternative': AlternativeProfile,
        'castle': CastleProfile,
        'hive': HiveProfile,
        'empire': EmpireProfile,
        'omega': OmegaProfile,
    }
    
    def __init__(self, default_profile: str = 'alternative', n_qubits: int = 6):
        """
        Initialize the Profile Manager.
        
        Args:
            default_profile: Name of default profile to activate
            n_qubits: Number of qubits for all profiles
        """
        if default_profile not in self.AVAILABLE_PROFILES:
            raise ValueError(
                f"Unknown profile: {default_profile}. "
                f"Available: {list(self.AVAILABLE_PROFILES.keys())}"
            )
        
        self.n_qubits = n_qubits
        self.profiles: Dict[str, Any] = {}
        self.active_profile_name = default_profile
        
        # Initialize default profile
        self._initialize_profile(default_profile)
        self.active_profile = self.profiles[default_profile]
        
        # Initialize ML addon attached to active profile
        self.ml_addon = MLQuantumAddon(profile=self.active_profile)
    
    def _initialize_profile(self, profile_name: str) -> None:
        """
        Initialize a profile if not already initialized.
        
        Args:
            profile_name: Name of profile to initialize
        """
        if profile_name not in self.profiles:
            profile_class = self.AVAILABLE_PROFILES[profile_name]
            self.profiles[profile_name] = profile_class(n_qubits=self.n_qubits)
    
    def switch_profile(self, profile_name: str, copy_state: bool = True) -> None:
        """
        Switch to a different quantum profile.
        
        Args:
            profile_name: Name of profile to switch to
            copy_state: If True, copy current state to new profile
        """
        if profile_name not in self.AVAILABLE_PROFILES:
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Available: {list(self.AVAILABLE_PROFILES.keys())}"
            )
        
        # Initialize profile if needed
        self._initialize_profile(profile_name)
        
        # Copy state if requested
        if copy_state and self.active_profile is not None:
            old_state = self.active_profile.get_state()
            self.profiles[profile_name].set_state(old_state)
        
        # Switch active profile
        self.active_profile = self.profiles[profile_name]
        self.active_profile_name = profile_name
        
        # Update ML addon attachment
        self.ml_addon.attach_profile(self.active_profile)
    
    def get_profile(self, profile_name: Optional[str] = None):
        """
        Get a specific profile (or active profile if name is None).
        
        Args:
            profile_name: Profile name, or None for active profile
        
        Returns:
            Quantum profile instance
        """
        if profile_name is None:
            return self.active_profile
        
        self._initialize_profile(profile_name)
        return self.profiles[profile_name]
    
    def get_active_profile_name(self) -> str:
        """
        Get the name of the currently active profile.
        
        Returns:
            str: Active profile name
        """
        return self.active_profile_name
    
    def list_profiles(self) -> List[str]:
        """
        List all available profile names.
        
        Returns:
            List[str]: Available profile names
        """
        return list(self.AVAILABLE_PROFILES.keys())
    
    def add_term(self, coefficient: float, pauli_string: str) -> None:
        """
        Add a term to the active profile's Hamiltonian.
        
        Args:
            coefficient: Weight/coefficient for this term
            pauli_string: String of Pauli operators
        """
        self.active_profile.add_term(coefficient, pauli_string)
    
    def compute_energy(self, state: Optional[np.ndarray] = None) -> float:
        """
        Compute energy expectation value for active profile.
        
        Args:
            state: State to compute energy for (None = current state)
        
        Returns:
            float: Energy expectation value
        """
        return self.active_profile.compute_energy(state)
    
    def get_hamiltonian(self) -> Hamiltonian:
        """
        Get the active profile's Hamiltonian.
        
        Returns:
            Hamiltonian: Active profile's Hamiltonian
        """
        return self.active_profile.get_hamiltonian()
    
    def time_evolution(self, time: float, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform time evolution under active profile's Hamiltonian.
        
        Args:
            time: Evolution time
            state: State to evolve (None = current state)
        
        Returns:
            np.ndarray: Evolved state
        """
        return self.active_profile.time_evolution(time, state)
    
    def get_state(self) -> np.ndarray:
        """
        Get the current quantum state of active profile.
        
        Returns:
            np.ndarray: Current state vector
        """
        return self.active_profile.get_state()
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Set the quantum state of active profile.
        
        Args:
            state: New state vector
        """
        self.active_profile.set_state(state)
    
    def get_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Compute ground state of active profile's Hamiltonian.
        
        Returns:
            Tuple[float, np.ndarray]: (ground_energy, ground_state)
        """
        return self.active_profile.get_ground_state()
    
    def get_ml_addon(self) -> MLQuantumAddon:
        """
        Get the ML quantum addon.
        
        Returns:
            MLQuantumAddon: ML addon instance
        """
        return self.ml_addon
    
    def compare_profiles(self, state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compare energy expectation values across all profiles for a given state.
        
        Args:
            state: State to evaluate (None = current state)
        
        Returns:
            Dict[str, float]: Profile name -> energy mapping
        """
        if state is None:
            state = self.get_state()
        
        energies = {}
        for profile_name in self.AVAILABLE_PROFILES:
            self._initialize_profile(profile_name)
            profile = self.profiles[profile_name]
            energies[profile_name] = profile.compute_energy(state)
        
        return energies
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of profile manager state.
        
        Returns:
            Dict[str, Any]: Summary information
        """
        return {
            'active_profile': self.active_profile_name,
            'n_qubits': self.n_qubits,
            'available_profiles': self.list_profiles(),
            'initialized_profiles': list(self.profiles.keys()),
            'hamiltonian_terms': self.get_hamiltonian().get_num_terms(),
            'ml_addon_summary': self.ml_addon.get_summary(),
            'current_energy': self.compute_energy(),
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"QuantumProfileManager(active={self.active_profile_name}, "
            f"n_qubits={self.n_qubits}, "
            f"profiles={len(self.profiles)} initialized)"
        )
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<QuantumProfileManager({self.active_profile_name})>"
