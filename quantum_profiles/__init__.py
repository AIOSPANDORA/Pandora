"""
Quantum Profiles Module

Provides quantum processor profiles with Hamiltonian support for
advanced quantum simulation and machine learning operations.
"""

from .hamiltonian import Hamiltonian
from .alternative import AlternativeProfile
from .castle import CastleProfile
from .hive import HiveProfile
from .empire import EmpireProfile
from .omega import OmegaProfile
from .ml_quantum_addon import MLQuantumAddon
from .profile_manager import QuantumProfileManager

__all__ = [
    'Hamiltonian',
    'AlternativeProfile',
    'CastleProfile',
    'HiveProfile',
    'EmpireProfile',
    'OmegaProfile',
    'MLQuantumAddon',
    'QuantumProfileManager',
]
Quantum Profiles Registry and Factory Functions

This module provides a central registry for quantum processor profiles and their add-ons.
Pandora can dynamically discover, retrieve, and instantiate quantum processors through
this extensible interface.

Functions:
    list_profiles() -> list: Returns a list of available profile names
    get_profile(name: str) -> object: Returns an instantiated processor for the given profile
    get_addons(name: str) -> list: Returns the available addons for a given profile
"""

# Registry of available quantum processor profiles
_PROFILE_REGISTRY = {}


def register_profile(name, module_name, get_profile_func):
    """
    Register a quantum processor profile in the registry.
    
    Args:
        name (str): Unique name for the profile
        module_name (str): Module path containing the profile
        get_profile_func (callable): Function that returns an instantiated processor
    """
    _PROFILE_REGISTRY[name] = {
        'module': module_name,
        'factory': get_profile_func
    }


def list_profiles():
    """
    List all registered quantum processor profiles.
    
    Returns:
        list: Names of all available profiles
    """
    return list(_PROFILE_REGISTRY.keys())


def get_profile(name):
    """
    Retrieve and instantiate a quantum processor profile by name.
    
    Args:
        name (str): Name of the profile to retrieve
        
    Returns:
        object: Instantiated quantum processor with configured add-ons
        
    Raises:
        KeyError: If the profile name is not found in the registry
    """
    if name not in _PROFILE_REGISTRY:
        available = ', '.join(list_profiles())
        raise KeyError(f"Profile '{name}' not found. Available profiles: {available}")
    
    profile_info = _PROFILE_REGISTRY[name]
    factory = profile_info['factory']
    return factory()


def get_addons(name):
    """
    Get the list of add-ons available for a specific profile.
    
    Args:
        name (str): Name of the profile
        
    Returns:
        list: List of addon names or descriptions
        
    Raises:
        KeyError: If the profile name is not found in the registry
    """
    if name not in _PROFILE_REGISTRY:
        available = ', '.join(list_profiles())
        raise KeyError(f"Profile '{name}' not found. Available profiles: {available}")
    
    # Get the profile instance and check for addons attribute
    processor = get_profile(name)
    if hasattr(processor, 'addons'):
        return [type(addon).__name__ for addon in processor.addons]
    return []


# Import and register profiles
try:
    from . import alternative_quantum_virtual_processor
    register_profile(
        'alternative',
        'quantum_profiles.alternative_quantum_virtual_processor',
        alternative_quantum_virtual_processor.get_profile
    )
except ImportError as e:
    print(f"Warning: Could not register 'alternative' profile: {e}")

try:
    from . import four_overlay_quantum_virtual_processor
    register_profile(
        'four_overlay',
        'quantum_profiles.four_overlay_quantum_virtual_processor',
        four_overlay_quantum_virtual_processor.get_profile
    )
except ImportError as e:
    print(f"Warning: Could not register 'four_overlay' profile: {e}")


__all__ = ['list_profiles', 'get_profile', 'get_addons', 'register_profile']
Quantum Profiles Package

This package contains various quantum processor profiles that Pandora can
dynamically select and switch between.

Available Profiles:
- 'empire': Empire Quantum Virtual Processor with hierarchical architecture
"""

from quantum_profiles.empire_quantum_virtual_processor import (
    EmpireQuantumVirtualProcessor,
    HiveQuantumVirtualProcessor
)

# Registry of available quantum profiles
QUANTUM_PROFILES = {
    'empire': EmpireQuantumVirtualProcessor
}


def get_profile(profile_name, **kwargs):
    """
    Get a quantum processor profile by name.
    
    Args:
        profile_name (str): Name of the profile to retrieve (e.g., 'empire')
        **kwargs: Configuration parameters for the profile
        
    Returns:
        Instance of the requested quantum processor profile
        
    Raises:
        KeyError: If the profile name is not registered
        
    Example:
        >>> processor = get_profile('empire', control_qubits=6, grid_size=(3, 3))
    """
    if profile_name not in QUANTUM_PROFILES:
        available = ', '.join(QUANTUM_PROFILES.keys())
        raise KeyError(
            f"Profile '{profile_name}' not found. "
            f"Available profiles: {available}"
        )
    
    profile_class = QUANTUM_PROFILES[profile_name]
    return profile_class(**kwargs)


def list_profiles():
    """
    List all available quantum processor profiles.
    
    Returns:
        list: List of available profile names
    """
    return list(QUANTUM_PROFILES.keys())


__all__ = [
    'EmpireQuantumVirtualProcessor',
    'HiveQuantumVirtualProcessor',
    'QUANTUM_PROFILES',
    'get_profile',
    'list_profiles'
]

This package provides various quantum processor profiles with integrated
machine learning capabilities for adaptive quantum computation.

Profiles:
- AlternativeQuantumVirtualProcessor: Alternative approaches to quantum computation
- CastleQuantumVirtualProcessor: Defensive and secure quantum computation
- HiveQuantumVirtualProcessor: Collaborative and distributed quantum computation
- EmpireQuantumVirtualProcessor: High-performance quantum computation

All profiles include MLQuantumAddon for on-the-fly ML-based process logging and training.
"""

from quantum_profiles.ml_quantum_addon import MLQuantumAddon
from quantum_profiles.alternative_quantum_virtual_processor import AlternativeQuantumVirtualProcessor
from quantum_profiles.castle_quantum_virtual_processor import CastleQuantumVirtualProcessor
from quantum_profiles.hive_quantum_virtual_processor import HiveQuantumVirtualProcessor
from quantum_profiles.empire_quantum_virtual_processor import EmpireQuantumVirtualProcessor

__all__ = [
    'MLQuantumAddon',
    'AlternativeQuantumVirtualProcessor',
    'CastleQuantumVirtualProcessor',
    'HiveQuantumVirtualProcessor',
    'EmpireQuantumVirtualProcessor'
]

__version__ = '1.0.0'
