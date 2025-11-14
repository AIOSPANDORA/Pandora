"""
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
