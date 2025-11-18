"""
Quantum Profiles Package

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
