#!/usr/bin/env python3
"""
Demonstration of Hamiltonian Support and Quantum Profiles

This script demonstrates all the new features:
1. Creating and manipulating Hamiltonians
2. Using different quantum processor profiles
3. ML-guided Hamiltonian learning
4. Profile management and comparison
5. Integration with QuantumVirtualProcessor
"""

import numpy as np
from quantum_profiles import (
    Hamiltonian,
    AlternativeProfile,
    CastleProfile,
    HiveProfile,
    EmpireProfile,
    OmegaProfile,
    MLQuantumAddon,
    QuantumProfileManager
)
from quantum_virtual_processor import QuantumVirtualProcessor


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_hamiltonian_basics():
    """Demonstrate basic Hamiltonian operations."""
    print_section("1. Basic Hamiltonian Operations")
    
    # Create a simple 3-qubit Hamiltonian
    h = Hamiltonian(n_qubits=3)
    print(f"\nCreated Hamiltonian for {h.n_qubits} qubits")
    
    # Add terms
    h.add_term(1.0, "ZII")  # Z on qubit 0
    h.add_term(0.5, "IZI")  # Z on qubit 1
    h.add_term(0.3, "IIZ")  # Z on qubit 2
    h.add_term(0.2, "XXI")  # X-X coupling on qubits 0-1
    
    print(f"\nHamiltonian with {h.get_num_terms()} terms:")
    print(h)
    
    # Compute properties
    ground_energy, ground_state = h.get_ground_state()
    print(f"\nGround state energy: {ground_energy:.4f}")
    
    # Test state
    test_state = np.zeros(2**3, dtype=complex)
    test_state[0] = 1.0  # |000⟩
    energy = h.compute_energy(test_state)
    print(f"Energy of |000⟩ state: {energy:.4f}")
    
    # Time evolution
    evolved = h.time_evolution(test_state, time=0.5)
    print(f"After time evolution (t=0.5), state norm: {np.linalg.norm(evolved):.6f}")


def demo_quantum_profiles():
    """Demonstrate different quantum processor profiles."""
    print_section("2. Quantum Processor Profiles")
    
    n_qubits = 4
    profiles = [
        ("Alternative", AlternativeProfile(n_qubits)),
        ("Castle", CastleProfile(n_qubits)),
        ("Hive", HiveProfile(n_qubits)),
        ("Empire", EmpireProfile(n_qubits)),
        ("Omega", OmegaProfile(n_qubits))
    ]
    
    print(f"\nComparing {len(profiles)} profiles with {n_qubits} qubits:\n")
    
    for name, profile in profiles:
        h = profile.get_hamiltonian()
        energy = profile.compute_energy()
        ground_energy, _ = profile.get_ground_state()
        
        print(f"{name:12} - Terms: {h.get_num_terms():3d}, "
              f"Initial E: {energy:8.4f}, "
              f"Ground E: {ground_energy:8.4f}")
        
        # Show characteristic feature
        if name == "Alternative":
            print(f"             → Emphasizes X/Y operators for superposition exploration")
        elif name == "Castle":
            print(f"             → Strong Z fields create defensive energy barriers")
        elif name == "Hive":
            print(f"             → All-to-all coupling for collective behavior")
        elif name == "Empire":
            print(f"             → Hierarchical structure from central control")
        elif name == "Omega":
            print(f"             → Balanced mix of all operators and interactions")


def demo_ml_quantum_addon():
    """Demonstrate ML quantum addon capabilities."""
    print_section("3. ML Quantum Addon - Hamiltonian Learning")
    
    # Create a profile and ML addon
    profile = AlternativeProfile(n_qubits=3)
    ml = MLQuantumAddon(profile=profile, learning_rate=0.05)
    
    print("\nML Quantum Addon attached to Alternative profile")
    
    # Log some measurements
    h = profile.get_hamiltonian()
    for i in range(5):
        state = profile.get_state()
        # Slightly perturb state
        state = state + 0.1 * np.random.randn(len(state))
        state = state / np.linalg.norm(state)
        
        expectation = ml.log_expectation_value(h, state, f"measurement_{i}")
        print(f"  Measurement {i+1}: Expectation = {expectation:.4f}")
    
    # Get statistics
    stats = ml.get_expectation_statistics()
    print(f"\nExpectation value statistics:")
    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}, Max: {stats['max']:.4f}")
    
    # Optimize Hamiltonian parameters
    print("\nOptimizing Hamiltonian to target energy...")
    target_energy = 0.0
    state = profile.get_state()
    
    initial_energy = h.compute_energy(state)
    print(f"  Initial energy: {initial_energy:.4f}")
    print(f"  Target energy:  {target_energy:.4f}")
    
    error, iterations = ml.optimize_hamiltonian_parameters(
        h, state, target_energy, num_iterations=30, convergence_threshold=0.05
    )
    
    final_energy = h.compute_energy(state)
    print(f"  Final energy:   {final_energy:.4f}")
    print(f"  Iterations:     {iterations}")
    print(f"  Final error:    {error:.4f}")


def demo_profile_manager():
    """Demonstrate profile manager functionality."""
    print_section("4. Quantum Profile Manager")
    
    # Create profile manager
    manager = QuantumProfileManager(default_profile='alternative', n_qubits=4)
    print(f"\nProfile Manager initialized with '{manager.get_active_profile_name()}' profile")
    print(f"Available profiles: {', '.join(manager.list_profiles())}")
    
    # Add a custom term
    print("\nAdding custom Hamiltonian term (0.8 * XYZI)...")
    manager.add_term(0.8, "XYZI")
    
    # Compute energy
    energy = manager.compute_energy()
    print(f"Energy with custom term: {energy:.4f}")
    
    # Compare across profiles
    print("\nComparing energy across all profiles for same state:")
    energies = manager.compare_profiles()
    for profile_name, energy in sorted(energies.items()):
        print(f"  {profile_name:12}: {energy:8.4f}")
    
    # Switch profiles
    print("\nSwitching to 'empire' profile...")
    manager.switch_profile('empire', copy_state=True)
    print(f"Active profile: {manager.get_active_profile_name()}")
    
    new_energy = manager.compute_energy()
    print(f"Energy after switch: {new_energy:.4f}")
    
    # Get summary
    print("\nProfile Manager Summary:")
    summary = manager.get_summary()
    for key, value in summary.items():
        if key != 'ml_addon_summary':
            print(f"  {key}: {value}")


def demo_quantum_virtual_processor():
    """Demonstrate enhanced QuantumVirtualProcessor."""
    print_section("5. Enhanced QuantumVirtualProcessor Integration")
    
    # Create processor with specific profile
    qvp = QuantumVirtualProcessor(qubits=4, profile='omega')
    print(f"\nQuantumVirtualProcessor initialized with Omega profile")
    
    # Access Hamiltonian
    h = qvp.get_hamiltonian()
    print(f"Hamiltonian has {h.get_num_terms()} terms")
    
    # Add custom term
    qvp.add_hamiltonian_term(0.5, "ZXYZ")
    print(f"Added custom term, now {qvp.get_hamiltonian().get_num_terms()} terms")
    
    # Compute energy
    energy = qvp.compute_energy()
    print(f"Current energy: {energy:.4f}")
    
    # Time evolution
    print("\nPerforming time evolution...")
    evolved = qvp.time_evolve(time=0.3)
    print(f"State evolved, norm preserved: {np.linalg.norm(evolved):.6f}")
    
    # Switch profile
    print("\nSwitching to Castle profile...")
    qvp.switch_profile('castle')
    
    # Access ML addon
    ml = qvp.get_ml_addon()
    print(f"ML addon attached: {ml}")
    
    # Get summary
    print("\nProcessor Summary:")
    summary = qvp.get_summary()
    print(f"  Active profile: {summary['active_profile']}")
    print(f"  Qubits: {summary['n_qubits']}")
    print(f"  Hamiltonian terms: {summary['hamiltonian_terms']}")
    print(f"  Current energy: {summary['current_energy']:.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  HAMILTONIAN SUPPORT & QUANTUM PROFILES DEMONSTRATION")
    print("=" * 70)
    print("\nShowcasing deep Hamiltonian machine learning integration")
    print("in Pandora's quantum interface with 5 processor profiles:")
    print("Alternative, Castle, Hive, Empire, and Omega")
    
    demo_hamiltonian_basics()
    demo_quantum_profiles()
    demo_ml_quantum_addon()
    demo_profile_manager()
    demo_quantum_virtual_processor()
    
    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll Hamiltonian support features have been successfully demonstrated!")
    print("The quantum interface is now fully integrated with:")
    print("  ✓ Robust Hamiltonian class with Pauli operators")
    print("  ✓ Five specialized quantum processor profiles")
    print("  ✓ ML-aware Hamiltonian learning and optimization")
    print("  ✓ Unified profile management interface")
    print("  ✓ Enhanced QuantumVirtualProcessor integration")
    print()


if __name__ == "__main__":
    main()
