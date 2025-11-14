"""
Test Suite for Hamiltonian Support and Quantum Profiles

Comprehensive tests for:
- Hamiltonian class operations
- Quantum profile functionality
- MLQuantumAddon learning capabilities
- Profile manager integration
- QuantumVirtualProcessor integration
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class TestHamiltonian:
    """Test Hamiltonian class functionality."""
    
    @staticmethod
    def test_initialization():
        """Test Hamiltonian initialization."""
        print("Testing Hamiltonian initialization...")
        h = Hamiltonian(n_qubits=3)
        assert h.n_qubits == 3
        assert h.get_num_terms() == 0
        print("✓ Hamiltonian initialization passed")
    
    @staticmethod
    def test_add_term():
        """Test adding terms to Hamiltonian."""
        print("Testing add_term...")
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "ZI")
        h.add_term(0.5, "IZ")
        assert h.get_num_terms() == 2
        print("✓ add_term passed")
    
    @staticmethod
    def test_matrix_assembly():
        """Test matrix assembly for simple Hamiltonian."""
        print("Testing matrix assembly...")
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "ZI")
        matrix = h.get_matrix()
        assert matrix.shape == (4, 4)
        # Check diagonal elements (Z ⊗ I eigenvalues: [1,1,-1,-1])
        expected_diag = np.array([1, 1, -1, -1])
        assert np.allclose(np.diag(matrix), expected_diag)
        print("✓ Matrix assembly passed")
    
    @staticmethod
    def test_energy_calculation():
        """Test energy expectation value calculation."""
        print("Testing energy calculation...")
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "ZI")
        
        # Ground state |00⟩
        state = np.array([1, 0, 0, 0], dtype=complex)
        energy = h.compute_energy(state)
        assert np.isclose(energy, 1.0)
        
        # Excited state |10⟩
        state = np.array([0, 0, 1, 0], dtype=complex)
        energy = h.compute_energy(state)
        assert np.isclose(energy, -1.0)
        print("✓ Energy calculation passed")
    
    @staticmethod
    def test_time_evolution():
        """Test time evolution under Hamiltonian."""
        print("Testing time evolution...")
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "XI")
        
        # Start in |00⟩
        state = np.array([1, 0, 0, 0], dtype=complex)
        
        # Evolve for time π/2 (should rotate in X basis)
        evolved = h.time_evolution(state, np.pi / 2)
        
        # Check norm preserved
        assert np.isclose(np.linalg.norm(evolved), 1.0)
        print("✓ Time evolution passed")
    
    @staticmethod
    def test_ground_state():
        """Test ground state calculation."""
        print("Testing ground state...")
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "ZI")
        h.add_term(1.0, "IZ")
        
        ground_energy, ground_state = h.get_ground_state()
        
        # Ground state should have lowest energy
        assert ground_energy <= 0  # Both Z terms contribute -1 in |11⟩
        assert np.isclose(np.linalg.norm(ground_state), 1.0)
        print(f"✓ Ground state passed (E_0 = {ground_energy:.4f})")
    
    @staticmethod
    def run_all():
        """Run all Hamiltonian tests."""
        print("\n=== Testing Hamiltonian ===")
        TestHamiltonian.test_initialization()
        TestHamiltonian.test_add_term()
        TestHamiltonian.test_matrix_assembly()
        TestHamiltonian.test_energy_calculation()
        TestHamiltonian.test_time_evolution()
        TestHamiltonian.test_ground_state()
        print("All Hamiltonian tests passed!\n")


class TestQuantumProfiles:
    """Test quantum profile classes."""
    
    @staticmethod
    def test_alternative_profile():
        """Test Alternative profile."""
        print("Testing AlternativeProfile...")
        profile = AlternativeProfile(n_qubits=3)
        assert profile.name == "Alternative"
        assert profile.n_qubits == 3
        assert profile.hamiltonian.get_num_terms() > 0
        
        # Test Hamiltonian operations
        energy = profile.compute_energy()
        assert isinstance(energy, (float, np.floating))
        print(f"✓ AlternativeProfile passed (initial E = {energy:.4f})")
    
    @staticmethod
    def test_castle_profile():
        """Test Castle profile."""
        print("Testing CastleProfile...")
        profile = CastleProfile(n_qubits=3)
        assert profile.name == "Castle"
        energy = profile.compute_energy()
        print(f"✓ CastleProfile passed (initial E = {energy:.4f})")
    
    @staticmethod
    def test_hive_profile():
        """Test Hive profile."""
        print("Testing HiveProfile...")
        profile = HiveProfile(n_qubits=3)
        assert profile.name == "Hive"
        energy = profile.compute_energy()
        print(f"✓ HiveProfile passed (initial E = {energy:.4f})")
    
    @staticmethod
    def test_empire_profile():
        """Test Empire profile."""
        print("Testing EmpireProfile...")
        profile = EmpireProfile(n_qubits=3)
        assert profile.name == "Empire"
        energy = profile.compute_energy()
        print(f"✓ EmpireProfile passed (initial E = {energy:.4f})")
    
    @staticmethod
    def test_omega_profile():
        """Test Omega profile."""
        print("Testing OmegaProfile...")
        profile = OmegaProfile(n_qubits=3)
        assert profile.name == "Omega"
        energy = profile.compute_energy()
        print(f"✓ OmegaProfile passed (initial E = {energy:.4f})")
    
    @staticmethod
    def test_profile_operations():
        """Test common profile operations."""
        print("Testing profile operations...")
        profile = AlternativeProfile(n_qubits=3)
        
        # Test add_term
        initial_terms = profile.hamiltonian.get_num_terms()
        profile.add_term(0.5, "XYZ")
        assert profile.hamiltonian.get_num_terms() == initial_terms + 1
        
        # Test get_state/set_state
        state = profile.get_state()
        assert state.shape[0] == 2 ** 3
        
        # Test time_evolution
        evolved = profile.time_evolution(0.1)
        assert np.isclose(np.linalg.norm(evolved), 1.0)
        
        # Test ground_state
        ground_energy, ground_state = profile.get_ground_state()
        assert isinstance(ground_energy, (float, np.floating))
        
        print("✓ Profile operations passed")
    
    @staticmethod
    def run_all():
        """Run all profile tests."""
        print("\n=== Testing Quantum Profiles ===")
        TestQuantumProfiles.test_alternative_profile()
        TestQuantumProfiles.test_castle_profile()
        TestQuantumProfiles.test_hive_profile()
        TestQuantumProfiles.test_empire_profile()
        TestQuantumProfiles.test_omega_profile()
        TestQuantumProfiles.test_profile_operations()
        print("All Quantum Profile tests passed!\n")


class TestMLQuantumAddon:
    """Test ML Quantum Addon functionality."""
    
    @staticmethod
    def test_initialization():
        """Test ML addon initialization."""
        print("Testing MLQuantumAddon initialization...")
        ml = MLQuantumAddon(learning_rate=0.01)
        assert ml.learning_rate == 0.01
        assert ml.total_measurements == 0
        print("✓ MLQuantumAddon initialization passed")
    
    @staticmethod
    def test_logging():
        """Test Hamiltonian and expectation logging."""
        print("Testing logging...")
        ml = MLQuantumAddon()
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "ZI")
        
        # Log Hamiltonian
        ml.log_hamiltonian(h, metadata={'test': 'value'})
        assert len(ml.hamiltonian_history) == 1
        
        # Log expectation value
        state = np.array([1, 0, 0, 0], dtype=complex)
        ml.log_expectation_value(h, state, "test_measurement")
        assert len(ml.expectation_history) == 1
        assert ml.total_measurements == 1
        
        print("✓ Logging passed")
    
    @staticmethod
    def test_gradient_computation():
        """Test gradient computation."""
        print("Testing gradient computation...")
        ml = MLQuantumAddon()
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "ZI")
        
        state = np.array([1, 0, 0, 0], dtype=complex)
        gradient = ml.compute_gradient(h, state, param_index=0)
        
        assert isinstance(gradient, (float, np.floating))
        print(f"✓ Gradient computation passed (grad = {gradient:.6f})")
    
    @staticmethod
    def test_parameter_optimization():
        """Test Hamiltonian parameter optimization."""
        print("Testing parameter optimization...")
        ml = MLQuantumAddon(learning_rate=0.1)
        h = Hamiltonian(n_qubits=2)
        h.add_term(2.0, "ZI")  # Start with coefficient 2.0
        
        state = np.array([1, 0, 0, 0], dtype=complex)
        target_energy = 1.5
        
        error, iterations = ml.optimize_hamiltonian_parameters(
            h, state, target_energy, num_iterations=50, convergence_threshold=0.1
        )
        
        final_energy = h.compute_energy(state)
        print(f"✓ Parameter optimization passed ({iterations} iterations, final E = {final_energy:.4f}, target = {target_energy})")
    
    @staticmethod
    def test_statistics():
        """Test statistics gathering."""
        print("Testing statistics...")
        ml = MLQuantumAddon()
        h = Hamiltonian(n_qubits=2)
        h.add_term(1.0, "ZI")
        
        state = np.array([1, 0, 0, 0], dtype=complex)
        
        # Log multiple measurements
        for i in range(5):
            ml.log_expectation_value(h, state, f"measurement_{i}")
        
        stats = ml.get_expectation_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert stats['count'] == 5
        
        summary = ml.get_summary()
        assert summary['total_measurements'] == 5
        
        print("✓ Statistics passed")
    
    @staticmethod
    def run_all():
        """Run all ML addon tests."""
        print("\n=== Testing MLQuantumAddon ===")
        TestMLQuantumAddon.test_initialization()
        TestMLQuantumAddon.test_logging()
        TestMLQuantumAddon.test_gradient_computation()
        TestMLQuantumAddon.test_parameter_optimization()
        TestMLQuantumAddon.test_statistics()
        print("All MLQuantumAddon tests passed!\n")


class TestProfileManager:
    """Test QuantumProfileManager."""
    
    @staticmethod
    def test_initialization():
        """Test profile manager initialization."""
        print("Testing ProfileManager initialization...")
        manager = QuantumProfileManager(default_profile='alternative', n_qubits=3)
        assert manager.get_active_profile_name() == 'alternative'
        assert manager.n_qubits == 3
        print("✓ ProfileManager initialization passed")
    
    @staticmethod
    def test_profile_switching():
        """Test switching between profiles."""
        print("Testing profile switching...")
        manager = QuantumProfileManager(default_profile='alternative', n_qubits=3)
        
        # Switch to castle
        manager.switch_profile('castle')
        assert manager.get_active_profile_name() == 'castle'
        
        # Switch to omega
        manager.switch_profile('omega')
        assert manager.get_active_profile_name() == 'omega'
        
        print("✓ Profile switching passed")
    
    @staticmethod
    def test_unified_operations():
        """Test unified Hamiltonian operations."""
        print("Testing unified operations...")
        manager = QuantumProfileManager(default_profile='hive', n_qubits=3)
        
        # Test add_term
        manager.add_term(0.5, "XYZ")
        
        # Test compute_energy
        energy = manager.compute_energy()
        assert isinstance(energy, (float, np.floating))
        
        # Test get_hamiltonian
        h = manager.get_hamiltonian()
        assert isinstance(h, Hamiltonian)
        
        # Test time_evolution
        evolved = manager.time_evolution(0.1)
        assert np.isclose(np.linalg.norm(evolved), 1.0)
        
        print("✓ Unified operations passed")
    
    @staticmethod
    def test_profile_comparison():
        """Test comparing energies across profiles."""
        print("Testing profile comparison...")
        manager = QuantumProfileManager(default_profile='alternative', n_qubits=3)
        
        energies = manager.compare_profiles()
        
        assert len(energies) == 5  # All 5 profiles
        assert 'alternative' in energies
        assert 'castle' in energies
        assert 'hive' in energies
        assert 'empire' in energies
        assert 'omega' in energies
        
        print(f"✓ Profile comparison passed")
        print(f"  Profile energies: {', '.join([f'{k}={v:.4f}' for k, v in energies.items()])}")
    
    @staticmethod
    def test_ml_integration():
        """Test ML addon integration."""
        print("Testing ML integration...")
        manager = QuantumProfileManager(default_profile='alternative', n_qubits=3)
        
        ml = manager.get_ml_addon()
        assert isinstance(ml, MLQuantumAddon)
        assert ml.profile is not None
        
        print("✓ ML integration passed")
    
    @staticmethod
    def run_all():
        """Run all profile manager tests."""
        print("\n=== Testing QuantumProfileManager ===")
        TestProfileManager.test_initialization()
        TestProfileManager.test_profile_switching()
        TestProfileManager.test_unified_operations()
        TestProfileManager.test_profile_comparison()
        TestProfileManager.test_ml_integration()
        print("All ProfileManager tests passed!\n")


class TestQuantumVirtualProcessor:
    """Test QuantumVirtualProcessor integration."""
    
    @staticmethod
    def test_initialization():
        """Test processor initialization."""
        print("Testing QuantumVirtualProcessor initialization...")
        qvp = QuantumVirtualProcessor(qubits=4, profile='castle')
        assert qvp.qubits == 4
        print("✓ QuantumVirtualProcessor initialization passed")
    
    @staticmethod
    def test_hamiltonian_operations():
        """Test Hamiltonian operations through processor."""
        print("Testing Hamiltonian operations...")
        qvp = QuantumVirtualProcessor(qubits=3, profile='alternative')
        
        # Test add_hamiltonian_term
        qvp.add_hamiltonian_term(0.5, "XYZ")
        
        # Test compute_energy
        energy = qvp.compute_energy()
        assert isinstance(energy, (float, np.floating))
        
        # Test get_hamiltonian
        h = qvp.get_hamiltonian()
        assert isinstance(h, Hamiltonian)
        
        # Test time_evolve
        evolved = qvp.time_evolve(0.1)
        assert np.isclose(np.linalg.norm(evolved), 1.0)
        
        print("✓ Hamiltonian operations passed")
    
    @staticmethod
    def test_profile_switching():
        """Test profile switching through processor."""
        print("Testing profile switching...")
        qvp = QuantumVirtualProcessor(qubits=3, profile='alternative')
        
        qvp.switch_profile('empire')
        summary = qvp.get_summary()
        assert summary['active_profile'] == 'empire'
        
        print("✓ Profile switching passed")
    
    @staticmethod
    def test_ml_access():
        """Test ML addon access."""
        print("Testing ML access...")
        qvp = QuantumVirtualProcessor(qubits=3, profile='omega')
        
        ml = qvp.get_ml_addon()
        assert isinstance(ml, MLQuantumAddon)
        
        print("✓ ML access passed")
    
    @staticmethod
    def run_all():
        """Run all processor tests."""
        print("\n=== Testing QuantumVirtualProcessor ===")
        TestQuantumVirtualProcessor.test_initialization()
        TestQuantumVirtualProcessor.test_hamiltonian_operations()
        TestQuantumVirtualProcessor.test_profile_switching()
        TestQuantumVirtualProcessor.test_ml_access()
        print("All QuantumVirtualProcessor tests passed!\n")


def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("HAMILTONIAN SUPPORT AND QUANTUM PROFILES TEST SUITE")
    print("=" * 60)
    
    try:
        TestHamiltonian.run_all()
        TestQuantumProfiles.run_all()
        TestMLQuantumAddon.run_all()
        TestProfileManager.run_all()
        TestQuantumVirtualProcessor.run_all()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
