"""
Tests for Empire Quantum Virtual Processor
===========================================

Test suite for validating the Empire quantum processor profile functionality.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_profiles import (
    EmpireQuantumVirtualProcessor,
    HiveQuantumVirtualProcessor,
    get_profile,
    list_profiles,
    QUANTUM_PROFILES
)


def test_hive_initialization():
    """Test HiveQuantumVirtualProcessor initialization."""
    print("Testing HiveQuantumVirtualProcessor initialization...")
    
    # Test default initialization
    hive = HiveQuantumVirtualProcessor()
    assert hive.hive_qubits == 8, "Default hive should have 8 qubits"
    assert hive.qubits == 8, "Inherited qubits property should be 8"
    assert hive.hive_id is None, "Default hive_id should be None"
    
    # Test with custom parameters
    hive_custom = HiveQuantumVirtualProcessor(qubits=16, hive_id=(1, 2))
    assert hive_custom.hive_qubits == 16, "Custom hive should have 16 qubits"
    assert hive_custom.hive_id == (1, 2), "Hive ID should be (1, 2)"
    
    print("✓ HiveQuantumVirtualProcessor initialization tests passed")


def test_empire_initialization():
    """Test EmpireQuantumVirtualProcessor initialization."""
    print("Testing EmpireQuantumVirtualProcessor initialization...")
    
    # Test default initialization
    empire = EmpireQuantumVirtualProcessor()
    assert empire.control_qubits == 4, "Default control lattice should have 4 qubits"
    assert empire.grid_size == (2, 2), "Default grid should be 2x2"
    assert empire.hive_qubits == 8, "Default hive qubits should be 8"
    assert len(empire.hive_grid) == 4, "Default grid should have 4 hives"
    
    # Test custom initialization
    empire_custom = EmpireQuantumVirtualProcessor(
        control_qubits=6,
        grid_size=(3, 3),
        hive_qubits=12
    )
    assert empire_custom.control_qubits == 6, "Custom control lattice should have 6 qubits"
    assert empire_custom.grid_size == (3, 3), "Custom grid should be 3x3"
    assert len(empire_custom.hive_grid) == 9, "3x3 grid should have 9 hives"
    
    print("✓ EmpireQuantumVirtualProcessor initialization tests passed")


def test_hive_grid_structure():
    """Test hive grid structure and access."""
    print("Testing hive grid structure...")
    
    empire = EmpireQuantumVirtualProcessor(grid_size=(2, 3))
    
    # Check all expected hives exist
    expected_hives = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for hive_id in expected_hives:
        assert hive_id in empire.hive_grid, f"Hive {hive_id} should exist"
        hive = empire.hive_grid[hive_id]
        assert isinstance(hive, HiveQuantumVirtualProcessor), "Should be HiveQuantumVirtualProcessor"
        assert hive.hive_id == hive_id, f"Hive should have correct ID {hive_id}"
    
    # Test list_hives
    hive_list = empire.list_hives()
    assert hive_list == expected_hives, "list_hives should return sorted hive IDs"
    
    print("✓ Hive grid structure tests passed")


def test_grid_expansion():
    """Test dynamic grid expansion."""
    print("Testing grid expansion...")
    
    empire = EmpireQuantumVirtualProcessor(grid_size=(2, 2))
    initial_hives = len(empire.hive_grid)
    assert initial_hives == 4, "Should start with 4 hives"
    
    # Expand grid
    empire.expand_grid((3, 4))
    assert empire.grid_size == (3, 4), "Grid size should be updated to (3, 4)"
    assert len(empire.hive_grid) == 12, "Should have 12 hives after expansion"
    
    # Verify all hives exist
    for row in range(3):
        for col in range(4):
            assert (row, col) in empire.hive_grid, f"Hive ({row}, {col}) should exist"
    
    # Test that we can't shrink
    try:
        empire.expand_grid((2, 2))
        assert False, "Should not allow grid shrinking"
    except ValueError as e:
        assert "Cannot shrink" in str(e), "Should raise appropriate error"
    
    print("✓ Grid expansion tests passed")


def test_control_lattice_expansion():
    """Test control lattice expansion."""
    print("Testing control lattice expansion...")
    
    empire = EmpireQuantumVirtualProcessor(control_qubits=4)
    assert empire.control_qubits == 4, "Should start with 4 control qubits"
    
    # Expand control lattice
    empire.expand_control_lattice(8)
    assert empire.control_qubits == 8, "Should have 8 control qubits after expansion"
    
    # Test that we can't reduce
    try:
        empire.expand_control_lattice(4)
        assert False, "Should not allow control lattice reduction"
    except ValueError as e:
        assert "Cannot reduce" in str(e), "Should raise appropriate error"
    
    print("✓ Control lattice expansion tests passed")


def test_gate_application():
    """Test applying gates to different targets."""
    print("Testing gate application...")
    
    empire = EmpireQuantumVirtualProcessor(grid_size=(2, 2))
    
    # Test applying gate to empire (should not raise errors)
    empire.apply_gate_to_empire("H", 0)
    
    # Test applying gate to control lattice
    empire.apply_gate_to_control_lattice("H", 0)
    
    # Test applying gate to specific hive
    empire.apply_gate_to_hive((0, 0), "H", 0)
    empire.apply_gate_to_hive((1, 1), "X", 1)
    
    # Test invalid hive
    try:
        empire.apply_gate_to_hive((5, 5), "H", 0)
        assert False, "Should raise error for non-existent hive"
    except KeyError as e:
        assert "does not exist" in str(e), "Should raise appropriate error"
    
    print("✓ Gate application tests passed")


def test_measurement():
    """Test measurement operations."""
    print("Testing measurement operations...")
    
    empire = EmpireQuantumVirtualProcessor(grid_size=(2, 2))
    
    # Test measuring entire empire
    results = empire.measure_empire()
    assert 'control' in results, "Should have control measurement"
    assert 'hives' in results, "Should have hives measurements"
    assert len(results['hives']) == 4, "Should have measurements for all 4 hives"
    
    # Test measuring control lattice (may return None from base class stub)
    control_result = empire.measure_control_lattice()
    # Just verify the method can be called without error
    
    # Test measuring specific hive (may return None from base class stub)
    empire.measure_hive((0, 0))
    # Just verify the method can be called without error
    
    # Test invalid hive
    try:
        empire.measure_hive((10, 10))
        assert False, "Should raise error for non-existent hive"
    except KeyError:
        pass  # Expected
    
    print("✓ Measurement tests passed")


def test_hive_access():
    """Test getting hive references."""
    print("Testing hive access...")
    
    empire = EmpireQuantumVirtualProcessor(grid_size=(2, 2))
    
    # Test getting valid hive
    hive = empire.get_hive((0, 0))
    assert isinstance(hive, HiveQuantumVirtualProcessor), "Should return HiveQuantumVirtualProcessor"
    assert hive.hive_id == (0, 0), "Should have correct hive_id"
    
    # Test invalid hive
    try:
        empire.get_hive((99, 99))
        assert False, "Should raise error for non-existent hive"
    except KeyError:
        pass  # Expected
    
    print("✓ Hive access tests passed")


def test_addon_support():
    """Test addon registration and execution."""
    print("Testing addon support...")
    
    empire = EmpireQuantumVirtualProcessor()
    
    # Create mock addon
    class MockAddon:
        def __init__(self):
            self.name = "test_addon"
            self.executed = False
        
        def execute(self, empire, *args, **kwargs):
            self.executed = True
            return "addon_result"
    
    addon = MockAddon()
    
    # Register addon
    empire.register_addon(addon)
    assert len(empire.addons) == 1, "Should have 1 registered addon"
    
    # Execute addon
    result = empire.execute_addon("test_addon")
    assert result == "addon_result", "Should return addon result"
    assert addon.executed, "Addon should be executed"
    
    # Test non-existent addon
    try:
        empire.execute_addon("non_existent")
        assert False, "Should raise error for non-existent addon"
    except ValueError as e:
        assert "not found" in str(e), "Should raise appropriate error"
    
    print("✓ Addon support tests passed")


def test_empire_stats():
    """Test getting Empire statistics."""
    print("Testing Empire statistics...")
    
    empire = EmpireQuantumVirtualProcessor(
        control_qubits=6,
        grid_size=(3, 3),
        hive_qubits=10
    )
    
    stats = empire.get_empire_stats()
    
    assert stats['control_lattice_qubits'] == 6, "Should report correct control qubits"
    assert stats['grid_size'] == (3, 3), "Should report correct grid size"
    assert stats['total_hives'] == 9, "Should report correct total hives"
    assert stats['hive_qubits'] == 10, "Should report correct hive qubits"
    assert stats['total_qubits'] == 6 + (9 * 10), "Should calculate total qubits correctly"
    assert stats['registered_addons'] == 0, "Should have no addons initially"
    
    print("✓ Empire statistics tests passed")


def test_profile_registry():
    """Test profile registration and retrieval."""
    print("Testing profile registry...")
    
    # Test that empire is registered
    assert 'empire' in QUANTUM_PROFILES, "Empire should be registered"
    
    # Test list_profiles
    profiles = list_profiles()
    assert 'empire' in profiles, "Empire should be in profile list"
    
    # Test get_profile
    empire = get_profile('empire')
    assert isinstance(empire, EmpireQuantumVirtualProcessor), "Should return Empire instance"
    
    # Test get_profile with parameters
    empire_custom = get_profile('empire', control_qubits=8, grid_size=(4, 4))
    assert empire_custom.control_qubits == 8, "Should respect custom parameters"
    assert empire_custom.grid_size == (4, 4), "Should respect custom grid size"
    
    # Test non-existent profile
    try:
        get_profile('non_existent')
        assert False, "Should raise error for non-existent profile"
    except KeyError as e:
        assert "not found" in str(e), "Should raise appropriate error"
        assert "Available profiles" in str(e), "Should list available profiles"
    
    print("✓ Profile registry tests passed")


def test_string_representations():
    """Test __repr__ methods."""
    print("Testing string representations...")
    
    # Test HiveQuantumVirtualProcessor repr
    hive = HiveQuantumVirtualProcessor(qubits=8, hive_id=(1, 2))
    repr_str = repr(hive)
    assert "HiveQuantumVirtualProcessor" in repr_str, "Should include class name"
    assert "(1, 2)" in repr_str, "Should include hive ID"
    assert "8 qubits" in repr_str, "Should include qubit count"
    
    # Test EmpireQuantumVirtualProcessor repr
    empire = EmpireQuantumVirtualProcessor(control_qubits=4, grid_size=(2, 2))
    repr_str = repr(empire)
    assert "EmpireQuantumVirtualProcessor" in repr_str, "Should include class name"
    assert "4 control qubits" in repr_str, "Should include control qubits"
    assert "4 hives" in repr_str, "Should include hive count"
    
    print("✓ String representation tests passed")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("Running Empire Quantum Virtual Processor Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_hive_initialization,
        test_empire_initialization,
        test_hive_grid_structure,
        test_grid_expansion,
        test_control_lattice_expansion,
        test_gate_application,
        test_measurement,
        test_hive_access,
        test_addon_support,
        test_empire_stats,
        test_profile_registry,
        test_string_representations
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
