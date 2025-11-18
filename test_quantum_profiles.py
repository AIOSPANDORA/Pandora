#!/usr/bin/env python3
"""
Test script for Quantum Profiles with ML Integration

This script tests all quantum profiles and ML addon functionality.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_profiles import (
    MLQuantumAddon,
    AlternativeQuantumVirtualProcessor,
    CastleQuantumVirtualProcessor,
    HiveQuantumVirtualProcessor,
    EmpireQuantumVirtualProcessor
)


def test_ml_quantum_addon():
    """Test MLQuantumAddon basic functionality."""
    print("\n=== Testing MLQuantumAddon ===")
    
    # Test initialization
    addon = MLQuantumAddon(train_on_fly=True)
    assert addon.train_on_fly == True
    assert addon.get_event_count() == 0
    print("✓ MLQuantumAddon initialized correctly")
    
    # Test event logging
    context = {'qubits': 4}
    addon.before_gate("H", 0, context)
    addon.after_gate("H", 0, context)
    assert addon.get_event_count() == 2
    print("✓ Event logging works")
    
    # Test event retrieval
    events = addon.get_events('before_gate')
    assert len(events) == 1
    assert events[0]['operation'] == 'H'
    print("✓ Event retrieval works")
    
    # Test statistics
    stats = addon.get_statistics()
    assert stats['total_events'] == 2
    assert 'before_gate' in stats['event_types']
    print("✓ Statistics generation works")
    
    # Test measurement hooks
    addon.before_measurement(0, context)
    addon.after_measurement(0, 1, context)
    assert addon.get_event_count() == 4
    print("✓ Measurement hooks work")
    
    # Test expansion logging
    addon.log_expansion("Test expansion", context)
    assert addon.get_event_count() == 5
    print("✓ Expansion logging works")
    
    # Test diagnostic logging
    addon.log_diagnostic("test_diagnostic", {"data": "test"}, context)
    assert addon.get_event_count() == 6
    print("✓ Diagnostic logging works")
    
    print("✓ All MLQuantumAddon tests passed!")
    return True


def test_quantum_profile(profile_class, profile_name):
    """Test a quantum profile."""
    print(f"\n=== Testing {profile_name} ===")
    
    # Test initialization
    processor = profile_class(qubits=6)
    assert processor.qubits == 6
    assert len(processor.addons) > 0
    print(f"✓ {profile_name} initialized correctly")
    
    # Test that ML addon is present
    ml_addon = processor.get_ml_addon()
    assert ml_addon is not None
    assert isinstance(ml_addon, MLQuantumAddon)
    print(f"✓ ML addon is present and correct type")
    
    # Test gate application
    initial_count = ml_addon.get_event_count()
    processor.apply_gate("H", 0)
    assert ml_addon.get_event_count() > initial_count
    print(f"✓ Gate application triggers ML logging")
    
    # Test measurement
    initial_count = ml_addon.get_event_count()
    result = processor.measure()
    assert ml_addon.get_event_count() > initial_count
    print(f"✓ Measurement triggers ML logging")
    
    # Test qubit expansion
    initial_count = ml_addon.get_event_count()
    processor.expand_qubits(2)
    assert processor.qubits == 8
    assert ml_addon.get_event_count() > initial_count
    print(f"✓ Qubit expansion works and triggers ML logging")
    
    # Test diagnostics
    initial_count = ml_addon.get_event_count()
    diagnostic = processor.get_diagnostic_info()
    assert diagnostic['qubits'] == 8
    assert diagnostic['profile'] == profile_name
    assert ml_addon.get_event_count() > initial_count
    print(f"✓ Diagnostics work and trigger ML logging")
    
    # Test statistics
    stats = processor.get_addon_statistics()
    assert 'addon_0' in stats
    assert stats['addon_0']['total_events'] > 0
    print(f"✓ Statistics retrieval works")
    
    print(f"✓ All {profile_name} tests passed!")
    return True


def test_ml_with_sklearn():
    """Test ML addon with scikit-learn model."""
    print("\n=== Testing ML Integration with scikit-learn ===")
    
    try:
        from sklearn.linear_model import SGDClassifier
        
        # Create addon with sklearn model
        model = SGDClassifier()
        addon = MLQuantumAddon(model=model, train_on_fly=True, buffer_size=5)
        print("✓ MLQuantumAddon created with SGDClassifier")
        
        # Simulate enough events to trigger training
        context = {'qubits': 4}
        for i in range(15):
            addon.before_gate("H", i % 4, context)
            addon.after_gate("H", i % 4, context)
        
        # Check that training occurred
        assert hasattr(addon.model, 'classes_'), "Model should have been trained"
        print("✓ On-the-fly training with sklearn works")
        
        print("✓ All scikit-learn integration tests passed!")
        return True
        
    except ImportError:
        print("⚠ scikit-learn not installed, skipping sklearn integration tests")
        return True


def test_custom_model():
    """Test ML addon with custom model."""
    print("\n=== Testing ML Integration with Custom Model ===")
    
    class CustomModel:
        """Simple custom model for testing."""
        def __init__(self):
            self.train_count = 0
            self.classes_ = None
        
        def partial_fit(self, X, y, classes=None):
            self.train_count += 1
            if classes is not None:
                self.classes_ = classes
    
    # Create addon with custom model
    model = CustomModel()
    addon = MLQuantumAddon(model=model, train_on_fly=True, buffer_size=3)
    print("✓ MLQuantumAddon created with custom model")
    
    # Simulate events to trigger training
    context = {'qubits': 4}
    for i in range(10):
        addon.before_gate("H", i % 4, context)
    
    # Check that training occurred
    assert model.train_count > 0, "Custom model should have been trained"
    print(f"✓ Custom model was trained {model.train_count} times")
    
    print("✓ All custom model integration tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Quantum Profiles with ML Integration - Test Suite")
    print("=" * 60)
    
    try:
        # Test ML addon
        test_ml_quantum_addon()
        
        # Test all quantum profiles
        test_quantum_profile(AlternativeQuantumVirtualProcessor, "Alternative")
        test_quantum_profile(CastleQuantumVirtualProcessor, "Castle")
        test_quantum_profile(HiveQuantumVirtualProcessor, "Hive")
        test_quantum_profile(EmpireQuantumVirtualProcessor, "Empire")
        
        # Test ML integration
        test_ml_with_sklearn()
        test_custom_model()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
