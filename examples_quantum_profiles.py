#!/usr/bin/env python3
"""
Quantum Profiles with ML Integration - Usage Examples

This script demonstrates how to use the quantum profiles with ML integration.
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


def example_basic_usage():
    """Example 1: Basic usage of quantum profiles."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a quantum processor with ML integration (default)
    processor = AlternativeQuantumVirtualProcessor(qubits=4)
    print(f"Created Alternative processor with {processor.qubits} qubits")
    
    # Apply some quantum gates
    processor.apply_gate("H", 0)  # Hadamard gate on qubit 0
    processor.apply_gate("X", 1)  # Pauli-X gate on qubit 1
    processor.apply_gate("H", 2)  # Hadamard gate on qubit 2
    print("Applied gates: H on qubit 0, X on qubit 1, H on qubit 2")
    
    # Measure
    result = processor.measure()
    print(f"Measurement result: {result}")
    
    # Check ML addon statistics
    stats = processor.get_addon_statistics()
    print(f"\nML Statistics:")
    print(f"  Total events logged: {stats['addon_0']['total_events']}")
    print(f"  Event types: {stats['addon_0']['event_types']}")


def example_with_sklearn():
    """Example 2: Using ML addon with scikit-learn."""
    print("\n" + "=" * 60)
    print("Example 2: Integration with scikit-learn")
    print("=" * 60)
    
    try:
        from sklearn.linear_model import SGDClassifier
        
        # Create a processor with custom sklearn model
        model = SGDClassifier()
        
        # Create processor (the ML addon in ADDONS will use default model)
        processor = CastleQuantumVirtualProcessor(qubits=6)
        
        # Replace the model in the ML addon
        ml_addon = processor.get_ml_addon()
        ml_addon.set_model(model)
        print("Created Castle processor with SGDClassifier")
        
        # Run some quantum operations
        for i in range(20):
            processor.apply_gate("H", i % 6)
            if i % 5 == 0:
                processor.measure()
        
        print(f"Performed 20 gate operations and 4 measurements")
        
        # Check if model was trained
        if hasattr(ml_addon.model, 'classes_'):
            print(f"✓ Model successfully trained on-the-fly")
            print(f"  Classes learned: {ml_addon.model.classes_}")
        
        stats = ml_addon.get_statistics()
        print(f"\nML Statistics:")
        print(f"  Total events: {stats['total_events']}")
        print(f"  Training enabled: {stats['train_on_fly']}")
        
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")


def example_custom_model():
    """Example 3: Using custom ML model."""
    print("\n" + "=" * 60)
    print("Example 3: Custom ML Model")
    print("=" * 60)
    
    class SimplePatternLearner:
        """Simple custom model that learns patterns in quantum operations."""
        
        def __init__(self):
            self.patterns = []
            self.classes_ = None
        
        def partial_fit(self, X, y, classes=None):
            """Learn from new data incrementally."""
            if classes is not None:
                self.classes_ = classes
            
            for features, label in zip(X, y):
                self.patterns.append({
                    'features': features,
                    'label': label
                })
            
            print(f"  Learned {len(X)} new patterns (total: {len(self.patterns)})")
    
    # Create processor with custom model
    model = SimplePatternLearner()
    processor = HiveQuantumVirtualProcessor(qubits=4)
    
    # Set custom model
    ml_addon = processor.get_ml_addon()
    ml_addon.set_model(model)
    print("Created Hive processor with custom pattern learner")
    
    # Perform operations
    print("\nPerforming quantum operations:")
    for i in range(15):
        processor.apply_gate("H", i % 4)
    
    print(f"\nTotal patterns learned: {len(model.patterns)}")


def example_all_profiles():
    """Example 4: Demonstrate all four quantum profiles."""
    print("\n" + "=" * 60)
    print("Example 4: All Quantum Profiles")
    print("=" * 60)
    
    profiles = [
        (AlternativeQuantumVirtualProcessor, "Alternative", 
         "Explores alternative quantum computation approaches"),
        (CastleQuantumVirtualProcessor, "Castle", 
         "Defensive and secure quantum computation"),
        (HiveQuantumVirtualProcessor, "Hive", 
         "Collaborative and distributed quantum computation"),
        (EmpireQuantumVirtualProcessor, "Empire", 
         "High-performance quantum computation")
    ]
    
    for ProfileClass, name, description in profiles:
        print(f"\n{name} Profile:")
        print(f"  {description}")
        
        # Create processor
        processor = ProfileClass(qubits=4)
        
        # Run some operations
        for i in range(5):
            processor.apply_gate("H", i % 4)
        processor.measure()
        
        # Get statistics
        ml_addon = processor.get_ml_addon()
        stats = ml_addon.get_statistics()
        print(f"  Operations logged: {stats['total_events']}")
        print(f"  ML training active: {stats['train_on_fly']}")


def example_expansion_and_diagnostics():
    """Example 5: Quantum system expansion and diagnostics."""
    print("\n" + "=" * 60)
    print("Example 5: System Expansion and Diagnostics")
    print("=" * 60)
    
    processor = EmpireQuantumVirtualProcessor(qubits=4)
    print(f"Initial system: {processor.qubits} qubits")
    
    # Perform some operations
    processor.apply_gate("H", 0)
    processor.apply_gate("X", 1)
    
    # Expand the system
    processor.expand_qubits(2)
    print(f"After expansion: {processor.qubits} qubits")
    
    # Get diagnostics
    diagnostics = processor.get_diagnostic_info()
    print(f"\nDiagnostics:")
    print(f"  Profile: {diagnostics['profile']}")
    print(f"  Qubits: {diagnostics['qubits']}")
    print(f"  Active addons: {diagnostics['addons']}")
    
    # Check ML addon events
    ml_addon = processor.get_ml_addon()
    events = ml_addon.get_events()
    
    print(f"\nEvent log summary:")
    print(f"  Total events: {len(events)}")
    
    # Count event types
    event_types = {}
    for event in events:
        et = event['event_type']
        event_types[et] = event_types.get(et, 0) + 1
    
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count}")


def example_accessing_ml_data():
    """Example 6: Accessing ML learning data."""
    print("\n" + "=" * 60)
    print("Example 6: Accessing ML Learning Data")
    print("=" * 60)
    
    processor = AlternativeQuantumVirtualProcessor(qubits=4)
    ml_addon = processor.get_ml_addon()
    
    # Perform various operations
    operations = [
        ("H", 0), ("X", 1), ("H", 2), ("X", 3),
        ("H", 0), ("X", 1), ("H", 2)
    ]
    
    print("Performing operations:")
    for gate, qubit in operations:
        processor.apply_gate(gate, qubit)
        print(f"  Applied {gate} to qubit {qubit}")
    
    # Measure
    processor.measure()
    print("  Performed measurement")
    
    # Access logged events
    print(f"\nLogged events:")
    events = ml_addon.get_events()
    print(f"  Total: {len(events)}")
    
    # Show first few events
    print(f"\nFirst 3 events:")
    for i, event in enumerate(events[:3]):
        print(f"  {i+1}. {event['event_type']}: {event['operation']} on register {event['register']}")
    
    # Get events by type
    gate_events = ml_addon.get_events('before_gate')
    measurement_events = ml_addon.get_events('before_measurement')
    print(f"\nFiltered events:")
    print(f"  Gate operations: {len(gate_events)}")
    print(f"  Measurements: {len(measurement_events)}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Quantum Profiles with ML Integration")
    print("Usage Examples and Demonstrations")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_with_sklearn()
        example_custom_model()
        example_all_profiles()
        example_expansion_and_diagnostics()
        example_accessing_ml_data()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
