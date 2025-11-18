# Quantum Profiles with ML Integration

## Overview

Pandora now includes integrated machine learning capabilities for all quantum profiles. Every quantum operation is automatically logged and can be used for on-the-fly incremental training of ML models.

## Features

- **Automatic ML Integration**: All quantum profiles include ML-based process logging by default
- **On-the-fly Training**: Incremental learning happens automatically as quantum operations occur
- **Multiple Profiles**: Four distinct quantum processor profiles with different computation philosophies
- **Flexible ML Models**: Works with scikit-learn models or custom pluggable models
- **Self-Instrumenting**: Profiles automatically monitor and learn from all operations

## Quantum Profiles

### 1. Alternative Quantum Virtual Processor
Explores alternative approaches to quantum computation with experimental algorithms.

```python
from quantum_profiles import AlternativeQuantumVirtualProcessor

processor = AlternativeQuantumVirtualProcessor(qubits=6)
processor.apply_gate("H", 0)
result = processor.measure()
```

### 2. Castle Quantum Virtual Processor
Focuses on defensive and secure quantum computation with robust error handling.

```python
from quantum_profiles import CastleQuantumVirtualProcessor

processor = CastleQuantumVirtualProcessor(qubits=6)
processor.apply_gate("X", 0)
result = processor.measure()
```

### 3. Hive Quantum Virtual Processor
Emphasizes collaborative and distributed quantum computation for parallel processing.

```python
from quantum_profiles import HiveQuantumVirtualProcessor

processor = HiveQuantumVirtualProcessor(qubits=6)
processor.apply_gate("H", 0)
result = processor.measure()
```

### 4. Empire Quantum Virtual Processor
High-performance quantum computation optimized for efficiency and speed.

```python
from quantum_profiles import EmpireQuantumVirtualProcessor

processor = EmpireQuantumVirtualProcessor(qubits=6)
processor.apply_gate("H", 0)
result = processor.measure()
```

## ML Quantum Addon

The `MLQuantumAddon` provides the machine learning capabilities integrated into all profiles.

### Basic Usage

```python
from quantum_profiles import MLQuantumAddon

# Create addon with default settings (train_on_fly=True)
addon = MLQuantumAddon()

# Use with quantum operations
addon.before_gate("H", 0, {"qubits": 4})
addon.after_gate("H", 0, {"qubits": 4})

# Access logged events
events = addon.get_events()
print(f"Logged {len(events)} events")

# Get statistics
stats = addon.get_statistics()
print(stats)
```

### With scikit-learn Models

```python
from quantum_profiles import AlternativeQuantumVirtualProcessor
from sklearn.linear_model import SGDClassifier

# Create processor
processor = AlternativeQuantumVirtualProcessor(qubits=4)

# Set custom sklearn model
model = SGDClassifier()
ml_addon = processor.get_ml_addon()
ml_addon.set_model(model)

# Perform operations - model trains automatically
for i in range(20):
    processor.apply_gate("H", i % 4)
    processor.measure()

# Check if model was trained
if hasattr(ml_addon.model, 'classes_'):
    print("Model trained on classes:", ml_addon.model.classes_)
```

### With Custom Models

Your custom model must implement the `partial_fit(X, y, classes=None)` method for incremental learning:

```python
class CustomModel:
    def __init__(self):
        self.patterns = []
        self.classes_ = None
    
    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
        # Your incremental learning logic here
        for features, label in zip(X, y):
            self.patterns.append({'features': features, 'label': label})

# Use with quantum processor
processor = HiveQuantumVirtualProcessor(qubits=4)
ml_addon = processor.get_ml_addon()
ml_addon.set_model(CustomModel())

# Operations will train your custom model
processor.apply_gate("H", 0)
processor.measure()
```

## Event Types Logged

The ML addon logs the following event types:

- `before_gate`: Called before applying a quantum gate
- `after_gate`: Called after applying a quantum gate
- `before_measurement`: Called before measuring qubits
- `after_measurement`: Called after measuring qubits
- `expansion`: Logged when the quantum system expands (adding qubits)
- `diagnostic`: Logged when diagnostic information is captured

## Accessing ML Data

### Get All Events
```python
ml_addon = processor.get_ml_addon()
events = ml_addon.get_events()
```

### Get Events by Type
```python
gate_events = ml_addon.get_events('before_gate')
measurement_events = ml_addon.get_events('after_measurement')
```

### Get Statistics
```python
stats = ml_addon.get_statistics()
# Returns:
# {
#     'total_events': 42,
#     'event_types': {'before_gate': 10, 'after_gate': 10, ...},
#     'buffer_size': 5,
#     'model_available': True,
#     'train_on_fly': True
# }
```

### Get Processor Statistics
```python
stats = processor.get_addon_statistics()
# Returns statistics for all addons in the processor
```

## Configuration Options

### Buffer Size
Controls how many events to buffer before training:

```python
addon = MLQuantumAddon(train_on_fly=True, buffer_size=20)
```

### Disable On-the-fly Training
```python
addon = MLQuantumAddon(train_on_fly=False)
# Events are still logged but model won't train automatically
```

### Logging Level
```python
import logging
addon = MLQuantumAddon(log_level=logging.DEBUG)
```

## Advanced Features

### System Expansion
```python
processor = EmpireQuantumVirtualProcessor(qubits=4)
processor.expand_qubits(2)  # Now has 6 qubits
# Expansion is automatically logged to ML addon
```

### Diagnostics
```python
diagnostics = processor.get_diagnostic_info()
# Returns diagnostic information and logs to ML addon
```

### Clear Event Log
```python
ml_addon = processor.get_ml_addon()
ml_addon.clear_events()  # Clears all logged events and buffers
```

## Architecture

### Profile Manager Compatibility

All quantum profiles are designed to work with Pandora's profile manager and registry system. Each profile:

1. Inherits from `QuantumVirtualProcessor`
2. Defines an `ADDONS` list with `MLQuantumAddon(train_on_fly=True)`
3. Implements hook methods (`_before_gate_hooks`, `_after_gate_hooks`, etc.)
4. Calls hooks before and after all quantum operations

### Hook System

The hook system ensures all operations are monitored:

```python
def apply_gate(self, gate, register):
    self._before_gate_hooks(gate, register)  # Notify addons
    super().apply_gate(gate, register)       # Perform operation
    self._after_gate_hooks(gate, register)   # Notify addons
```

## Performance Considerations

- **Buffer Size**: Larger buffers mean less frequent training but more memory usage
- **Model Complexity**: Complex models may slow down quantum operations
- **Event Logging**: All events are kept in memory; use `clear_events()` for long-running processes
- **On-the-fly Training**: Can be disabled with `train_on_fly=False` if needed

## Examples

See `examples_quantum_profiles.py` for comprehensive usage examples including:
- Basic usage
- scikit-learn integration
- Custom model integration
- All profile demonstrations
- System expansion and diagnostics
- Accessing ML learning data

## Testing

Run the test suite:
```bash
python3 test_quantum_profiles.py
```

## Requirements

- Python 3.7+
- numpy (required)
- scikit-learn (optional, for ML model integration)

Install requirements:
```bash
pip install numpy scikit-learn
```

## Future Enhancements

- Export/import trained models
- Real-time visualization of learning progress
- Distributed learning across multiple quantum processors
- Integration with quantum error correction
- Advanced feature extraction for different gate types
