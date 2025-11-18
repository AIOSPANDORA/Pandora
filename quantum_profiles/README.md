# Quantum Profiles

This directory contains various quantum processor profiles that Pandora can dynamically select and switch between.

## Available Profiles

### Empire Profile (`'empire'`)

The Empire Quantum Virtual Processor is a hierarchical quantum computing architecture featuring:

- **Central Control Lattice**: A QuantumVirtualProcessor supervised by Pandora (default: 4 qubits, configurable)
- **Kaleidoscopic Hives Grid**: An expandable 2D grid of HiveQuantumVirtualProcessors (default: 2x2 grid, 8 qubits per hive)

#### Features

1. **Dynamic Expansion**: Expand both the control lattice size and the hive grid dimensions at runtime
2. **Flexible Targeting**: Direct quantum operations to:
   - The entire Empire (control + all hives)
   - Just the control lattice
   - Any specific hive block
3. **Full Addon Support**: Register and execute addons to extend functionality
4. **Comprehensive API**: Clean, well-documented interface for all operations

#### Usage

```python
from quantum_profiles import get_profile

# Create Empire processor with default settings
empire = get_profile('empire')

# Create with custom configuration
empire = get_profile('empire', 
                    control_qubits=6, 
                    grid_size=(3, 3), 
                    hive_qubits=12)

# Apply gates to different targets
empire.apply_gate_to_empire("H", 0)           # Entire empire
empire.apply_gate_to_control_lattice("X", 1)  # Control only
empire.apply_gate_to_hive((0, 0), "Y", 2)     # Specific hive

# Perform measurements
results = empire.measure_empire()         # Measure everything
control = empire.measure_control_lattice() # Control only
hive = empire.measure_hive((1, 1))        # Specific hive

# Dynamic expansion
empire.expand_grid((4, 4))              # Expand hive grid
empire.expand_control_lattice(8)        # Expand control lattice

# Addon support
empire.register_addon(my_addon)
result = empire.execute_addon("addon_name")
```

#### Architecture Details

**Control Lattice**:
- Implemented as a `QuantumVirtualProcessor`
- Coordinates and supervises all hive operations
- Configurable qubit count (default: 4)
- Can be expanded dynamically

**Kaleidoscopic Hives**:
- Each hive is a `HiveQuantumVirtualProcessor` 
- Arranged in a 2D grid with (row, col) addressing
- Default configuration: 2x2 grid with 8 qubits per hive
- Grid can be expanded without recreating the Empire

**Total Qubits**: `control_qubits + (grid_rows × grid_cols × hive_qubits)`

#### API Reference

##### EmpireQuantumVirtualProcessor

**Constructor**:
```python
EmpireQuantumVirtualProcessor(control_qubits=4, grid_size=(2,2), hive_qubits=8)
```

**Key Methods**:
- `expand_grid(new_grid_size)` - Expand the hive grid
- `expand_control_lattice(new_qubit_count)` - Expand control lattice
- `apply_gate_to_empire(gate, reg)` - Apply gate to entire empire
- `apply_gate_to_control_lattice(gate, reg)` - Apply gate to control only
- `apply_gate_to_hive(hive_id, gate, reg)` - Apply gate to specific hive
- `measure_empire()` - Measure entire empire
- `measure_control_lattice()` - Measure control only
- `measure_hive(hive_id)` - Measure specific hive
- `get_hive(hive_id)` - Get reference to specific hive
- `list_hives()` - List all hive IDs
- `register_addon(addon)` - Register an addon
- `execute_addon(addon_name, *args, **kwargs)` - Execute addon
- `get_empire_stats()` - Get Empire statistics

##### HiveQuantumVirtualProcessor

**Constructor**:
```python
HiveQuantumVirtualProcessor(qubits=8, hive_id=None)
```

Inherits all methods from `QuantumVirtualProcessor` and adds hive-specific identification.

## Adding New Profiles

To add a new quantum processor profile:

1. Create a new Python file in this directory (e.g., `myprofile_quantum_virtual_processor.py`)
2. Implement your processor class
3. Register it in `__init__.py` by adding to `QUANTUM_PROFILES` dictionary
4. Update this README with documentation

Example:
```python
# In __init__.py
from quantum_profiles.myprofile_quantum_virtual_processor import MyProfileProcessor

QUANTUM_PROFILES = {
    'empire': EmpireQuantumVirtualProcessor,
    'myprofile': MyProfileProcessor,
}
```

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
python3 test_empire_quantum_processor.py
```

Run the demonstration:
```bash
python3 demo_empire_processor.py
```

## Compatibility

All quantum processor profiles should be compatible with:
- The base `QuantumVirtualProcessor` interface
- Pandora's addon system
- Dynamic configuration and runtime modification
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
