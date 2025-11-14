# Quantum Profiles Directory

This directory provides an extensible framework for quantum processor profiles and add-ons in Pandora. You can easily add new quantum processor implementations, customize existing ones, or create add-ons to extend functionality.

## Overview

The quantum_profiles directory is an open folder designed for:
- **Extensible Quantum Processor Profiles**: Different quantum processor implementations with varying capabilities
- **Add-ons System**: Modular extensions that can be attached to any processor
- **Dynamic Discovery**: Pandora automatically discovers and registers profiles placed in this directory
- **Drag-and-Drop Extensibility**: Simply add new Python files to extend Pandora's quantum capabilities

## Available Profiles

### 1. Alternative Quantum Virtual Processor (`alternative`)

A flexible quantum processor with comprehensive add-on support, diagnostics, and fallback mechanisms.

**Features:**
- 6 qubits by default (configurable)
- Add-on support for extensibility
- Built-in diagnostics and health checks
- Fallback mechanisms for stability
- Logger add-on included

**Usage:**
```python
from quantum_profiles import get_profile

# Get the alternative processor
processor = get_profile('alternative')

# Apply quantum gates
processor.apply_gate('H', 0)  # Hadamard on qubit 0
processor.apply_gate('X', 1)  # Pauli-X on qubit 1

# Measure
result = processor.measure()
print(f"Measurement: {result}")

# Add custom add-ons
from quantum_profiles.alternative_quantum_virtual_processor import QuantumAddon

class MyAddon(QuantumAddon):
    def on_gate_apply(self, gate, reg):
        print(f"Custom addon: Gate {gate} applied to {reg}")

processor.add_addon(MyAddon())
```

### 2. Four Overlay Quantum Virtual Processor (`four_overlay`)

A multi-buffer quantum processor that maintains 4 independent quantum overlays (4 × 25 qubits = 100 qubits total).

**Features:**
- 4 overlays with 25 qubits each by default
- Dynamic overlay management (add/remove overlays)
- Customizable qubit count per overlay
- Parallel quantum operations across overlays
- Add-on support

**Usage:**
```python
from quantum_profiles import get_profile

# Get the four overlay processor
processor = get_profile('four_overlay')

# Work with the current overlay
processor.apply_gate('H', 0)
result = processor.measure(0)

# Switch to a different overlay
processor.select_overlay(1)
processor.apply_gate('X', 5)

# Add a new overlay with custom qubit count
new_overlay_id = processor.add_overlay(qubit_count=50)

# Measure across all overlays
all_results = processor.measure_all_overlays(reg=0)

# Get state info
info = processor.get_state_info()
print(f"Total qubits: {processor.get_total_qubits()}")
```

## Registry Functions

The `quantum_profiles` module provides several functions for managing profiles:

```python
from quantum_profiles import list_profiles, get_profile, get_addons

# List all available profiles
profiles = list_profiles()
print(f"Available profiles: {profiles}")

# Get a specific profile
processor = get_profile('alternative')

# Get addons for a profile
addons = get_addons('alternative')
print(f"Available addons: {addons}")
```

## Creating Custom Profiles

You can easily create your own quantum processor profiles by following these steps:

1. **Create a new Python file** in the `quantum_profiles` directory (e.g., `my_custom_processor.py`)

2. **Implement your processor class** with the required methods:
   ```python
   class MyCustomProcessor:
       def __init__(self, qubits=10):
           self.qubits = qubits
           # Initialize your processor
       
       def apply_gate(self, gate, reg):
           # Implement gate application
           pass
       
       def measure(self, reg=None):
           # Implement measurement
           pass
   ```

3. **Add a `get_profile()` factory function**:
   ```python
   def get_profile():
       """Factory function for Pandora to instantiate the processor."""
       return MyCustomProcessor(qubits=10)
   ```

4. **Register your profile** in `__init__.py`:
   ```python
   from . import my_custom_processor
   register_profile(
       'my_custom',
       'quantum_profiles.my_custom_processor',
       my_custom_processor.get_profile
   )
   ```

5. **Use your profile** in Pandora:
   ```python
   from quantum_profiles import get_profile
   processor = get_profile('my_custom')
   ```

## Creating Custom Add-ons

Add-ons extend processor functionality without modifying the core implementation:

```python
from quantum_profiles.alternative_quantum_virtual_processor import QuantumAddon

class PerformanceMonitorAddon(QuantumAddon):
    """Add-on that tracks performance metrics."""
    
    def __init__(self):
        super().__init__(name="PerformanceMonitor")
        self.gate_count = {}
    
    def on_gate_apply(self, gate, reg):
        self.gate_count[gate] = self.gate_count.get(gate, 0) + 1
    
    def get_stats(self):
        return self.gate_count

# Use the add-on
processor = get_profile('alternative')
monitor = PerformanceMonitorAddon()
processor.add_addon(monitor)

# Perform operations...
processor.apply_gate('H', 0)
processor.apply_gate('X', 1)
processor.apply_gate('H', 2)

# Get stats
print(monitor.get_stats())  # {'H': 2, 'X': 1}
```

## Architecture & Design Philosophy

### Extensibility
- **Open Folder Structure**: Simply drop new files to extend functionality
- **Plugin Architecture**: Add-ons can be developed independently and attached at runtime
- **Factory Pattern**: Profile factory functions enable clean instantiation

### Backwards Compatibility
- All profiles maintain compatibility with the original `QuantumVirtualProcessor` interface
- Existing code using `apply_gate()` and `measure()` works without changes
- Add-ons are optional and don't affect core functionality

### Uncertainty & Harmony
- Quantum processors accept uncertainty as a fundamental principle
- Classical and quantum operations work in harmony
- Fallback mechanisms ensure stability even with quantum uncertainty

## File Structure

```
quantum_profiles/
├── __init__.py                                 # Registry and factory functions
├── alternative_quantum_virtual_processor.py    # Alternative processor with add-ons
├── four_overlay_quantum_virtual_processor.py   # Multi-overlay processor
├── README.md                                   # This file
└── [your_custom_processor.py]                  # Your custom profiles (drag & drop)
```

## Drag-and-Drop Usage

1. Create your processor file anywhere
2. Drag and drop it into the `quantum_profiles` directory
3. Update `__init__.py` to register your profile
4. Start using it immediately with `get_profile('your_profile_name')`

## Best Practices

1. **Always provide a `get_profile()` function** - This is how Pandora discovers your profile
2. **Include docstrings** - Document your processor's capabilities and usage
3. **Support add-ons when possible** - Makes your processor more extensible
4. **Include fallback mechanisms** - Ensure stability under uncertainty
5. **Test independently** - Each profile should work standalone

## Support & Contributions

To add new profiles or improve existing ones:
1. Follow the structure of existing profiles
2. Ensure backwards compatibility with base interface
3. Document your additions thoroughly
4. Test with various add-on combinations

## Examples

### Example 1: Hybrid Classical-Quantum Computation
```python
processor = get_profile('alternative')

# Quantum preparation
processor.apply_gate('H', 0)
processor.apply_gate('H', 1)

# Classical processing
result = processor.measure([0, 1])
classical_value = sum(result)

# More quantum operations based on classical result
if classical_value > 0:
    processor.apply_gate('X', 2)

final = processor.measure()
```

### Example 2: Parallel Overlay Processing
```python
processor = get_profile('four_overlay')

# Prepare different states in each overlay
for overlay_id in range(4):
    processor.select_overlay(overlay_id)
    processor.apply_gate('H', 0)
    processor.apply_gate('X', overlay_id)

# Measure all overlays simultaneously
results = processor.measure_all_overlays()
print(f"Parallel results: {results}")
```

---

**Note**: This is an extensible framework - feel free to experiment, create, and share your quantum processor profiles!
