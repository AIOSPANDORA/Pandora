# Quantum Profiles with Hamiltonian Support

This module provides comprehensive Hamiltonian support and deep Hamiltonian machine learning capabilities for Pandora's quantum interface.

## Overview

The quantum profiles system integrates advanced Hamiltonian mechanics into quantum processor simulations, enabling:

- **Hamiltonian Construction**: Build quantum Hamiltonians from weighted Pauli operators
- **Time Evolution**: Evolve quantum states under Hamiltonian dynamics
- **Energy Calculations**: Compute expectation values and ground states
- **ML Integration**: Learn and optimize Hamiltonian parameters
- **Profile Management**: Switch between specialized quantum processor profiles

## Architecture

```
quantum_profiles/
├── __init__.py              # Module exports
├── hamiltonian.py           # Core Hamiltonian class
├── base_profile.py          # Base class for all profiles
├── alternative.py           # Alternative profile
├── castle.py                # Castle profile
├── hive.py                  # Hive profile
├── empire.py                # Empire profile
├── omega.py                 # Omega profile
├── ml_quantum_addon.py      # ML learning capabilities
└── profile_manager.py       # Unified profile management
```

## Quantum Profiles

### 1. Alternative Profile
**Philosophy**: Explores alternative computational pathways through superposition

**Characteristics**:
- Strong transverse X fields on all qubits
- Y terms for quantum phase exploration
- Weak nearest-neighbor Z-Z coupling

**Use Cases**: Quantum search, optimization via superposition

### 2. Castle Profile
**Philosophy**: Defensive and stable quantum operations

**Characteristics**:
- Strong longitudinal Z fields creating stable computational basis
- Z-Z nearest-neighbor interactions forming energy barriers
- Weak X perturbations for controlled transitions

**Use Cases**: Error-resistant quantum memory, stable quantum states

### 3. Hive Profile
**Philosophy**: Collective behavior through interconnectedness

**Characteristics**:
- All-to-all Z-Z interactions for collective coupling
- Uniform X fields for collective superposition
- Global Y terms for phase coherence

**Use Cases**: Quantum consensus, distributed quantum computing

### 4. Empire Profile
**Philosophy**: Hierarchical control with cascading effects

**Characteristics**:
- Central control qubits with strong Z fields
- Hierarchical X-X coupling radiating from center with distance-dependent strength
- Peripheral Z stabilization

**Use Cases**: Centralized quantum control, hierarchical quantum algorithms

### 5. Omega Profile
**Philosophy**: Ultimate balanced configuration

**Characteristics**:
- Balanced single-qubit X, Y, Z terms
- Mixed two-qubit XX, YY, ZZ interactions
- Three-body terms for higher-order correlations

**Use Cases**: General-purpose quantum computation, comprehensive quantum simulation

## Core Components

### Hamiltonian Class

The `Hamiltonian` class provides robust quantum Hamiltonian operations:

```python
from quantum_profiles import Hamiltonian

# Create a Hamiltonian for 3 qubits
h = Hamiltonian(n_qubits=3)

# Add weighted Pauli terms
h.add_term(1.0, "ZII")   # Z on qubit 0
h.add_term(0.5, "IZI")   # Z on qubit 1
h.add_term(0.2, "XXI")   # X-X coupling on qubits 0-1

# Compute expectation value
import numpy as np
state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
energy = h.compute_energy(state)

# Time evolution
evolved = h.time_evolution(state, time=0.5)

# Ground state
ground_energy, ground_state = h.get_ground_state()
```

**Key Methods**:
- `add_term(coefficient, pauli_string)`: Add a term to the Hamiltonian
- `get_matrix()`: Get full matrix representation
- `compute_energy(state)`: Calculate expectation value ⟨ψ|H|ψ⟩
- `time_evolution(state, time)`: Evolve state via U(t) = e^(-iHt)
- `get_ground_state()`: Find minimum energy eigenstate

### MLQuantumAddon

Machine learning capabilities for Hamiltonian-aware quantum computing:

```python
from quantum_profiles import MLQuantumAddon, Hamiltonian

ml = MLQuantumAddon(learning_rate=0.01)

# Log Hamiltonian configurations
ml.log_hamiltonian(h, metadata={'epoch': 1})

# Log expectation values
ml.log_expectation_value(h, state, "measurement_1")

# Optimize Hamiltonian parameters
error, iterations = ml.optimize_hamiltonian_parameters(
    h, state, target_energy=0.0, num_iterations=100
)

# Learn from measurement data
measurements = [(state1, energy1), (state2, energy2)]
learned_h = ml.learn_from_measurements(measurements)

# Get statistics
stats = ml.get_expectation_statistics()
summary = ml.get_summary()
```

**Key Features**:
- Hamiltonian and expectation value logging
- Gradient-based parameter optimization
- Learning from measurement data
- Statistical analysis of expectation values

### QuantumProfileManager

Unified interface for managing quantum processor profiles:

```python
from quantum_profiles import QuantumProfileManager

# Initialize with default profile
manager = QuantumProfileManager(default_profile='alternative', n_qubits=4)

# Add Hamiltonian term
manager.add_term(0.5, "XYZI")

# Compute energy
energy = manager.compute_energy()

# Time evolution
evolved = manager.time_evolution(time=0.3)

# Switch profiles
manager.switch_profile('castle', copy_state=True)

# Compare energies across all profiles
energies = manager.compare_profiles()
# Returns: {'alternative': 0.4, 'castle': 9.0, ...}

# Access ML addon
ml = manager.get_ml_addon()
```

**Key Features**:
- Seamless profile switching with state preservation
- Unified Hamiltonian operations across all profiles
- Profile comparison capabilities
- Integrated ML addon

### Enhanced QuantumVirtualProcessor

The main quantum processor now includes full Hamiltonian support:

```python
from quantum_virtual_processor import QuantumVirtualProcessor

# Initialize with specific profile
qvp = QuantumVirtualProcessor(qubits=4, profile='omega')

# Add Hamiltonian terms
qvp.add_hamiltonian_term(0.5, "ZXYZ")

# Compute energy
energy = qvp.compute_energy()

# Time evolution
evolved = qvp.time_evolve(time=0.3)

# Switch profile
qvp.switch_profile('empire')

# Access ML addon
ml = qvp.get_ml_addon()

# Get summary
summary = qvp.get_summary()
```

## Usage Examples

### Basic Hamiltonian Operations

```python
from quantum_profiles import Hamiltonian
import numpy as np

# Create Hamiltonian
h = Hamiltonian(n_qubits=2)
h.add_term(1.0, "ZI")
h.add_term(1.0, "IZ")
h.add_term(0.5, "XX")

# Compute ground state
ground_energy, ground_state = h.get_ground_state()
print(f"Ground state energy: {ground_energy}")

# Time evolution from |00⟩
initial_state = np.array([1, 0, 0, 0], dtype=complex)
final_state = h.time_evolution(initial_state, time=1.0)
```

### Using Quantum Profiles

```python
from quantum_profiles import AlternativeProfile, CastleProfile

# Alternative profile for exploration
alt = AlternativeProfile(n_qubits=3)
alt.add_term(0.3, "XYZ")
energy = alt.compute_energy()

# Castle profile for stability
castle = CastleProfile(n_qubits=3)
ground_energy, ground_state = castle.get_ground_state()
```

### ML-Guided Hamiltonian Learning

```python
from quantum_profiles import Hamiltonian, MLQuantumAddon
import numpy as np

# Create ML addon
ml = MLQuantumAddon(learning_rate=0.05)

# Generate training data
h_true = Hamiltonian(n_qubits=2)
h_true.add_term(1.5, "ZI")
h_true.add_term(0.8, "IZ")

measurements = []
for _ in range(5):
    state = np.random.randn(4) + 1j * np.random.randn(4)
    state = state / np.linalg.norm(state)
    energy = h_true.compute_energy(state)
    measurements.append((state, energy))

# Learn Hamiltonian
learned_h = ml.learn_from_measurements(measurements)
```

### Profile Comparison

```python
from quantum_profiles import QuantumProfileManager
import numpy as np

manager = QuantumProfileManager(default_profile='alternative', n_qubits=3)

# Create a test state
state = np.zeros(8, dtype=complex)
state[0] = 1.0

# Compare energies across all profiles
energies = manager.compare_profiles(state)

for profile_name, energy in sorted(energies.items(), key=lambda x: x[1]):
    print(f"{profile_name:12}: {energy:8.4f}")
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_quantum_hamiltonian.py
```

The test suite includes:
- Hamiltonian operations (initialization, term addition, matrix assembly)
- Energy calculations and expectation values
- Time evolution under Hamiltonian dynamics
- Ground state computation
- All 5 quantum profiles
- ML addon functionality (logging, optimization, learning)
- Profile manager operations
- QuantumVirtualProcessor integration

## Demonstration

Run the demonstration script to see all features in action:

```bash
python3 demo_quantum_hamiltonian.py
```

The demonstration showcases:
1. Basic Hamiltonian operations
2. Comparison of all 5 quantum profiles
3. ML-guided Hamiltonian parameter optimization
4. Profile management and switching
5. Integration with QuantumVirtualProcessor

## Dependencies

- **NumPy**: Matrix operations and linear algebra
- **SciPy**: Matrix exponential for time evolution

Install dependencies:
```bash
pip3 install numpy scipy
```

## Implementation Details

### Time Evolution

Time evolution is implemented using the matrix exponential:

U(t) = exp(-iHt)

The scipy.linalg.expm function is used for numerical stability with anti-Hermitian matrices.

### Gradient Computation

Parameter gradients use finite differences:

∂E/∂θ ≈ (E(θ+ε) - E(θ-ε)) / (2ε)

This enables gradient-based optimization of Hamiltonian parameters.

### Profile Initialization

Each profile automatically initializes its characteristic Hamiltonian in the `initialize_characteristic_hamiltonian()` method, which is called during construction.

## Performance Considerations

- **Matrix Size**: The Hamiltonian matrix has dimension 2^n × 2^n for n qubits
- **Scaling**: Recommended for up to 10-12 qubits on standard hardware
- **Caching**: Matrix representations are cached until Hamiltonian is modified
- **Time Evolution**: Uses optimized scipy routines for matrix exponential

## Future Enhancements

Potential extensions to the framework:
- Variational quantum eigensolvers (VQE)
- Quantum approximate optimization algorithm (QAOA)
- Trotterized time evolution for larger systems
- Noise models and error mitigation
- GPU acceleration for larger Hamiltonians

## References

- Nielsen & Chuang, "Quantum Computation and Quantum Information"
- Quantum Hamiltonian mechanics
- Pauli operator algebra
- Time-dependent Schrödinger equation

## License

Part of the Pandora AIOS project.
