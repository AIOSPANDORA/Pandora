"""
Empire Quantum Virtual Processor

Features a central control lattice (supervised by Pandora) and an expandable
grid of Kaleidoscopic Hives. Each component is a quantum virtual processor
with configurable qubit sizes.

Architecture:
- Central Control Lattice: A QuantumVirtualProcessor (default 4 qubits)
- Kaleidoscopic Hives Grid: A 2D grid of HiveQuantumVirtualProcessors (default 2x2 grid, 8 qubits each)
- Supports dynamic expansion of both control lattice and hive grid
- Routes gates/measurements to entire empire, control lattice, or specific hives
- Full addon support
"""

from quantum_virtual_processor import QuantumVirtualProcessor


class HiveQuantumVirtualProcessor(QuantumVirtualProcessor):
    """
    A specialized quantum processor representing a single Kaleidoscopic Hive.
    Inherits from QuantumVirtualProcessor and adds hive-specific functionality.
    """
    
    def __init__(self, qubits=8, hive_id=None):
        """
        Initialize a Kaleidoscopic Hive quantum processor.
        
        Args:
            qubits (int): Number of qubits in this hive (default: 8)
            hive_id (tuple): Optional identifier for this hive's position (row, col)
        """
        super().__init__(qubits=qubits)
        self.hive_id = hive_id
        self.hive_qubits = qubits
        
    def __repr__(self):
        """String representation of the hive."""
        hive_pos = f" at {self.hive_id}" if self.hive_id else ""
        return f"<HiveQuantumVirtualProcessor{hive_pos}: {self.hive_qubits} qubits>"


class EmpireQuantumVirtualProcessor:
    """
    Empire Quantum Virtual Processor Profile
    
    A hierarchical quantum computing architecture with:
    - A central control lattice for coordination
    - An expandable grid of Kaleidoscopic Hives for distributed quantum operations
    
    The Empire profile allows Pandora/API to:
    - Dynamically expand the number of hives (grid size)
    - Adjust control lattice size
    - Direct operations to the entire empire, control lattice only, or specific hives
    - Support addons for enhanced functionality
    """
    
    def __init__(self, control_qubits=4, grid_size=(2, 2), hive_qubits=8):
        """
        Initialize the Empire quantum processor.
        
        Args:
            control_qubits (int): Number of qubits in the central control lattice (default: 4)
            grid_size (tuple): Grid dimensions (rows, cols) for hives (default: (2, 2))
            hive_qubits (int): Number of qubits per hive (default: 8)
        """
        self.control_lattice = QuantumVirtualProcessor(qubits=control_qubits)
        self.control_qubits = control_qubits
        self.grid_size = grid_size
        self.hive_qubits = hive_qubits
        self.hive_grid = {}
        self.addons = []
        
        # Initialize the hive grid
        self._initialize_hive_grid()
        
    def _initialize_hive_grid(self):
        """Initialize the grid of Kaleidoscopic Hives."""
        rows, cols = self.grid_size
        self.hive_grid = {}
        for row in range(rows):
            for col in range(cols):
                hive_id = (row, col)
                self.hive_grid[hive_id] = HiveQuantumVirtualProcessor(
                    qubits=self.hive_qubits,
                    hive_id=hive_id
                )
    
    def expand_grid(self, new_grid_size):
        """
        Dynamically expand the hive grid to a new size.
        
        Args:
            new_grid_size (tuple): New grid dimensions (rows, cols)
            
        Returns:
            bool: True if expansion was successful
        """
        new_rows, new_cols = new_grid_size
        old_rows, old_cols = self.grid_size
        
        if new_rows < old_rows or new_cols < old_cols:
            raise ValueError("Cannot shrink grid size. New size must be >= current size.")
        
        # Add new hives for expanded grid
        for row in range(new_rows):
            for col in range(new_cols):
                hive_id = (row, col)
                if hive_id not in self.hive_grid:
                    self.hive_grid[hive_id] = HiveQuantumVirtualProcessor(
                        qubits=self.hive_qubits,
                        hive_id=hive_id
                    )
        
        self.grid_size = new_grid_size
        return True
    
    def expand_control_lattice(self, new_qubit_count):
        """
        Expand the control lattice to support more qubits.
        
        Args:
            new_qubit_count (int): New number of qubits for the control lattice
            
        Returns:
            bool: True if expansion was successful
        """
        if new_qubit_count < self.control_qubits:
            raise ValueError("Cannot reduce control lattice size. New size must be >= current size.")
        
        # Create new control lattice with expanded qubit count
        self.control_lattice = QuantumVirtualProcessor(qubits=new_qubit_count)
        self.control_qubits = new_qubit_count
        return True
    
    def apply_gate_to_empire(self, gate, reg):
        """
        Apply a quantum gate to the entire Empire (control lattice + all hives).
        
        Args:
            gate (str): Gate type to apply
            reg (int): Register/qubit index
        """
        # Apply to control lattice
        self.control_lattice.apply_gate(gate, reg)
        
        # Apply to all hives
        for hive in self.hive_grid.values():
            hive.apply_gate(gate, reg)
    
    def apply_gate_to_control_lattice(self, gate, reg):
        """
        Apply a quantum gate only to the central control lattice.
        
        Args:
            gate (str): Gate type to apply
            reg (int): Register/qubit index
        """
        self.control_lattice.apply_gate(gate, reg)
    
    def apply_gate_to_hive(self, hive_id, gate, reg):
        """
        Apply a quantum gate to a specific hive block.
        
        Args:
            hive_id (tuple): Hive identifier (row, col)
            gate (str): Gate type to apply
            reg (int): Register/qubit index
            
        Raises:
            KeyError: If hive_id does not exist in the grid
        """
        if hive_id not in self.hive_grid:
            raise KeyError(f"Hive {hive_id} does not exist in the grid.")
        
        self.hive_grid[hive_id].apply_gate(gate, reg)
    
    def measure_empire(self):
        """
        Measure the entire Empire (control lattice + all hives).
        
        Returns:
            dict: Measurement results with 'control' and 'hives' keys
        """
        results = {
            'control': self.control_lattice.measure(),
            'hives': {}
        }
        
        for hive_id, hive in self.hive_grid.items():
            results['hives'][hive_id] = hive.measure()
        
        return results
    
    def measure_control_lattice(self):
        """
        Measure only the central control lattice.
        
        Returns:
            Measurement result from the control lattice
        """
        return self.control_lattice.measure()
    
    def measure_hive(self, hive_id):
        """
        Measure a specific hive block.
        
        Args:
            hive_id (tuple): Hive identifier (row, col)
            
        Returns:
            Measurement result from the specified hive
            
        Raises:
            KeyError: If hive_id does not exist in the grid
        """
        if hive_id not in self.hive_grid:
            raise KeyError(f"Hive {hive_id} does not exist in the grid.")
        
        return self.hive_grid[hive_id].measure()
    
    def get_hive(self, hive_id):
        """
        Get a reference to a specific hive processor.
        
        Args:
            hive_id (tuple): Hive identifier (row, col)
            
        Returns:
            HiveQuantumVirtualProcessor: The hive processor
            
        Raises:
            KeyError: If hive_id does not exist in the grid
        """
        if hive_id not in self.hive_grid:
            raise KeyError(f"Hive {hive_id} does not exist in the grid.")
        
        return self.hive_grid[hive_id]
    
    def list_hives(self):
        """
        List all hive identifiers in the grid.
        
        Returns:
            list: List of hive IDs (row, col) tuples
        """
        return sorted(self.hive_grid.keys())
    
    def register_addon(self, addon):
        """
        Register an addon for the Empire profile.
        
        Addons can extend the functionality of the Empire processor.
        
        Args:
            addon: Addon object to register
        """
        self.addons.append(addon)
    
    def execute_addon(self, addon_name, *args, **kwargs):
        """
        Execute a registered addon by name.
        
        Args:
            addon_name (str): Name of the addon to execute
            *args: Positional arguments for the addon
            **kwargs: Keyword arguments for the addon
            
        Returns:
            Result from the addon execution
            
        Raises:
            ValueError: If addon is not found
        """
        for addon in self.addons:
            if hasattr(addon, 'name') and addon.name == addon_name:
                if hasattr(addon, 'execute'):
                    return addon.execute(self, *args, **kwargs)
        
        raise ValueError(f"Addon '{addon_name}' not found or does not have execute method.")
    
    def get_empire_stats(self):
        """
        Get statistics about the Empire configuration.
        
        Returns:
            dict: Dictionary with Empire statistics
        """
        return {
            'control_lattice_qubits': self.control_qubits,
            'grid_size': self.grid_size,
            'total_hives': len(self.hive_grid),
            'hive_qubits': self.hive_qubits,
            'total_qubits': self.control_qubits + (len(self.hive_grid) * self.hive_qubits),
            'registered_addons': len(self.addons)
        }
    
    def __repr__(self):
        """String representation of the Empire."""
        stats = self.get_empire_stats()
        return (f"<EmpireQuantumVirtualProcessor: "
                f"{stats['control_lattice_qubits']} control qubits, "
                f"{stats['total_hives']} hives ({self.grid_size[0]}x{self.grid_size[1]}), "
                f"{stats['total_qubits']} total qubits>")

A high-performance quantum processor profile focused on optimized quantum computation
with integrated ML-based process logging and adaptive learning.

Philosophy:
- Emphasizes performance and optimization
- Aggressive computation strategies
- Self-instrumenting with ML-based monitoring
- Learns from all quantum operations for maximum efficiency
"""

from quantum_virtual_processor import QuantumVirtualProcessor
from quantum_profiles.ml_quantum_addon import MLQuantumAddon


class EmpireQuantumVirtualProcessor(QuantumVirtualProcessor):
    """
    Empire quantum processor with ML-based self-instrumentation.
    
    This profile emphasizes high-performance quantum computation while
    automatically logging and learning from all operations.
    
    Usage:
        >>> processor = EmpireQuantumVirtualProcessor(qubits=6)
        >>> processor.apply_gate("H", 0)
        >>> result = processor.measure()
        >>> stats = processor.get_addon_statistics()
    """
    
    # Define addons list with ML quantum addon enabled by default
    ADDONS = [
        MLQuantumAddon(train_on_fly=True)
    ]
    
    def __init__(self, qubits=6, **kwargs):
        """
        Initialize Empire quantum processor with ML instrumentation.
        
        Args:
            qubits: Number of qubits (default: 6)
            **kwargs: Additional arguments
        """
        super().__init__(qubits)
        
        # Initialize addons
        self.addons = [addon for addon in self.ADDONS]
        
        # Store processor context for addon hooks
        self._context = {
            'qubits': qubits,
            'profile': 'Empire'
        }
    
    def _before_gate_hooks(self, gate: str, register: int) -> None:
        """
        Call all addon before_gate hooks.
        
        Args:
            gate: Gate name
            register: Register/qubit index
        """
        for addon in self.addons:
            if hasattr(addon, 'before_gate'):
                addon.before_gate(gate, register, self._context)
    
    def _after_gate_hooks(self, gate: str, register: int) -> None:
        """
        Call all addon after_gate hooks.
        
        Args:
            gate: Gate name
            register: Register/qubit index
        """
        for addon in self.addons:
            if hasattr(addon, 'after_gate'):
                addon.after_gate(gate, register, self._context)
    
    def _before_measurement_hooks(self, register: int = None) -> None:
        """
        Call all addon before_measurement hooks.
        
        Args:
            register: Optional register/qubit index
        """
        for addon in self.addons:
            if hasattr(addon, 'before_measurement'):
                addon.before_measurement(register, self._context)
    
    def _after_measurement_hooks(self, register: int, result: int) -> None:
        """
        Call all addon after_measurement hooks.
        
        Args:
            register: Register/qubit index
            result: Measurement result
        """
        for addon in self.addons:
            if hasattr(addon, 'after_measurement'):
                addon.after_measurement(register, result, self._context)
    
    def apply_gate(self, gate: str, register: int, **kwargs) -> None:
        """
        Apply a quantum gate with ML monitoring.
        
        Args:
            gate: Gate name (e.g., 'H', 'X', 'CNOT')
            register: Register/qubit to apply gate to
            **kwargs: Additional gate parameters
        """
        # Update context with any additional parameters
        context = self._context.copy()
        context.update(kwargs)
        self._context = context
        
        # Before hooks
        self._before_gate_hooks(gate, register)
        
        # Apply the gate (call parent implementation)
        super().apply_gate(gate, register)
        
        # After hooks
        self._after_gate_hooks(gate, register)
    
    def measure(self, register: int = None) -> int:
        """
        Measure quantum register with ML monitoring.
        
        Args:
            register: Optional register to measure (None = measure all)
            
        Returns:
            Measurement result
        """
        # Before hooks
        self._before_measurement_hooks(register)
        
        # Perform measurement (call parent implementation)
        result = super().measure()
        
        # After hooks
        self._after_measurement_hooks(register if register is not None else 0, result)
        
        return result
    
    def expand_qubits(self, additional_qubits: int) -> None:
        """
        Expand the quantum system with additional qubits.
        
        Args:
            additional_qubits: Number of qubits to add
        """
        old_qubits = self.qubits
        self.qubits += additional_qubits
        self._context['qubits'] = self.qubits
        
        # Log expansion to addons
        for addon in self.addons:
            if hasattr(addon, 'log_expansion'):
                addon.log_expansion(
                    f"Expanded from {old_qubits} to {self.qubits} qubits",
                    self._context
                )
    
    def get_diagnostic_info(self) -> dict:
        """
        Get diagnostic information about processor state.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostic = {
            'qubits': self.qubits,
            'profile': 'Empire',
            'addons': len(self.addons)
        }
        
        # Log diagnostic to addons
        for addon in self.addons:
            if hasattr(addon, 'log_diagnostic'):
                addon.log_diagnostic('processor_state', diagnostic, self._context)
        
        return diagnostic
    
    def get_addon_statistics(self) -> dict:
        """
        Get statistics from all addons.
        
        Returns:
            Dictionary mapping addon index to statistics
        """
        stats = {}
        for i, addon in enumerate(self.addons):
            if hasattr(addon, 'get_statistics'):
                stats[f'addon_{i}'] = addon.get_statistics()
        return stats
    
    def get_ml_addon(self) -> MLQuantumAddon:
        """
        Get the ML quantum addon instance.
        
        Returns:
            MLQuantumAddon instance or None if not found
        """
        for addon in self.addons:
            if isinstance(addon, MLQuantumAddon):
                return addon
        return None
