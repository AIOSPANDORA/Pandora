"""
Four Overlay Quantum Virtual Processor

A quantum processor that maintains multiple quantum buffers (overlays) for
parallel quantum computation. By default, it creates 4 overlays with 25 qubits each,
but Pandora can dynamically add more overlays and customize qubit counts.

Components:
    - FourOverlayQuantumVirtualProcessor: Multi-buffer quantum processor
    - Overlay management functions
    - Add-on support for extensibility
"""

import logging
from typing import List, Optional, Any, Dict
from .alternative_quantum_virtual_processor import (
    QuantumAddon, LoggerAddon, QuantumQubitBuffer
)


class QuantumOverlay:
    """
    Represents a single quantum overlay buffer.
    
    An overlay is an independent quantum computation space with its own
    qubit buffer and state management.
    """
    
    def __init__(self, overlay_id: int, qubit_count: int = 25):
        """
        Initialize a quantum overlay.
        
        Args:
            overlay_id (int): Unique identifier for this overlay
            qubit_count (int): Number of qubits in this overlay
        """
        self.id = overlay_id
        self.qubit_count = qubit_count
        self.buffer = QuantumQubitBuffer(qubit_count)
        self.operation_count = 0
        self.is_active = True
    
    def apply_gate(self, gate: str, reg: int) -> None:
        """
        Apply a quantum gate to a qubit in this overlay.
        
        Args:
            gate (str): Name of the gate
            reg (int): Qubit register index
        """
        if not self.is_active:
            return
        
        # Simulate gate operation
        if gate.upper() == 'X':
            current = self.buffer.get_state(reg)
            self.buffer.set_state(reg, 1 - current)
        elif gate.upper() == 'H':
            self.buffer.set_superposition(reg, True)
        elif gate.upper() == 'Z':
            pass  # Phase operations
        
        self.operation_count += 1
    
    def measure(self, reg: Optional[int] = None) -> Any:
        """
        Measure qubits in this overlay.
        
        Args:
            reg (int, optional): Specific qubit to measure, or None for all
            
        Returns:
            Measurement result
        """
        import random
        
        if reg is not None:
            if self.buffer.is_superposition[reg]:
                result = random.choice([0, 1])
                self.buffer.set_state(reg, result)
                self.buffer.set_superposition(reg, False)
            else:
                result = self.buffer.get_state(reg)
        else:
            result = []
            for i in range(self.qubit_count):
                if self.buffer.is_superposition[i]:
                    val = random.choice([0, 1])
                    self.buffer.set_state(i, val)
                    self.buffer.set_superposition(i, False)
                    result.append(val)
                else:
                    result.append(self.buffer.get_state(i))
        
        return result
    
    def reset(self) -> None:
        """Reset the overlay to initial state."""
        self.buffer.reset()
        self.operation_count = 0
    
    def deactivate(self) -> None:
        """Deactivate this overlay."""
        self.is_active = False
    
    def activate(self) -> None:
        """Activate this overlay."""
        self.is_active = True


class FourOverlayQuantumVirtualProcessor:
    """
    Quantum processor with multiple overlay buffers for parallel computation.
    
    Maintains 4 overlays by default (4 x 25 qubits), but supports dynamic
    addition of overlays and customization of qubit counts. Each overlay
    operates independently, allowing for parallel quantum operations.
    """
    
    def __init__(
        self,
        overlay_count: int = 4,
        qubits_per_overlay: int = 25,
        addons: Optional[List[QuantumAddon]] = None
    ):
        """
        Initialize the four overlay processor.
        
        Args:
            overlay_count (int): Number of overlays to create (default: 4)
            qubits_per_overlay (int): Qubits per overlay (default: 25)
            addons (List[QuantumAddon], optional): Add-ons to install
        """
        self.overlay_count = overlay_count
        self.qubits_per_overlay = qubits_per_overlay
        self.overlays: Dict[int, QuantumOverlay] = {}
        self.addons = addons if addons is not None else []
        self.current_overlay = 0
        self.total_operations = 0
        
        # Create initial overlays
        for i in range(overlay_count):
            self._create_overlay(i, qubits_per_overlay)
    
    def _create_overlay(self, overlay_id: int, qubit_count: int) -> QuantumOverlay:
        """
        Create a new overlay.
        
        Args:
            overlay_id (int): Unique ID for the overlay
            qubit_count (int): Number of qubits
            
        Returns:
            QuantumOverlay: The created overlay
        """
        overlay = QuantumOverlay(overlay_id, qubit_count)
        self.overlays[overlay_id] = overlay
        return overlay
    
    def add_overlay(self, qubit_count: Optional[int] = None) -> int:
        """
        Add a new overlay to the processor.
        
        Args:
            qubit_count (int, optional): Number of qubits (default: qubits_per_overlay)
            
        Returns:
            int: ID of the newly created overlay
        """
        if qubit_count is None:
            qubit_count = self.qubits_per_overlay
        
        # Find next available ID
        new_id = max(self.overlays.keys()) + 1 if self.overlays else 0
        self._create_overlay(new_id, qubit_count)
        self.overlay_count += 1
        
        # Notify add-ons
        for addon in self.addons:
            if addon.enabled and hasattr(addon, 'on_overlay_added'):
                addon.on_overlay_added(new_id, qubit_count)
        
        return new_id
    
    def remove_overlay(self, overlay_id: int) -> bool:
        """
        Remove an overlay from the processor.
        
        Args:
            overlay_id (int): ID of overlay to remove
            
        Returns:
            bool: True if removed, False if overlay not found
        """
        if overlay_id in self.overlays:
            del self.overlays[overlay_id]
            self.overlay_count -= 1
            
            # Adjust current overlay if needed
            if self.current_overlay == overlay_id and self.overlays:
                self.current_overlay = next(iter(self.overlays.keys()))
            
            return True
        return False
    
    def select_overlay(self, overlay_id: int) -> bool:
        """
        Select the active overlay for operations.
        
        Args:
            overlay_id (int): ID of overlay to activate
            
        Returns:
            bool: True if selected, False if overlay not found
        """
        if overlay_id in self.overlays:
            self.current_overlay = overlay_id
            return True
        return False
    
    def get_overlay(self, overlay_id: int) -> Optional[QuantumOverlay]:
        """
        Get a specific overlay by ID.
        
        Args:
            overlay_id (int): Overlay ID
            
        Returns:
            QuantumOverlay: The overlay, or None if not found
        """
        return self.overlays.get(overlay_id)
    
    def apply_gate(self, gate: str, reg: int, overlay_id: Optional[int] = None) -> None:
        """
        Apply a quantum gate to a register.
        
        Args:
            gate (str): Name of the gate
            reg (int): Register/qubit index
            overlay_id (int, optional): Target overlay (default: current)
        """
        target_id = overlay_id if overlay_id is not None else self.current_overlay
        
        if target_id not in self.overlays:
            return
        
        # Notify add-ons
        for addon in self.addons:
            if addon.enabled:
                addon.on_gate_apply(f"{gate}@overlay{target_id}", reg)
        
        # Apply gate to overlay
        self.overlays[target_id].apply_gate(gate, reg)
        self.total_operations += 1
    
    def measure(
        self,
        reg: Optional[int] = None,
        overlay_id: Optional[int] = None
    ) -> Any:
        """
        Measure quantum state in an overlay.
        
        Args:
            reg (int, optional): Specific register to measure, or None for all
            overlay_id (int, optional): Target overlay (default: current)
            
        Returns:
            Measurement result
        """
        target_id = overlay_id if overlay_id is not None else self.current_overlay
        
        if target_id not in self.overlays:
            return None
        
        result = self.overlays[target_id].measure(reg)
        
        # Notify add-ons
        for addon in self.addons:
            if addon.enabled:
                addon.on_measure(f"overlay{target_id}: {result}")
        
        return result
    
    def measure_all_overlays(self, reg: Optional[int] = None) -> Dict[int, Any]:
        """
        Measure a register across all overlays.
        
        Args:
            reg (int, optional): Register to measure in each overlay
            
        Returns:
            dict: Mapping of overlay_id to measurement result
        """
        results = {}
        for overlay_id, overlay in self.overlays.items():
            if overlay.is_active:
                results[overlay_id] = overlay.measure(reg)
        
        return results
    
    def reset(self, overlay_id: Optional[int] = None) -> None:
        """
        Reset overlays to initial state.
        
        Args:
            overlay_id (int, optional): Specific overlay to reset, or None for all
        """
        if overlay_id is not None:
            if overlay_id in self.overlays:
                self.overlays[overlay_id].reset()
        else:
            for overlay in self.overlays.values():
                overlay.reset()
            self.total_operations = 0
    
    def add_addon(self, addon: QuantumAddon) -> None:
        """
        Add an add-on to the processor.
        
        Args:
            addon (QuantumAddon): Add-on instance to register
        """
        if addon not in self.addons:
            self.addons.append(addon)
    
    def remove_addon(self, addon: QuantumAddon) -> None:
        """
        Remove an add-on from the processor.
        
        Args:
            addon (QuantumAddon): Add-on instance to remove
        """
        if addon in self.addons:
            self.addons.remove(addon)
    
    def get_state_info(self) -> dict:
        """
        Get comprehensive state information about the processor.
        
        Returns:
            dict: State information for all overlays
        """
        overlay_info = {}
        for overlay_id, overlay in self.overlays.items():
            overlay_info[overlay_id] = {
                'qubits': overlay.qubit_count,
                'operations': overlay.operation_count,
                'active': overlay.is_active
            }
        
        return {
            'overlay_count': self.overlay_count,
            'qubits_per_overlay': self.qubits_per_overlay,
            'current_overlay': self.current_overlay,
            'total_operations': self.total_operations,
            'overlays': overlay_info,
            'addons': [addon.name for addon in self.addons if addon.enabled]
        }
    
    def get_total_qubits(self) -> int:
        """
        Calculate total number of qubits across all overlays.
        
        Returns:
            int: Total qubit count
        """
        return sum(overlay.qubit_count for overlay in self.overlays.values())


def get_profile():
    """
    Factory function to create a fully configured FourOverlayQuantumVirtualProcessor.
    
    Creates a processor with 4 overlays of 25 qubits each (100 total qubits),
    with standard add-ons pre-configured.
    
    Returns:
        FourOverlayQuantumVirtualProcessor: Configured processor instance
    """
    # Create processor with default 4x25 configuration
    processor = FourOverlayQuantumVirtualProcessor(
        overlay_count=4,
        qubits_per_overlay=25
    )
    
    # Add standard add-ons
    logger_addon = LoggerAddon(log_level=logging.INFO)
    processor.add_addon(logger_addon)
    
    return processor
