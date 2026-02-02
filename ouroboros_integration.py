"""
Integration module for Ouroboros Sync with existing Pandora AIOS Ouroboros components.

This module provides integration between the Rust-based Ouroboros sync consensus
mechanism and the existing Python-based Ouroboros overlay system.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import time

try:
    from ouroboros_sync_wrapper import OuroborosSyncWrapper, RUST_MODULE_AVAILABLE
except ImportError:
    RUST_MODULE_AVAILABLE = False

try:
    from ouroboros_overlay import OuroborosOverlay
    OVERLAY_AVAILABLE = True
except ImportError:
    OVERLAY_AVAILABLE = False


class OuroborosIntegratedSystem:
    """
    Integrated Ouroboros system combining overlay and sync consensus.
    
    This class provides a unified interface to both the quantum overlay
    processing (Python) and the blockchain consensus mechanism (Rust).
    """
    
    def __init__(self,
                 num_qubits: int = 8,
                 ergotropy_threshold: float = 0.5,
                 stability_window: int = 10,
                 max_recursion_depth: int = 5):
        """
        Initialize the integrated Ouroboros system.
        
        Args:
            num_qubits: Number of qubits for overlay processing
            ergotropy_threshold: Minimum ergotropy for block validation
            stability_window: History window for stability checks
            max_recursion_depth: Maximum recursion depth for stability
        """
        self.num_qubits = num_qubits
        
        # Initialize overlay if available
        if OVERLAY_AVAILABLE:
            self.overlay = OuroborosOverlay()
            self.overlay.initialize_overlay(
                num_qubits=num_qubits,
                enable_virtual_processor=True,
                enable_ergotropy=True
            )
        else:
            self.overlay = None
        
        # Initialize sync consensus if available
        if RUST_MODULE_AVAILABLE:
            self.sync = OuroborosSyncWrapper(
                ergotropy_threshold=ergotropy_threshold,
                stability_window=stability_window,
                max_recursion_depth=max_recursion_depth
            )
        else:
            self.sync = None
        
        self.processing_history = []
    
    def process_quantum_state(self, quantum_state: np.ndarray) -> Dict[str, Any]:
        """
        Process a quantum state through the overlay and validate via consensus.
        
        Args:
            quantum_state: Input quantum state vector
            
        Returns:
            Dictionary with processing results and validation status
        """
        result = {
            'timestamp': time.time(),
            'overlay_processing': None,
            'consensus_validation': None,
            'integrated': False
        }
        
        # Process through overlay
        if self.overlay is not None:
            processed_state = self.overlay.apply_overlay_logic(quantum_state)
            
            # Calculate ergotropy from processed state
            ergotropy = self._calculate_ergotropy(processed_state)
            
            # Get overlay measurement
            measurement = self.overlay.process_measurement({
                'state_vector': processed_state,
                'bitstring': None,
                'counts': {},
                'metadata': {}
            })
            
            result['overlay_processing'] = {
                'processed_state': processed_state,
                'ergotropy': ergotropy,
                'measurement': measurement,
                'cycle': self.overlay.cycle_counter
            }
            
            # Validate through consensus if available
            if self.sync is not None:
                block_data = f"cycle_{self.overlay.cycle_counter}_ergotropy_{ergotropy:.4f}"
                block = self.sync.create_block(data=block_data, ergotropy=ergotropy)
                validation = self.sync.validate_and_add_block(block)
                
                result['consensus_validation'] = validation
                result['integrated'] = validation['valid']
                
                # Store in history
                self.processing_history.append({
                    'timestamp': result['timestamp'],
                    'ergotropy': ergotropy,
                    'valid': validation['valid']
                })
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'overlay_available': OVERLAY_AVAILABLE,
            'sync_available': RUST_MODULE_AVAILABLE,
            'integrated': OVERLAY_AVAILABLE and RUST_MODULE_AVAILABLE,
        }
        
        if self.overlay is not None:
            status['overlay_info'] = self.overlay.get_overlay_info()
        
        if self.sync is not None:
            status['chain_info'] = self.sync.get_chain_info()
        
        status['processing_history_length'] = len(self.processing_history)
        
        return status
    
    def _calculate_ergotropy(self, state: np.ndarray) -> float:
        """
        Calculate ergotropy from quantum state.
        
        Ergotropy is a measure of extractable work from a quantum state.
        This is a simplified calculation based on state purity.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Ergotropy value (0.0 to 1.0)
        """
        # Normalize state
        normalized = state / np.linalg.norm(state)
        
        # Calculate purity (sum of 4th powers of amplitudes)
        purity = np.sum(np.abs(normalized) ** 4)
        
        # Calculate von Neumann entropy approximation
        probabilities = np.abs(normalized) ** 2
        probabilities = probabilities[probabilities > 1e-10]  # Filter near-zero
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Ergotropy as function of purity and entropy
        # Higher purity and lower entropy indicate higher ergotropy
        max_entropy = np.log2(len(state))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        ergotropy = 0.5 * purity + 0.5 * (1.0 - normalized_entropy)
        
        return float(np.clip(ergotropy, 0.0, 1.0))
    
    def reset(self):
        """Reset the integrated system."""
        if self.overlay is not None:
            self.overlay.reset_overlay()
        
        if self.sync is not None:
            self.sync.reset()
        
        self.processing_history.clear()


def demo_integrated_system():
    """
    Demonstration of the integrated Ouroboros system.
    """
    print("=" * 70)
    print("Ouroboros Integrated System - Demo")
    print("=" * 70)
    
    if not OVERLAY_AVAILABLE or not RUST_MODULE_AVAILABLE:
        print("\nWARNING: Not all components are available:")
        print(f"  - Overlay: {'Available' if OVERLAY_AVAILABLE else 'Not available'}")
        print(f"  - Sync: {'Available' if RUST_MODULE_AVAILABLE else 'Not available'}")
        print("\nCannot run full integration demo.")
        return
    
    # Create integrated system
    print("\n1. Initializing integrated system...")
    system = OuroborosIntegratedSystem(
        num_qubits=8,
        ergotropy_threshold=0.4,
        stability_window=10
    )
    
    status = system.get_system_status()
    print(f"   System integrated: {status['integrated']}")
    
    # Process quantum states
    print("\n2. Processing quantum states...")
    for i in range(5):
        # Create random quantum state
        state = np.random.randn(8) + 1j * np.random.randn(8)
        state = state / np.linalg.norm(state)
        
        result = system.process_quantum_state(state)
        
        if result['overlay_processing']:
            ergotropy = result['overlay_processing']['ergotropy']
            print(f"   Cycle {i}: ergotropy={ergotropy:.4f}", end="")
            
            if result['consensus_validation']:
                valid = result['consensus_validation']['valid']
                print(f", validated={valid}")
            else:
                print()
    
    # Show system status
    print("\n3. System status:")
    final_status = system.get_system_status()
    
    if 'overlay_info' in final_status:
        overlay = final_status['overlay_info']
        print(f"   Overlay cycles: {overlay['cycle_counter']}")
        print(f"   Total bounces: {overlay['total_bounces']}")
    
    if 'chain_info' in final_status:
        chain = final_status['chain_info']
        print(f"   Chain length: {chain['length']}")
        if chain['stability_stats']['count'] > 0:
            stats = chain['stability_stats']
            print(f"   Mean ergotropy: {stats['mean']:.4f}")
            print(f"   Ergotropy variance: {stats['variance']:.4f}")
    
    print("\n" + "=" * 70)
    print("Integration demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demo_integrated_system()
