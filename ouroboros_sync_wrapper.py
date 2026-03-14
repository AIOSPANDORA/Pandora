"""
Python wrapper for the Rust-based Ouroboros sync consensus mechanism.

This module provides a Python interface to the Rust implementation of the
Ouroboros consensus mechanism for validating blocks from Æthel Forge.

Features:
- Block validation with cryptographic hash checking
- Ergotropy threshold verification
- Recursive stability analysis
- Chain continuity validation

Usage:
    from ouroboros_sync_wrapper import OuroborosSyncWrapper, AethelBlock
    
    # Create sync instance
    sync = OuroborosSyncWrapper(ergotropy_threshold=0.5)
    
    # Create and validate blocks
    block = sync.create_block(data="transaction data", ergotropy=0.8)
    result = sync.validate_and_add_block(block)
    
    print(f"Block validated: {result['valid']}")
"""

import sys
import time
from typing import Dict, Any, Optional, List

# Import the Rust module
try:
    import ouroboros_sync
    RUST_MODULE_AVAILABLE = True
except ImportError as e:
    RUST_MODULE_AVAILABLE = False
    print(f"Warning: Rust module 'ouroboros_sync' not available: {e}")
    print("Make sure the module is built and in the Python path.")


class OuroborosSyncWrapper:
    """
    Python wrapper for the Rust-based Ouroboros sync consensus mechanism.
    
    This class provides a high-level interface to the Ouroboros consensus
    mechanism, handling block creation, validation, and chain management.
    """
    
    def __init__(self, 
                 ergotropy_threshold: float = 0.5,
                 stability_window: int = 10,
                 max_recursion_depth: int = 5):
        """
        Initialize the Ouroboros sync wrapper.
        
        Args:
            ergotropy_threshold: Minimum ergotropy value for block acceptance
            stability_window: Number of blocks to consider for stability analysis
            max_recursion_depth: Maximum depth for recursive stability checks
        """
        if not RUST_MODULE_AVAILABLE:
            raise RuntimeError("Rust module 'ouroboros_sync' is not available")
        
        # Create configuration
        self.config = ouroboros_sync.OuroborosSyncConfig(
            ergotropy_threshold=ergotropy_threshold,
            stability_window=stability_window,
            max_recursion_depth=max_recursion_depth
        )
        
        # Create sync instance
        self.sync = ouroboros_sync.OuroborosSync(self.config)
        
        # Track genesis block hash
        self.genesis_hash = "0" * 64
        
    def create_block(self, data: str, ergotropy: float) -> Any:
        """
        Create a new block for the chain.
        
        Args:
            data: Block data (e.g., transaction information)
            ergotropy: Ergotropy value for the block
            
        Returns:
            AethelBlock instance
        """
        # Get index and previous hash
        last_block = self.sync.get_last_block()
        if last_block is None:
            index = 0
            previous_hash = self.genesis_hash
        else:
            index = last_block.index + 1
            previous_hash = last_block.hash
        
        # Create block
        timestamp = time.time()
        block = ouroboros_sync.AethelBlock(
            index=index,
            timestamp=timestamp,
            data=data,
            previous_hash=previous_hash,
            ergotropy=ergotropy
        )
        
        return block
    
    def validate_block(self, block: Any) -> Dict[str, Any]:
        """
        Validate a block without adding it to the chain.
        
        Args:
            block: AethelBlock to validate
            
        Returns:
            Dictionary with validation results
        """
        result = self.sync.validate_block(block)
        return {
            'valid': result.valid,
            'reason': result.reason,
            'ergotropy_check': result.ergotropy_check,
            'stability_check': result.stability_check,
            'hash_check': result.hash_check,
            'chain_check': result.chain_check
        }
    
    def validate_and_add_block(self, block: Any) -> Dict[str, Any]:
        """
        Validate and add a block to the chain.
        
        Args:
            block: AethelBlock to validate and add
            
        Returns:
            Dictionary with validation results and addition status
        """
        validation_result = self.validate_block(block)
        
        if validation_result['valid']:
            added = self.sync.add_block(block)
            validation_result['added'] = added
        else:
            validation_result['added'] = False
        
        return validation_result
    
    def get_chain_length(self) -> int:
        """Get the current length of the blockchain."""
        return self.sync.get_chain_length()
    
    def get_last_block(self) -> Optional[Any]:
        """Get the last block in the chain."""
        return self.sync.get_last_block()
    
    def get_block(self, index: int) -> Optional[Any]:
        """Get a block by index."""
        return self.sync.get_block(index)
    
    def get_stability_stats(self) -> Dict[str, float]:
        """
        Get statistical information about chain stability.
        
        Returns:
            Dictionary with stability statistics (mean, variance, min, max, count)
        """
        return self.sync.get_stability_stats()
    
    def reset(self) -> None:
        """Reset the blockchain to empty state."""
        self.sync.reset()
    
    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current chain state.
        
        Returns:
            Dictionary with chain information
        """
        stats = self.get_stability_stats()
        last_block = self.get_last_block()
        
        return {
            'length': self.get_chain_length(),
            'last_block': {
                'index': last_block.index if last_block else None,
                'hash': last_block.hash if last_block else None,
                'ergotropy': last_block.ergotropy if last_block else None,
                'timestamp': last_block.timestamp if last_block else None,
            } if last_block else None,
            'stability_stats': stats,
            'config': {
                'ergotropy_threshold': self.config.ergotropy_threshold,
                'stability_window': self.config.stability_window,
                'max_recursion_depth': self.config.max_recursion_depth,
            }
        }


def demo_ouroboros_sync():
    """
    Demonstration of the Ouroboros sync consensus mechanism.
    """
    print("=" * 70)
    print("Ouroboros Sync Consensus Mechanism - Demo")
    print("=" * 70)
    
    if not RUST_MODULE_AVAILABLE:
        print("ERROR: Rust module not available. Cannot run demo.")
        return
    
    # Create sync instance
    print("\n1. Creating Ouroboros sync instance...")
    sync = OuroborosSyncWrapper(
        ergotropy_threshold=0.5,
        stability_window=10,
        max_recursion_depth=5
    )
    print(f"   Configuration: {sync.get_chain_info()['config']}")
    
    # Create and add valid blocks
    print("\n2. Creating and validating blocks...")
    for i in range(5):
        ergotropy = 0.5 + (i * 0.1)  # Increasing ergotropy
        block = sync.create_block(
            data=f"Transaction batch {i}",
            ergotropy=ergotropy
        )
        result = sync.validate_and_add_block(block)
        
        print(f"   Block {i}: ergotropy={ergotropy:.2f}, valid={result['valid']}, "
              f"reason='{result['reason']}'")
    
    # Show chain info
    print("\n3. Chain information:")
    chain_info = sync.get_chain_info()
    print(f"   Chain length: {chain_info['length']}")
    print(f"   Last block: index={chain_info['last_block']['index']}, "
          f"ergotropy={chain_info['last_block']['ergotropy']:.2f}")
    
    stats = chain_info['stability_stats']
    print(f"   Stability stats: mean={stats['mean']:.3f}, "
          f"variance={stats['variance']:.3f}")
    
    # Try to add invalid block (low ergotropy)
    print("\n4. Testing validation with low ergotropy block...")
    invalid_block = sync.create_block(
        data="Invalid transaction",
        ergotropy=0.3  # Below threshold
    )
    result = sync.validate_and_add_block(invalid_block)
    print(f"   Result: valid={result['valid']}, reason='{result['reason']}'")
    
    # Try to add block with unstable ergotropy
    print("\n5. Testing recursive stability check...")
    unstable_block = sync.create_block(
        data="Unstable transaction",
        ergotropy=2.0  # Large deviation from recent blocks
    )
    result = sync.validate_and_add_block(unstable_block)
    print(f"   Result: valid={result['valid']}, reason='{result['reason']}'")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demo_ouroboros_sync()
