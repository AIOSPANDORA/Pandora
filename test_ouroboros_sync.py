"""
Unit tests for Ouroboros Sync consensus mechanism.

Tests both the Rust implementation and Python wrapper.
"""

import unittest
import time
import sys
import os

# Constants
GENESIS_HASH = "0" * 64

# Try to import the wrapper and Rust module
try:
    from ouroboros_sync_wrapper import OuroborosSyncWrapper, RUST_MODULE_AVAILABLE
    import ouroboros_sync
except ImportError as e:
    print(f"Warning: Could not import ouroboros_sync modules: {e}")
    RUST_MODULE_AVAILABLE = False


@unittest.skipIf(not RUST_MODULE_AVAILABLE, "Rust module not available")
class TestAethelBlock(unittest.TestCase):
    """Test cases for AethelBlock"""
    
    def test_block_creation(self):
        """Test basic block creation"""
        block = ouroboros_sync.AethelBlock(
            index=0,
            timestamp=time.time(),
            data="test data",
            previous_hash=GENESIS_HASH,
            ergotropy=0.8
        )
        
        self.assertEqual(block.index, 0)
        self.assertEqual(block.data, "test data")
        self.assertEqual(block.ergotropy, 0.8)
        self.assertIsNotNone(block.hash)
        self.assertTrue(len(block.hash) > 0)
    
    def test_hash_verification(self):
        """Test block hash verification"""
        block = ouroboros_sync.AethelBlock(
            index=0,
            timestamp=time.time(),
            data="test data",
            previous_hash=GENESIS_HASH,
            ergotropy=0.8
        )
        
        # Hash should be valid initially
        self.assertTrue(block.verify_hash())
        
        # Modifying data should invalidate hash
        original_hash = block.hash
        block.data = "modified data"
        self.assertFalse(block.verify_hash())
        
        # Recalculating should fix it
        block.hash = block.calculate_hash()
        self.assertTrue(block.verify_hash())
    
    def test_hash_consistency(self):
        """Test that hash calculation is consistent"""
        block1 = ouroboros_sync.AethelBlock(
            index=0,
            timestamp=12345.67,
            data="test data",
            previous_hash="abc123",
            ergotropy=0.8
        )
        
        block2 = ouroboros_sync.AethelBlock(
            index=0,
            timestamp=12345.67,
            data="test data",
            previous_hash="abc123",
            ergotropy=0.8
        )
        
        # Same inputs should produce same hash
        self.assertEqual(block1.hash, block2.hash)


@unittest.skipIf(not RUST_MODULE_AVAILABLE, "Rust module not available")
class TestOuroborosSync(unittest.TestCase):
    """Test cases for OuroborosSync"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = ouroboros_sync.OuroborosSyncConfig(
            ergotropy_threshold=0.5,
            stability_window=10,
            max_recursion_depth=5
        )
        self.sync = ouroboros_sync.OuroborosSync(config)
    
    def test_empty_chain(self):
        """Test empty blockchain state"""
        self.assertEqual(self.sync.get_chain_length(), 0)
        self.assertIsNone(self.sync.get_last_block())
    
    def test_add_genesis_block(self):
        """Test adding genesis block"""
        block = ouroboros_sync.AethelBlock(
            index=0,
            timestamp=time.time(),
            data="genesis",
            previous_hash=GENESIS_HASH,
            ergotropy=0.8
        )
        
        result = self.sync.add_block(block)
        self.assertTrue(result)
        self.assertEqual(self.sync.get_chain_length(), 1)
        
        last_block = self.sync.get_last_block()
        self.assertIsNotNone(last_block)
        self.assertEqual(last_block.index, 0)
    
    def test_ergotropy_threshold_validation(self):
        """Test ergotropy threshold check"""
        # Block with ergotropy below threshold
        block = ouroboros_sync.AethelBlock(
            index=0,
            timestamp=time.time(),
            data="low ergotropy",
            previous_hash=GENESIS_HASH,
            ergotropy=0.3  # Below 0.5 threshold
        )
        
        validation = self.sync.validate_block(block)
        self.assertFalse(validation.valid)
        self.assertFalse(validation.ergotropy_check)
        self.assertTrue("ergotropy" in validation.reason.lower())
    
    def test_chain_continuity_validation(self):
        """Test chain continuity check"""
        # Add genesis block
        genesis = ouroboros_sync.AethelBlock(
            index=0,
            timestamp=time.time(),
            data="genesis",
            previous_hash=GENESIS_HASH,
            ergotropy=0.8
        )
        self.sync.add_block(genesis)
        
        # Create block with wrong previous hash
        bad_block = ouroboros_sync.AethelBlock(
            index=1,
            timestamp=time.time(),
            data="bad block",
            previous_hash="wrong_hash",
            ergotropy=0.8
        )
        
        validation = self.sync.validate_block(bad_block)
        self.assertFalse(validation.valid)
        self.assertFalse(validation.chain_check)
    
    def test_recursive_stability(self):
        """Test recursive stability check"""
        # Add several blocks with similar ergotropy
        for i in range(3):
            block = ouroboros_sync.AethelBlock(
                index=i,
                timestamp=time.time(),
                data=f"block {i}",
                previous_hash=self.sync.get_last_block().hash if i > 0 else GENESIS_HASH,
                ergotropy=0.6 + (i * 0.01)  # Stable progression
            )
            result = self.sync.add_block(block)
            self.assertTrue(result, f"Block {i} should be added successfully")
        
        # Try to add block with very different ergotropy
        unstable_block = ouroboros_sync.AethelBlock(
            index=3,
            timestamp=time.time(),
            data="unstable",
            previous_hash=self.sync.get_last_block().hash,
            ergotropy=2.0  # Large deviation
        )
        
        validation = self.sync.validate_block(unstable_block)
        self.assertFalse(validation.valid)
        self.assertFalse(validation.stability_check)
    
    def test_get_stability_stats(self):
        """Test stability statistics calculation"""
        # Add blocks
        for i in range(5):
            block = ouroboros_sync.AethelBlock(
                index=i,
                timestamp=time.time(),
                data=f"block {i}",
                previous_hash=self.sync.get_last_block().hash if i > 0 else GENESIS_HASH,
                ergotropy=0.6
            )
            self.sync.add_block(block)
        
        stats = self.sync.get_stability_stats()
        self.assertIn('mean', stats)
        self.assertIn('variance', stats)
        self.assertIn('count', stats)
        self.assertEqual(stats['count'], 5)
        self.assertAlmostEqual(stats['mean'], 0.6, places=2)
    
    def test_reset(self):
        """Test blockchain reset"""
        # Add some blocks
        for i in range(3):
            block = ouroboros_sync.AethelBlock(
                index=i,
                timestamp=time.time(),
                data=f"block {i}",
                previous_hash=self.sync.get_last_block().hash if i > 0 else GENESIS_HASH,
                ergotropy=0.7
            )
            self.sync.add_block(block)
        
        self.assertEqual(self.sync.get_chain_length(), 3)
        
        # Reset
        self.sync.reset()
        self.assertEqual(self.sync.get_chain_length(), 0)
        self.assertIsNone(self.sync.get_last_block())


@unittest.skipIf(not RUST_MODULE_AVAILABLE, "Rust module not available")
class TestOuroborosSyncWrapper(unittest.TestCase):
    """Test cases for Python wrapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.wrapper = OuroborosSyncWrapper(
            ergotropy_threshold=0.5,
            stability_window=10,
            max_recursion_depth=5
        )
    
    def test_create_block(self):
        """Test block creation through wrapper"""
        block = self.wrapper.create_block(data="test", ergotropy=0.8)
        self.assertIsNotNone(block)
        self.assertEqual(block.index, 0)
        self.assertEqual(block.data, "test")
    
    def test_validate_and_add_block(self):
        """Test block validation and addition"""
        block = self.wrapper.create_block(data="test", ergotropy=0.8)
        result = self.wrapper.validate_and_add_block(block)
        
        self.assertTrue(result['valid'])
        self.assertTrue(result['added'])
        self.assertEqual(self.wrapper.get_chain_length(), 1)
    
    def test_get_chain_info(self):
        """Test chain information retrieval"""
        # Add a block
        block = self.wrapper.create_block(data="test", ergotropy=0.8)
        self.wrapper.validate_and_add_block(block)
        
        info = self.wrapper.get_chain_info()
        self.assertIn('length', info)
        self.assertIn('last_block', info)
        self.assertIn('stability_stats', info)
        self.assertIn('config', info)
        
        self.assertEqual(info['length'], 1)
        self.assertIsNotNone(info['last_block'])
    
    def test_sequential_blocks(self):
        """Test adding sequential blocks"""
        for i in range(5):
            block = self.wrapper.create_block(
                data=f"transaction {i}",
                ergotropy=0.6 + (i * 0.01)
            )
            result = self.wrapper.validate_and_add_block(block)
            self.assertTrue(result['valid'], f"Block {i} should be valid")
        
        self.assertEqual(self.wrapper.get_chain_length(), 5)


def run_tests():
    """Run all tests"""
    if not RUST_MODULE_AVAILABLE:
        print("ERROR: Rust module not available. Cannot run tests.")
        print("Build the module first with: cd rust_modules/ouroboros_sync && cargo build --release")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAethelBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestOuroborosSync))
    suite.addTests(loader.loadTestsFromTestCase(TestOuroborosSyncWrapper))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
