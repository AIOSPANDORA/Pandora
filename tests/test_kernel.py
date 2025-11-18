"""
Unit tests for Pandora AIOS Kernel
"""

import unittest
from pandora_aios.kernel import Kernel


class TestKernel(unittest.TestCase):
    """Test cases for Kernel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.kernel = Kernel()
        self.kernel.boot()
        
    def tearDown(self):
        """Clean up after tests"""
        self.kernel.shutdown()
        
    def test_boot(self):
        """Test kernel boot"""
        self.assertTrue(self.kernel.running)
        
    def test_create_process(self):
        """Test process creation"""
        process = self.kernel.create_process("test_process", memory=20)
        self.assertIsNotNone(process)
        self.assertEqual(process.name, "test_process")
        self.assertEqual(process.memory, 20)
        self.assertEqual(self.kernel.memory_used, 20)
        
    def test_create_process_out_of_memory(self):
        """Test process creation with insufficient memory"""
        with self.assertRaises(MemoryError):
            self.kernel.create_process("huge_process", memory=2000)
            
    def test_kill_process(self):
        """Test process termination"""
        process = self.kernel.create_process("test_process", memory=20)
        pid = process.pid
        self.assertTrue(self.kernel.kill_process(pid))
        self.assertEqual(self.kernel.memory_used, 0)
        self.assertIsNone(self.kernel.get_process(pid))
        
    def test_list_processes(self):
        """Test listing processes"""
        self.kernel.create_process("proc1", memory=10)
        self.kernel.create_process("proc2", memory=15)
        processes = self.kernel.list_processes()
        self.assertEqual(len(processes), 2)
        
    def test_memory_info(self):
        """Test memory information"""
        self.kernel.create_process("test", memory=100)
        mem_info = self.kernel.get_memory_info()
        self.assertEqual(mem_info['total'], 1024)
        self.assertEqual(mem_info['used'], 100)
        self.assertEqual(mem_info['free'], 924)
        
    def test_ai_assisted_process(self):
        """Test AI-assisted process creation"""
        process = self.kernel.create_process("ai_proc", ai_assisted=True)
        self.assertTrue(process.ai_assisted)


if __name__ == "__main__":
    unittest.main()
