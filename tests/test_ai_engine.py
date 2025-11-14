"""
Unit tests for Pandora AIOS AI Engine
"""

import unittest
from pandora_aios.ai_engine import AIEngine
from pandora_aios.kernel import Kernel


class TestAIEngine(unittest.TestCase):
    """Test cases for AI Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ai_engine = AIEngine()
        self.kernel = Kernel()
        
    def test_initialization(self):
        """Test AI engine initialization"""
        self.assertTrue(self.ai_engine.enabled)
        self.assertTrue(self.ai_engine.learning_mode)
        self.assertEqual(self.ai_engine.tasks_processed, 0)
        
    def test_optimize_process_priority(self):
        """Test process priority optimization"""
        proc1 = self.kernel.create_process("proc1", ai_assisted=False)
        proc2 = self.kernel.create_process("proc2", ai_assisted=True)
        
        recommendations = self.ai_engine.optimize_process_priority([proc1, proc2])
        
        # AI-assisted process should get higher priority
        self.assertEqual(recommendations[proc1.pid], 0)
        self.assertEqual(recommendations[proc2.pid], 1)
        
    def test_predict_memory_usage(self):
        """Test memory usage prediction"""
        prediction = self.ai_engine.predict_memory_usage(100, 3)
        self.assertIsInstance(prediction, int)
        self.assertGreaterEqual(prediction, 100)
        
    def test_recommend_action(self):
        """Test system action recommendations"""
        system_state = {"memory": {"free": 50, "used": 974, "total": 1024}}
        recommendation = self.ai_engine.recommend_action(system_state)
        self.assertIn("low-priority", recommendation.lower())
        
    def test_analyze_system_health(self):
        """Test system health analysis"""
        processes = []
        memory_info = {"total": 1024, "used": 100, "free": 924}
        
        health = self.ai_engine.analyze_system_health(processes, memory_info)
        
        self.assertIn("status", health)
        self.assertIn("score", health)
        self.assertEqual(health["status"], "healthy")
        self.assertGreaterEqual(health["score"], 70)
        
    def test_enable_disable(self):
        """Test enabling and disabling AI engine"""
        self.ai_engine.disable()
        self.assertFalse(self.ai_engine.enabled)
        
        self.ai_engine.enable()
        self.assertTrue(self.ai_engine.enabled)
        
    def test_get_stats(self):
        """Test getting AI engine statistics"""
        stats = self.ai_engine.get_stats()
        self.assertIn("enabled", stats)
        self.assertIn("learning_mode", stats)
        self.assertIn("tasks_processed", stats)


if __name__ == "__main__":
    unittest.main()
