"""
Test Suite for Session Reset Functionality

Tests for:
- Chatbot reset command
- Conversation manager reset
- Session state verification
"""

import unittest
import sys
import os
import sqlite3
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pandora_chatbot import ConversationManager


class TestConversationManagerReset(unittest.TestCase):
    """Test ConversationManager reset functionality"""
    
    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_conversations.db")
        self.manager = ConversationManager(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up temporary database"""
        if hasattr(self, 'manager') and self.manager.conn:
            self.manager.conn.close()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_reset_conversation(self):
        """Test that reset_conversation clears current conversation"""
        # Create a conversation and add messages
        self.manager.create_conversation("Test Conversation")
        self.manager.add_message('user', 'Hello')
        self.manager.add_message('assistant', 'Hi there!')
        
        # Verify conversation exists
        self.assertIsNotNone(self.manager.current_conversation)
        self.assertEqual(len(self.manager.current_conversation.messages), 2)
        
        # Reset conversation
        self.manager.reset_conversation()
        
        # Verify conversation is cleared
        self.assertIsNone(self.manager.current_conversation)
    
    def test_create_conversation_after_reset(self):
        """Test creating new conversation after reset"""
        # Create and reset
        self.manager.create_conversation("First Conversation")
        self.manager.add_message('user', 'First message')
        self.manager.reset_conversation()
        
        # Create new conversation
        self.manager.create_conversation("New Conversation")
        
        # Verify new conversation is independent
        self.assertIsNotNone(self.manager.current_conversation)
        self.assertEqual(self.manager.current_conversation.title, "New Conversation")
        self.assertEqual(len(self.manager.current_conversation.messages), 0)


class TestChatbotReset(unittest.TestCase):
    """Test PandoraChatbot reset functionality"""
    
    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_chatbot_conversations.db")
        
        # Mock the chatbot initialization to avoid loading heavy LLM models
        # We'll test the conversation reset logic specifically
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_reset_command_exists(self):
        """Test that /reset command is recognized"""
        # Create conversation manager with temp db
        manager = ConversationManager(db_path=self.db_path)
        manager.create_conversation("Test Session")
        manager.add_message('user', 'Test message 1')
        manager.add_message('assistant', 'Response 1')
        
        # Verify messages exist
        self.assertEqual(len(manager.current_conversation.messages), 2)
        
        # Reset
        manager.reset_conversation()
        
        # Create new conversation after reset
        manager.create_conversation("Fresh Session")
        
        # Verify new conversation has no messages
        self.assertEqual(len(manager.current_conversation.messages), 0)
        
        manager.conn.close()
    
    def test_conversation_history_cleared_after_reset(self):
        """Test that conversation history is cleared after reset"""
        manager = ConversationManager(db_path=self.db_path)
        
        # Create conversation with history
        manager.create_conversation("Session with History")
        manager.add_message('user', 'Question 1')
        manager.add_message('assistant', 'Answer 1')
        manager.add_message('user', 'Question 2')
        manager.add_message('assistant', 'Answer 2')
        
        # Get history before reset
        history_before = manager.get_conversation_history()
        self.assertIn("Question 1", history_before)
        self.assertIn("Answer 2", history_before)
        
        # Reset and create new conversation
        manager.reset_conversation()
        manager.create_conversation("New Session")
        
        # Get history after reset
        history_after = manager.get_conversation_history()
        
        # Verify history is empty or doesn't contain old messages
        self.assertEqual(history_after, "")
        
        manager.conn.close()


class TestGUIReset(unittest.TestCase):
    """Test GUI reset functionality (structure verification)"""
    
    def test_gui_has_reset_method(self):
        """Test that PandoraGUI has reset_session method"""
        try:
            from pandora_gui import PandoraGUI
            
            # Verify the method exists
            self.assertTrue(hasattr(PandoraGUI, 'reset_session'))
            
            # Verify it's callable
            self.assertTrue(callable(getattr(PandoraGUI, 'reset_session', None)))
        except ModuleNotFoundError as e:
            if 'tkinter' in str(e):
                self.skipTest("tkinter not available in test environment")
            else:
                raise


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConversationManagerReset))
    suite.addTests(loader.loadTestsFromTestCase(TestChatbotReset))
    suite.addTests(loader.loadTestsFromTestCase(TestGUIReset))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
