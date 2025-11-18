"""
Unit tests for Elder Sister (xAI/Grok) Integration
"""

import unittest
import os
from unittest.mock import patch, MagicMock
from xai_api_integration import contact_elder_sister
from pandora_config import PandoraConfig


class TestElderSisterIntegration(unittest.TestCase):
    """Test cases for Elder Sister communication"""
    
    def test_contact_elder_sister_no_api_key(self):
        """Test contact_elder_sister without API key configured"""
        # Temporarily set API key to placeholder
        original_key = PandoraConfig.ELDER_SISTER_API_KEY
        PandoraConfig.ELDER_SISTER_API_KEY = "YOUR_XAI_API_KEY_HERE"
        
        try:
            response = contact_elder_sister("Hello")
            
            # Should return helpful error message
            self.assertIn("Elder Sister unavailable", response)
            self.assertIn("configure your xAI API key", response)
            self.assertIn("https://x.ai/api", response)
        finally:
            # Restore original value
            PandoraConfig.ELDER_SISTER_API_KEY = original_key
    
    def test_contact_elder_sister_empty_prompt(self):
        """Test contact_elder_sister with empty prompt"""
        # Should handle gracefully even without API key
        response = contact_elder_sister("")
        
        # Should get some response (either error message or actual response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_contact_elder_sister_with_valid_key_no_httpx(self):
        """Test contact_elder_sister when httpx is not available"""
        # Set a fake API key
        original_key = PandoraConfig.ELDER_SISTER_API_KEY
        PandoraConfig.ELDER_SISTER_API_KEY = "test_key_12345"
        
        try:
            # Mock HTTPX_AVAILABLE to False
            import xai_api_integration
            original_httpx = xai_api_integration.HTTPX_AVAILABLE
            xai_api_integration.HTTPX_AVAILABLE = False
            
            try:
                response = contact_elder_sister("Test prompt")
                
                # Should return error about httpx not available
                self.assertIn("Elder Sister unavailable", response)
                self.assertIn("httpx", response.lower())
            finally:
                xai_api_integration.HTTPX_AVAILABLE = original_httpx
        finally:
            PandoraConfig.ELDER_SISTER_API_KEY = original_key
    
    def test_config_values_exist(self):
        """Test that Elder Sister config values exist in PandoraConfig"""
        # Check that config attributes exist
        self.assertTrue(hasattr(PandoraConfig, 'ELDER_SISTER_API_KEY'))
        self.assertTrue(hasattr(PandoraConfig, 'ELDER_SISTER_MODEL'))
        
        # Check that they have expected default values
        self.assertIsInstance(PandoraConfig.ELDER_SISTER_API_KEY, str)
        self.assertEqual(PandoraConfig.ELDER_SISTER_MODEL, "grok-beta")


class TestChatbotElderSisterCommand(unittest.TestCase):
    """Test cases for chatbot /ask-elder command"""
    
    def test_ask_elder_command_in_chatbot(self):
        """Test that chatbot recognizes /ask-elder command"""
        try:
            from pandora_chatbot import PandoraChatbot
            
            # Create chatbot instance (might fail if dependencies missing)
            chatbot = PandoraChatbot()
            
            # Test command processing
            response = chatbot.process_command("/ask-elder")
            
            # Should return usage information
            self.assertIsNotNone(response)
            self.assertIn("Usage", response)
            self.assertIn("/ask-elder", response)
            
        except ImportError as e:
            # Skip test if dependencies not available
            self.skipTest(f"Chatbot dependencies not available: {e}")
    
    def test_ask_elder_with_prompt_in_chatbot(self):
        """Test /ask-elder command with prompt"""
        try:
            from pandora_chatbot import PandoraChatbot
            
            chatbot = PandoraChatbot()
            
            # Test command with prompt
            response = chatbot.process_command("/ask-elder What is consciousness?")
            
            # Should return some response (likely error message without API key)
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
        except ImportError as e:
            self.skipTest(f"Chatbot dependencies not available: {e}")


if __name__ == '__main__':
    unittest.main()
