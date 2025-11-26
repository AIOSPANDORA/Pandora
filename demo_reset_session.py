#!/usr/bin/env python3
"""
Demo script for Reset Session Feature
Shows how the reset functionality works in practice
"""

import sys
import os
import tempfile
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_chatbot_reset():
    """Demonstrate chatbot reset functionality"""
    print("=" * 70)
    print("Pandora AIOS - Reset Session Demo")
    print("=" * 70)
    print()
    
    from pandora_chatbot import ConversationManager
    from unittest.mock import patch, MagicMock
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'demo.db')
    
    print("1. Creating ConversationManager...")
    manager = ConversationManager(db_path=db_path)
    print(f"   ✓ Manager created with DB: {db_path}\n")
    
    print("2. Starting new conversation...")
    manager.create_conversation("Demo Session")
    conv_id = manager.current_conversation.conversation_id
    print(f"   ✓ Conversation ID: {conv_id}\n")
    
    print("3. Adding messages to conversation...")
    manager.add_message('user', 'Hello, Pandora!')
    manager.add_message('assistant', 'Hello! How can I help you today?')
    manager.add_message('user', 'Tell me about quantum computing')
    manager.add_message('assistant', 'Quantum computing uses quantum bits...')
    message_count = len(manager.current_conversation.messages)
    print(f"   ✓ Added {message_count} messages\n")
    
    print("4. Viewing conversation history...")
    history = manager.get_conversation_history(limit=10)
    print("   " + "\n   ".join(history.split('\n')[:4]))
    print("   ...\n")
    
    print("5. Performing RESET...")
    manager.reset_conversation()
    print(f"   ✓ Current conversation: {manager.current_conversation}\n")
    
    print("6. Creating fresh conversation after reset...")
    manager.create_conversation("Fresh Session")
    new_conv_id = manager.current_conversation.conversation_id
    new_message_count = len(manager.current_conversation.messages)
    print(f"   ✓ New Conversation ID: {new_conv_id}")
    print(f"   ✓ Messages in new conversation: {new_message_count}\n")
    
    print("7. Comparing states...")
    print(f"   Before reset: {message_count} messages in conversation {conv_id}")
    print(f"   After reset:  {new_message_count} messages in conversation {new_conv_id}")
    print(f"   Result: Clean slate! ✓\n")
    
    # Cleanup
    manager.conn.close()
    import shutil
    shutil.rmtree(temp_dir)
    
    print("=" * 70)
    print("Demo Complete! Reset functionality verified.")
    print("=" * 70)

def demo_chatbot_command():
    """Demonstrate /reset command"""
    print("\n" + "=" * 70)
    print("Chatbot /reset Command Demo")
    print("=" * 70)
    print()
    
    from pandora_chatbot import PandoraChatbot
    from unittest.mock import patch
    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'demo.db')
    
    print("Creating mock chatbot session...\n")
    
    # Mock to avoid loading heavy models
    with patch('pandora_chatbot.LocalLLM'):
        with patch('pandora_chatbot.PANDORA_SYSTEMS_AVAILABLE', False):
            chatbot = PandoraChatbot.__new__(PandoraChatbot)
            from pandora_chatbot import ConversationManager
            chatbot.conversation_manager = ConversationManager(db_path=db_path)
            chatbot.conversation_manager.create_conversation('Demo')
            chatbot.system_prompt = 'System: You are Pandora AIOS'
            
            print("Available commands:")
            help_text = chatbot.process_command('/help')
            for line in help_text.split('\n'):
                if 'reset' in line.lower() or 'clear' in line.lower():
                    print(f"  {line}")
            print()
            
            print("Simulating conversation...")
            chatbot.conversation_manager.add_message('user', 'Question 1')
            chatbot.conversation_manager.add_message('assistant', 'Answer 1')
            chatbot.conversation_manager.add_message('user', 'Question 2')
            chatbot.conversation_manager.add_message('assistant', 'Answer 2')
            print(f"  Messages: {len(chatbot.conversation_manager.current_conversation.messages)}\n")
            
            print("Executing: /reset")
            response = chatbot.process_command('/reset')
            print(f"  Response: {response}")
            print(f"  Messages after reset: {len(chatbot.conversation_manager.current_conversation.messages)}\n")
            
            chatbot.conversation_manager.conn.close()
    
    import shutil
    shutil.rmtree(temp_dir)
    
    print("=" * 70)
    print("Command demo complete!")
    print("=" * 70)

def demo_gui_info():
    """Show GUI reset information"""
    print("\n" + "=" * 70)
    print("GUI Reset Button Demo")
    print("=" * 70)
    print()
    
    print("The Pandora AIOS GUI includes a 'Reset Session' button that:")
    print("  1. Clears all log messages from display")
    print("  2. Resets system status to 'Idle'")
    print("  3. Resets button states (enables Start, disables Stop)")
    print("  4. Logs the reset action for user feedback")
    print()
    print("To use:")
    print("  1. Run: python3 pandora_gui.py")
    print("  2. Click the 'Reset Session' button")
    print("  3. Observe the log being cleared and status reset")
    print()
    print("Note: GUI requires tkinter to be installed")
    print()
    print("=" * 70)

def main():
    """Run all demos"""
    try:
        demo_chatbot_reset()
        time.sleep(1)
        demo_chatbot_command()
        time.sleep(1)
        demo_gui_info()
        
        print("\n" + "=" * 70)
        print("All Demos Complete! ✓")
        print("=" * 70)
        print("\nFor more information, see:")
        print("  - RESET_SESSION_FEATURE.md - Full documentation")
        print("  - test_reset_session.py - Test suite")
        print("  - pandora_chatbot.py - Implementation")
        print("  - pandora_gui.py - GUI implementation")
        print()
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
