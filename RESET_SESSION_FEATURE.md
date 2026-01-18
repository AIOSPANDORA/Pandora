# Reset Session Feature

## Overview
The Reset Session feature allows users to completely clear and reset their Pandora AIOS session, returning to a clean slate without needing to restart the application.

## Chatbot Reset (`/reset`)

### Usage
In the Pandora AIOS chatbot, type:
```
/reset
```

### What It Does
- Clears all conversation history
- Resets the conversation manager to initial state
- Creates a fresh conversation session
- Reinitializes with the system prompt
- **Note:** Keeps LLM and Pandora systems loaded (no reinitialization overhead)

### Difference from `/clear`
- `/clear` - Creates a new conversation but keeps it in the same session
- `/reset` - Completely resets the entire session to initial state

### Example
```
You: Hello, how are you?
Pandora: I am Pandora AIOS, ready to assist!
You: Tell me about quantum computing
Pandora: [Long response about quantum computing...]
You: /reset
Pandora: Session fully reset. All conversation history cleared. Starting with a clean slate.
You: Hello
Pandora: I am Pandora AIOS, an ethically-driven AI system. How can I assist you today?
```

## GUI Reset

### Usage
In the Pandora AIOS GUI, click the **"Reset Session"** button.

### What It Does
- Clears the system log display
- Resets system status to "Idle"
- Resets button states (enables Start, disables Stop)
- Logs the reset action
- Provides visual feedback to the user

### Use Cases
- Starting a new workflow
- Clearing cluttered logs
- Testing from a clean state
- Privacy (removing conversation history)
- Troubleshooting (resetting to known good state)

## Technical Details

### ConversationManager.reset_conversation()
```python
def reset_conversation(self):
    """Reset current conversation to a fresh state"""
    # Clear in-memory conversation
    self.current_conversation = None
```

### PandoraChatbot.reset_session()
```python
def reset_session(self):
    """Reset the entire chatbot session to initial state"""
    # Reset conversation manager
    self.conversation_manager.reset_conversation()
    
    # Create fresh conversation
    self.conversation_manager.create_conversation("Pandora AIOS Session")
    self.conversation_manager.add_message('system', self.system_prompt)
    
    # Note: LLM and Pandora systems are kept loaded
```

### PandoraGUI.reset_session()
```python
def reset_session(self):
    """Reset the GUI session"""
    # Clear the log display
    # Reset status to Idle
    # Reset button states
    # Log the action
```

## Testing

Run the test suite:
```bash
python3 test_reset_session.py
```

Tests include:
- Conversation manager reset functionality
- Session state verification
- Message clearing
- New conversation creation after reset
- GUI method existence check

## Benefits

1. **Privacy** - Clear sensitive conversation data
2. **Fresh Start** - Begin with a clean slate
3. **Efficiency** - No need to restart the application
4. **Testing** - Easy way to return to initial state
5. **User Control** - Gives users full control over their session

## See Also
- `pandora_chatbot.py` - Chatbot implementation
- `pandora_gui.py` - GUI implementation
- `test_reset_session.py` - Test suite
